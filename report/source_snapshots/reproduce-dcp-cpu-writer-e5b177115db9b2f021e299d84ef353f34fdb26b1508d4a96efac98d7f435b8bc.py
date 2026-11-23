"""Compare actual PROCESS DCP CPU writes with CUDA visible, without training."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

ROOT = Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
sys.path[:0] = [str(ROOT/'pretrain/backends/torchtitan'),str(ROOT/'model/src'),str(ROOT/'pretrain/src')]
from yuformer_ext.checkpoint import DurableFileSystemWriter


class ObservedWriter(DurableFileSystemWriter):
    def __init__(self,path,evidence,**kwargs):
        super().__init__(path,**kwargs)
        self.evidence = str(evidence)

    def write_data(self,plan,planner):
        before = torch.cuda.is_initialized()
        result = super().write_data(plan,planner)
        value = dict(pid=os.getpid(),cuda_available=torch.cuda.is_available(),
                     cuda_initialized_before=before,cuda_initialized_after=torch.cuda.is_initialized(),
                     per_thread_copy_ahead=self.per_thread_copy_ahead,
                     actual_writer_type='DurableFileSystemWriter',
                     state_dict_tensors='CPU',async_checkpointer_type='PROCESS')
        Path(self.evidence).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['default','cpu_only'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve original attempts')
    # This standalone probe uses one actual async worker. No model/optimizer training.
    args.output.mkdir(parents=True)
    dist.init_process_group('gloo')
    try:
        rank=dist.get_rank()
        parameters={} if args.mode=='default' else dict(per_thread_copy_ahead=0)
        writer=ObservedWriter(args.output/'checkpoint',args.output/f'worker-rank{rank}.json',**parameters)
        state={'tensor':torch.arange(4096,dtype=torch.float32),'scalar':{'step':4}}
        stager = DefaultStager(StagingOptions(True,True,True,True))
        try:
            response = dcp.async_save(state,storage_writer=writer,process_group=dist.group.WORLD,
                                     async_checkpointer_type=AsyncCheckpointerType.PROCESS,async_stager=stager)
            response.staging_completion.result(timeout=120)
            response.upload_completion.result(timeout=120)
        finally:
            stager.close()
        restored={'tensor':torch.zeros_like(state['tensor']),'scalar':{'step':0}}
        dcp.load(restored,checkpoint_id=args.output/'checkpoint',process_group=dist.group.WORLD)
        if not torch.equal(restored['tensor'],state['tensor']) or restored['scalar']!=state['scalar']:
            raise ValueError('CPU checkpoint roundtrip differs')
        evidence=json.loads((args.output/f'worker-rank{rank}.json').read_bytes())
        if not evidence['cuda_available'] or evidence['cuda_initialized_before']:
            raise ValueError('probe did not begin with CUDA visible and worker CUDA uninitialized')
        expected=args.mode=='default'
        if evidence['cuda_initialized_after'] != expected:
            raise ValueError('actual worker CUDA-context result differs from the hypothesis')
        print(json.dumps(dict(status='PASS_CAUSAL_REPRODUCTION',mode=args.mode,evidence=evidence)),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
