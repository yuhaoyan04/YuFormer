"""Qualify actual DCP publication ordering and distributed persistence on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'pretrain/backends/torchtitan'), str(ROOT/'model/src'),
                str(ROOT/'experiments/e4_curriculum/scripts')]
import copy
import os
from pathlib import Path
import stat

import unittest
from unittest.mock import patch
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from yuformer_ext.checkpoint import DurableFileSystemWriter, YuFormerCheckpointManager
from torchtitan.components.checkpointer.dcp import AsyncMode


def state():
    torch.manual_seed(20261115)
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model(torch.ones(2,8)).sum().backward()
    optimizer.step()
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=123,
                dataloader={'consumed_samples':1024})


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple,list)):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert_equal(a,b)
    else:
        assert left == right


def roundtrip(tmp_path):
    path = tmp_path/'run/checkpoint/step-123'
    events = []
    original_sync, original_rename = os.fsync, os.rename
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append(('dirsync', str(Path(os.readlink(f'/proc/self/fd/{fd}')))))
        return original_sync(fd)
    def rename(source, target, *args, **kwargs):
        if str(target).endswith('/.metadata'):
            events.append(('metadata', str(target)))
        return original_rename(source,target,*args,**kwargs)
    patches = [patch.object(os, "fsync", fsync), patch.object(os, "rename", rename)]
    for item in patches: item.start()
    expected = state()
    dcp.save(expected, storage_writer=DurableFileSystemWriter(path))
    marker = next(i for i,e in enumerate(events) if e[0] == 'metadata')
    assert [e[1] for e in events[:marker] if e[0] == 'dirsync'] == [str(p) for p in (path,*list(path.parents)[:3])]
    assert [e[1] for e in events[marker+1:] if e[0] == 'dirsync'] == [str(p) for p in (path,*list(path.parents)[:3])]
    actual = copy.deepcopy(expected)
    actual['step'] = 0
    actual['dataloader']['consumed_samples'] = 0
    for value in actual['model'].values():
        value.zero_()
    dcp.load(actual, checkpoint_id=path)
    assert_equal(expected,actual)
    for item in reversed(patches): item.stop()


def directory_fault(tmp_path, failure_number):
    previous = tmp_path/'run/checkpoint/step-122'
    dcp.save(state(), storage_writer=DurableFileSystemWriter(previous))
    previous_metadata = (previous/'.metadata').read_bytes()
    target = previous.with_name('step-123')
    original_sync = os.fsync
    calls = 0
    def fsync(fd):
        nonlocal calls
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            calls += 1
            if calls == failure_number:
                raise OSError(5, 'injected_directory_fsync_failure')
        return original_sync(fd)
    patcher = patch.object(os, "fsync", fsync)
    patcher.start()
    future = dcp.async_save(state(), storage_writer=DurableFileSystemWriter(target))
    with unittest.TestCase().assertRaisesRegex(CheckpointException, 'injected_directory_fsync_failure'):
        future.result(timeout=60)
    patcher.stop()
    assert (previous/'.metadata').read_bytes() == previous_metadata
    assert (target/'.metadata').exists() is (failure_number == 5)


def disabling_sync(tmp_path):
    with unittest.TestCase().assertRaisesRegex(ValueError, 'file fsync'):
        DurableFileSystemWriter(tmp_path, sync_files=False)



import argparse
import datetime
import hashlib
import json
import signal
import subprocess
import tempfile
import time
import torch.distributed as dist
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from frozen_training_evidence import publish_checked
from freeze_1t_recipe import source_sha256

def distributed_worker(folder):
    output_folder = folder
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    mesh = init_device_mesh('cpu', (2,))
    original = torch.arange(64, dtype=torch.float32).reshape(8,8)
    actual = distribute_tensor(original, mesh, [Shard(0)])
    manager = object.__new__(YuFormerCheckpointManager)
    manager.pg = dist.group.WORLD
    manager.stager = DefaultStager(StagingOptions(use_pinned_memory=False, use_shared_memory=True,
        use_async_staging=True, use_non_blocking_copy=False))
    tests = []
    try:
        for mode in (AsyncMode.DISABLED, AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
            folder = output_folder/mode.value/'run/checkpoint/step-123'
            if folder.exists():
                raise ValueError('preserve earlier distributed checkpoint attempts')
            state = dict(weights=actual, step=123, dataloader={'consumed_samples':1024})
            result = manager.dcp_save(state, str(folder), mode)
            if mode == AsyncMode.ASYNC:
                result.result(timeout=120)
            elif mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                result.upload_completion.result(timeout=120)
            dist.barrier()
            restored = dict(weights=distribute_tensor(torch.zeros_like(original),mesh,[Shard(0)]),
                            step=0,dataloader={'consumed_samples':0})
            dcp.load(restored, checkpoint_id=folder)
            torch.testing.assert_close(restored['weights'].full_tensor(),original,atol=0,rtol=0)
            assert restored['step'] == 123 and restored['dataloader']['consumed_samples'] == 1024
            assert (folder/'.metadata').is_file()
            tests.append(dict(mode=mode.value, status='PASS_REAL_TWO_RANK_SHARDED_ROUNDTRIP'))
        if rank == 0:
            result = dict(world_size=2, device='cpu', backend='gloo', tests=tests)
            (output_folder/'distributed.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    finally:
        manager.stager.close()
        # This prototype deliberately bypasses the trainer manager constructor.
        manager.enable = False
        dist.destroy_process_group()




def qualify(args):
    actual_backend=ROOT/'pretrain/backends/torchtitan/yuformer_ext/checkpoint.py'
    source_inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),actual_backend)}
    model_source=source_sha256(ROOT)
    result=dict(status='RUNNING',tests=[],production_qualified=False,world_size=2,device='cpu',source_inputs=source_inputs,training_source_sha256=model_source)
    work=args.work_dir or ROOT/'data/experiments/checkpoint_directory_durability_qualification'/str(time.time_ns())
    work.mkdir(parents=True,exist_ok=False)
    try:
        for name,fn in [('actual_dcp_roundtrip_and_publication_order',roundtrip),('pre_metadata_directory_fsync_error',lambda p:directory_fault(p,1)),('post_metadata_directory_fsync_error',lambda p:directory_fault(p,5)),('file_fsync_cannot_be_disabled',disabling_sync)]:
            start=time.monotonic()
            with tempfile.TemporaryDirectory(prefix=name+'-',dir=work) as directory:
                fn(Path(directory))
            result['tests'].append(dict(name=name,status='PASS',seconds=time.monotonic()-start))
        distributed=work/'distributed'
        distributed.mkdir()
        with (work/'distributed.log').open('x') as output:
            child=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__)),'--distributed-worker','--work-dir',str(distributed)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=240)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        if code:raise ValueError('actual distributed CPU checkpoint qualification failed; inspect '+str(work/'distributed.log'))
        distributed_report=json.loads((distributed/'distributed.json').read_bytes())
        if distributed_report['world_size']!=2 or distributed_report['device']!='cpu':raise ValueError('distributed checkpoint test topology differs')
        for test in distributed_report['tests']:
            if test['status']!='PASS_REAL_TWO_RANK_SHARDED_ROUNDTRIP':raise ValueError('actual distributed checkpoint roundtrip failed')
            result['tests'].append(dict(name='two_rank:'+test['mode'],status='PASS'))
        expected={'actual_dcp_roundtrip_and_publication_order','pre_metadata_directory_fsync_error','post_metadata_directory_fsync_error','file_fsync_cannot_be_disabled','two_rank:disabled','two_rank:async','two_rank:async_with_pinned_mem'}
        if len(result['tests'])!=7 or {t['name'] for t in result['tests']}!=expected:raise ValueError('all seven actual checkpoint durability cases are required')
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha for p,sha in source_inputs.items()) or source_sha256(ROOT)!=model_source:raise ValueError('checkpoint qualification source changed')
        result.update(status='PASS_CHECKPOINT_DIRECTORY_DURABILITY',work_dir=str(work.relative_to(ROOT)))
    except BaseException as error:
        result.update(status='FAIL_CHECKPOINT_DIRECTORY_DURABILITY',error=repr(error))
        raise
    finally:
        result['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        publish_checked(args.output,(json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
        print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distributed-worker',action='store_true')
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'data/health/checkpoint-directory-durability-qualification.json')
    args=parser.parse_args()
    if args.distributed_worker:
        if args.work_dir is None:raise ValueError('distributed worker needs its explicit isolated work directory')
        distributed_worker(args.work_dir)
    else:qualify(args)
