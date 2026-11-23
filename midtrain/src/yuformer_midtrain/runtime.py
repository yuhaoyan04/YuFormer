"""Audit weight-only stage initialization before the first optimizer update."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from yuformer_ext.runtime import YuFormerTrainer, capture_rng


def tensor_sha256(tensor):
    value=tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


class MidtrainTrainer(YuFormerTrainer):
    @dataclass(kw_only=True,slots=True)
    class Config(YuFormerTrainer.Config):
        pass

    def __init__(self,config):
        super().__init__(config)
        # Trainer.__init__ creates the fresh stage optimizer/scheduler/cursor;
        # upstream train() loads DCP afterwards and then enters batch_generator.
        self._stage_initial_rng=capture_rng()

    def batch_generator(self,dataloader):
        rank=dist.get_rank()
        if not getattr(self,'_stage_initialization_reviewed',False):
            fresh=self.step==0
            weight_digest=None;parameters_checked=0
            if fresh:
                if (any(optimizer.state for optimizer in self.optimizers)
                        or self.lr_schedulers.state_dict()['last_epoch']!=0
                        or dataloader.state_dict()['consumed_samples']!=0 or self.ntokens_seen!=0):
                    raise RuntimeError('new long-context stage inherited previous optimizer/scheduler/cursor state')
                current_rng=capture_rng()
                # Tensor pickling includes storage metadata; compare RNG arrays
                # directly rather than treating pickle bytes as the RNG value.
                before=self._stage_initial_rng
                import numpy as np
                if (before['python']!=current_rng['python']
                        or before['numpy'][0]!=current_rng['numpy'][0]
                        or not np.array_equal(before['numpy'][1],current_rng['numpy'][1])
                        or before['numpy'][2:]!=current_rng['numpy'][2:]
                        or not torch.equal(before['torch_cpu'],current_rng['torch_cpu'])
                        or not torch.equal(before['torch_cuda'],current_rng['torch_cuda'])):
                    raise RuntimeError('weight-only initialization replaced the fresh stage RNG state')
                parent=Path(self.config.checkpoint.initial_load_path)
                checksum=hashlib.sha256()
                # Every model tensor is compared, including tied-state entries.
                # Full tensors are assembled one at a time, bounding GPU memory
                # to one parameter rather than a consolidated model copy.
                with torch.no_grad():
                    for part in self.model_parts:
                        for name,value in sorted(part.state_dict().items()):
                            actual=value.full_tensor() if isinstance(value,DTensor) else value
                            if rank==0:
                                actual_cpu=actual.detach().cpu()
                                expected=torch.empty(actual.shape,dtype=actual.dtype,device='cpu')
                                payload={name:expected}
                                dcp.load(payload,checkpoint_id=parent,no_dist=True)
                                if not torch.equal(actual_cpu,payload[name]):
                                    raise RuntimeError('stage initialization weights differ from actual parent DCP: '+name)
                                checksum.update(name.encode()+b'\0');checksum.update(tensor_sha256(actual_cpu).encode())
                            parameters_checked+=1
                            del actual
                weight_digest=checksum.hexdigest() if rank==0 else None
            receipt={'status':'PASS_STAGE_INITIALIZATION' if fresh else 'PASS_LOCAL_FULL_STATE_RESUME',
                'step':self.step,'scheduler_step':self.lr_schedulers.state_dict()['last_epoch'],
                'consumed_samples':dataloader.state_dict()['consumed_samples'],
                'window_manifest_sha256':dataloader.state_dict()['window_manifest_sha256'],
                'parent_model_state_sha256':weight_digest,'model_state_tensors_compared':parameters_checked,
                'initial_load_model_only':self.config.checkpoint.initial_load_model_only,
                'attempt_id':os.environ['YUFORMER_ATTEMPT_ID'],'rank':rank}
            self._write_receipt(f'stage_transition-rank{rank}.json',receipt)
            dist.barrier()
            self._stage_initialization_reviewed=True
        return super().batch_generator(dataloader)
