"""Bind asynchronous staging to the rank's CUDA device before entering a stream."""
from dataclasses import dataclass

import torch
from torch.distributed.checkpoint.staging import DefaultStager
from torch.distributed.checkpoint.staging import StagingOptions
from torchtitan.components.checkpointer.dcp import AsyncMode

from yuformer_ext.checkpoint import YuFormerCheckpointManager


class RankBoundStager(DefaultStager):
    def __init__(self,config,*,device):
        self.stage_device=int(device)
        with torch.cuda.device(self.stage_device):
            super().__init__(config)

    def _stage(self,state_dict,**kwargs):
        # CUDA's current device is thread-local. A fresh executor thread starts
        # on device0 even when the main training thread owns another rank.
        with torch.cuda.device(self.stage_device):
            return super()._stage(state_dict,**kwargs)


class RankBoundCheckpointManager(YuFormerCheckpointManager):
    @dataclass(kw_only=True,slots=True)
    class Config(YuFormerCheckpointManager.Config):
        pass

    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        if self.enable and self.async_mode==AsyncMode.ASYNC_WITH_PINNED_MEM:
            # Upstream leaves stager=None until the first non-final save.
            # Install our instance eagerly so that lazy path cannot bypass it.
            options=self.stager._config if self.stager is not None else StagingOptions(
                use_pinned_memory=True,use_shared_memory=True,
                use_async_staging=True,use_non_blocking_copy=True)
            if self.stager is not None:
                self.stager.close()
            self.stager=RankBoundStager(options,device=torch.cuda.current_device())
            import json,os
            from pathlib import Path
            payload={'checkpoint_manager':type(self).__name__,'stager':type(self.stager).__name__,
                     'stage_device':self.stager.stage_device,'rank':int(os.environ['RANK'])}
            target=Path(kwargs['base_folder'])/f'checkpoint_policy-rank{payload["rank"]}.json'
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload,sort_keys=True)+'\n')
            os.replace(temporary,target)

    def dcp_save(self,*args,**kwargs):
        # CUDA context/stream allocations bypass the PyTorch caching allocator.
        # Release unused 64K activation blocks before starting staging so those
        # external allocations have physical headroom as well.
        torch.cuda.empty_cache()
        return super().dcp_save(*args,**kwargs)
