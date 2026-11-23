"""DCP policy with explicit recent and milestone retention."""
from dataclasses import dataclass
from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
import os

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torchtitan.components.checkpointer.dcp import AsyncMode
from torchtitan.tools.utils import GarbageCollection


def sync_checkpoint_directories(path):
    path = Path(path).resolve(strict=True)
    for directory in (path, *list(path.parents)[:3]):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class DurableFileSystemWriter(FileSystemWriter):
    def __init__(self, path, **kwargs):
        if kwargs.get('sync_files', True) is not True:
            raise ValueError('durable checkpoints require shard and metadata file fsync')
        kwargs.setdefault('per_thread_copy_ahead', 0)
        super().__init__(path, sync_files=True, **{k:v for k,v in kwargs.items() if k != 'sync_files'})

    def finish(self, metadata, results):
        # Shard names and their parent directories must persist before publishing metadata.
        sync_checkpoint_directories(self.path)
        super().finish(metadata, results)
        sync_checkpoint_directories(self.path)


class YuFormerCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        keep_every: int = 0

    def __init__(self, config: Config, **kwargs):
        self.keep_every = config.keep_every
        if self.keep_every < 0:
            raise ValueError("checkpoint milestone interval cannot be negative")
        super().__init__(config=config, **kwargs)

    def dcp_save(self, state_dict, checkpoint_id, async_mode, enable_garbage_collection=False, to_hf=False):
        if to_hf:
            return super().dcp_save(state_dict, checkpoint_id, async_mode,
                                    enable_garbage_collection=enable_garbage_collection, to_hf=True)
        writer = DurableFileSystemWriter(checkpoint_id)
        if async_mode == AsyncMode.ASYNC:
            result = dcp.async_save(state_dict, checkpoint_id=checkpoint_id, storage_writer=writer, process_group=self.pg)
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            result = dcp.async_save(state_dict, checkpoint_id=checkpoint_id, storage_writer=writer, process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS, async_stager=self.stager)
        else:
            result = dcp.save(state_dict, checkpoint_id=checkpoint_id, storage_writer=writer)
        if enable_garbage_collection:
            GarbageCollection.collect('GC collection invoked by checkpointer.')
        return result

    def _purge_stale_checkpoints(self):
        if not self._should_purge():
            return
        checkpoints = sorted(
            (int(path.name.removeprefix("step-")), path)
            for path in Path(self.folder).glob("step-*")
            if path.is_dir() and path.name.removeprefix("step-").isdigit()
            and (path / ".metadata").is_file()
        )
        for step, path in checkpoints[:-self.keep_latest_k]:
            if self.keep_every and step % self.keep_every == 0:
                continue
            self.purge_queue.put(str(path))
