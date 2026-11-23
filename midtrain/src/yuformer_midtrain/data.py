"""TorchTitan adapter for the independently qualified long-window record format."""
from dataclasses import dataclass

import torch
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError

from .window_records import WindowCursor
from .window_release import load_window_release


class WindowDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        release_manifest: str = ''
        release_manifest_sha256: str = ''
        source_root: str = ''
        tokenizer_sha256: str = ''
        samples: int = 0

    def __init__(self, config, *, dp_world_size, dp_rank, tokenizer, seq_len,
                 local_batch_size, snapshot_every_n_steps=None):
        del tokenizer, snapshot_every_n_steps
        records = load_window_release(config.release_manifest,
            expected_sha256=config.release_manifest_sha256, source_root=config.source_root,
            seq_len=seq_len, tokenizer_sha256=config.tokenizer_sha256)
        self.cursor = WindowCursor(records, world_size=dp_world_size, rank=dp_rank,
                                   local_batch_size=local_batch_size, samples=config.samples)

    def __iter__(self):
        while True:
            try:
                inputs, labels = self.cursor.next_batch()
            except StopIteration as exc:
                raise DataloaderExhaustedError(str(exc)) from exc
            yield {'input': torch.from_numpy(inputs)}, torch.from_numpy(labels)

    def state_dict(self):
        return self.cursor.state_dict()

    def load_state_dict(self, state):
        self.cursor.load_state_dict(state)
