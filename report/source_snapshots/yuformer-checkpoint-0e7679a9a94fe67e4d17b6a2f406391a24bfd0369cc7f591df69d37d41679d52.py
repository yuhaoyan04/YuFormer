"""DCP policy with explicit recent and milestone retention."""
from dataclasses import dataclass
from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager


class YuFormerCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        keep_every: int = 0

    def __init__(self, config: Config, **kwargs):
        self.keep_every = config.keep_every
        if self.keep_every < 0:
            raise ValueError("checkpoint milestone interval cannot be negative")
        super().__init__(config=config, **kwargs)

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
