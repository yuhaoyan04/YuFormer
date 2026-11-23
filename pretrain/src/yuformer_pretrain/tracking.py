"""experiments/_common/tracker.py 的加载适配（该目录非包，按路径 importlib 加载）。

查找顺序：$YUFORMER_TRACKER_ROOT/tracker.py → 仓库相对路径 experiments/_common/tracker.py
（本文件位于 <repo>/pretrain/src/yuformer_pretrain/tracking.py，parents[3] = <repo>）。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_tracker_module():
    candidates = []
    env_root = os.environ.get("YUFORMER_TRACKER_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "tracker.py")
    candidates.append(Path(__file__).resolve().parents[3] / "experiments" / "_common" / "tracker.py")
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("yuformer_tracker", path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"tracker.py 未找到（试过: {candidates}）")


_tracker_mod = _load_tracker_module()

RunTracker = _tracker_mod.RunTracker
file_sha256 = _tracker_mod.file_sha256
edge_sha256 = _tracker_mod.edge_sha256
canonical_sha256 = _tracker_mod.canonical_sha256
