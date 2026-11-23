import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "pretrain" / "src", REPO_ROOT / 'pretrain' / 'scripts', REPO_ROOT / "model" / "src", REPO_ROOT / "experiments" / "_common"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
