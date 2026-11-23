from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ingest_profiles.yaml"
)

DEFAULT_TEXT_FIELDS: tuple[str, ...] = ("text",)


class ProfileError(ValueError):
    pass


def load_profiles(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else DEFAULT_PROFILES_PATH
    if not p.is_file():
        raise ProfileError(f"ingest profiles not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        raise ProfileError(f"no profiles defined in {p}")
    for name, prof in profiles.items():
        if not isinstance(prof, dict):
            raise ProfileError(f"profile {name}: not a mapping")
        if not prof.get("domain"):
            raise ProfileError(f"profile {name}: missing domain")
        if not prof.get("extensions"):
            raise ProfileError(f"profile {name}: missing extensions")
        prof.setdefault("text_fields", list(DEFAULT_TEXT_FIELDS))
        prof.setdefault("path_fields", [])
        prof.setdefault("score_fields", [])
        prof.setdefault("lang_fields", [])
        prof.setdefault("exclude_dirs", [])
        prof.setdefault("mode", "text")
    return dict(sorted(profiles.items()))
