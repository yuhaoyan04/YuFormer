#!/usr/bin/env python3
"""Audit a trained tokenizer and assign an immutable artifact identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from tokenizers import Tokenizer  # noqa: E402
from yuformer_tokenizer.special_tokens import (  # noqa: E402
    DEFAULT_SPECIAL_TOKENS,
    UNK_TOKEN,
    build_chat_template,
)


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(tokenizer_dir: Path) -> dict[str, Any]:
    required = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "train_meta.json"]
    missing = [name for name in required if not (tokenizer_dir / name).is_file()]
    if missing:
        return {"status": "FAIL", "tokenizer_id": None, "missing": missing, "checks": {}}
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"
    meta_path = tokenizer_dir / "train_meta.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifact_hashes = {
        name: {"bytes": (tokenizer_dir / name).stat().st_size, "sha256": sha256_file(tokenizer_dir / name)}
        for name in required[:-1]
    }
    checks: dict[str, bool] = {
        "vocab_size_exact": tokenizer.get_vocab_size() == int(meta.get("actual_vocab_size", -1)),
        "config_vocab_size_exact": tokenizer.get_vocab_size() == int(config.get("vocab_size", -1)),
        "special_token_roster_exact": config.get("special_tokens") == DEFAULT_SPECIAL_TOKENS,
        "training_special_tokens_exact": (meta.get("special_tokens") or [])
        == [{"token": token, "id": tokenizer.token_to_id(token)} for token in DEFAULT_SPECIAL_TOKENS],
        "chat_template_exact": config.get("chat_template") == build_chat_template(),
        "chat_template_sha256_exact": config.get("chat_template_sha256")
        == hashlib.sha256(build_chat_template().encode("utf-8")).hexdigest(),
        "training_artifact_hashes_exact": all(
            (meta.get("artifacts") or {}).get(name) == record for name, record in artifact_hashes.items()
        ),
        "run_identity_exact": canonical_sha256(meta.get("training_contract"))
        == meta.get("run_identity_sha256"),
        "complete_corpus": meta.get("complete_corpus") is True and int(meta.get("max_samples", -1)) == 0,
        "observed_corpus_exact": True,
        "corpus_manifests_exact": True,
        "special_tokens_atomic": True,
        "unicode_roundtrip": True,
    }
    corpus_records = meta.get("corpus") or []
    corpus_statuses: list[str | None] = []
    manifest_hashes: list[str] = []
    manifest_totals = {"docs": 0, "utf8_bytes": 0}
    for record in corpus_records:
        manifest_path = Path(record.get("manifest") or "")
        if not manifest_path.is_file() or sha256_file(manifest_path) != record.get("manifest_sha256"):
            checks["corpus_manifests_exact"] = False
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        corpus_statuses.append(payload.get("status"))
        manifest_hashes.append(record["manifest_sha256"])
        totals = payload.get("totals") or {}
        manifest_totals["docs"] += int(totals.get("docs", 0))
        manifest_totals["utf8_bytes"] += int(totals.get("utf8_bytes", 0))
        if payload.get("snapshot_id") != record.get("snapshot_id"):
            checks["corpus_manifests_exact"] = False
    observed = meta.get("observed_corpus") or {}
    checks["observed_corpus_exact"] = (
        int(observed.get("samples", -1)) == manifest_totals["docs"]
        and int(observed.get("utf8_bytes", -1)) == manifest_totals["utf8_bytes"]
    )
    for token in DEFAULT_SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None or tokenizer.encode(token).ids != [token_id]:
            checks["special_tokens_atomic"] = False
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    for sample in ["Hello, 世界!", "emoji 🚀🔥😀", "rare: 𝔄 𠮷 ﷺ ẞ", "def f(x):\n\treturn x + 1"]:
        encoding = tokenizer.encode(sample)
        if unk_id in encoding.ids or tokenizer.decode(encoding.ids) != sample:
            checks["unicode_roundtrip"] = False
    technical_pass = all(checks.values())
    upstream_pass = bool(corpus_statuses) and all(status == "PASS" for status in corpus_statuses)
    identity = {
        "tokenizer_sha256": artifact_hashes["tokenizer.json"]["sha256"],
        "tokenizer_config_sha256": artifact_hashes["tokenizer_config.json"]["sha256"],
        "training_config_sha256": meta.get("config_sha256"),
        "training_run_identity_sha256": meta.get("run_identity_sha256"),
        "corpus_manifest_sha256": sorted(manifest_hashes),
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    status = "PASS" if technical_pass and upstream_pass and meta.get("status") == "PASS" else (
        "BLOCKED" if technical_pass else "FAIL"
    )
    return {
        "schema_version": 1,
        "audit": "tokenizer_artifact",
        "status": status,
        "tokenizer_id": f"yuformer-tokenizer-{identity_sha[:20]}" if status == "PASS" else None,
        "candidate_identity": f"yuformer-tokenizer-candidate-{identity_sha[:20]}",
        "vocab_size": tokenizer.get_vocab_size(),
        "checks": checks,
        "upstream_statuses": corpus_statuses,
        "training_status": meta.get("status"),
        "identity": identity,
        "artifacts": artifact_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.tokenizer)
    atomic_json(args.out, result)
    print(json.dumps({"status": result["status"], "tokenizer_id": result.get("tokenizer_id")}, indent=2))
    return 0 if result["status"] in {"PASS", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
