#!/usr/bin/env python3
"""Full-byte audit of three frozen train streams and nine held-out streams."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
import source_document_binding
SPEC = importlib.util.spec_from_file_location(
    "token_stream_auditor", ROOT / "pretrain/scripts/audit_token_stream_release.py"
)
AUDITOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(AUDITOR)
STAGES = ("warmup", "stable", "anneal")
DOMAINS = ("web", "code", "math", "chinese", "academic", "books",
           "encyclopedia", "qa", "multilingual")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(root: Path) -> dict[str, Any]:
    release_path = root / "data/processed/releases/pretrain-1t-frozen.release.json"
    freeze_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    release_builder = root / "experiments/e4_curriculum/scripts/build_1t_order_release.py"
    base_path = root / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"
    gate_path = root / "data/TOKENIZER_READY.json"
    tokenizer = root / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json"
    release, base, gate = (json.loads(path.read_text(encoding="utf-8"))
                           for path in (release_path, base_path, gate_path))
    gate["_gate_sha256"] = AUDITOR.sha256_file(gate_path)
    tokenizer_sha = AUDITOR.sha256_file(tokenizer)
    if any(value.get("status") != "PASS" for value in (release, base, gate)):
        raise ValueError("release/base/tokenizer gates must PASS")
    from training_data_origin import resolve_data_origin
    resolve_data_origin(root, release_path, freeze_path)
    if release.get("builder_tool_sha256") != AUDITOR.sha256_file(release_builder):
        raise ValueError("derived release is stale for the release builder")
    train_dir = root / "data/token_streams/pretrain-1t"
    val_dir = root / "data/token_streams/m3_20261020_yf64k"
    rows = []
    for name in STAGES:
        stage = release["stages"][name]
        row = AUDITOR.audit_one(
            stream_dir=train_dir, name=name, order_path=root / stage["order"],
            expected_docs=int(stage["selection_docs"]),
            expected_reference_tokens=int(stage["selection_reference_tokens"]),
            snapshot=release, tokenizer_sha256=tokenizer_sha, tokenizer_gate=gate,
        )
        try:
            meta = json.loads((train_dir / f"{name}.meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        verified = source_document_binding.verified_source_document_binding(meta)
        row["checks"]["source_document_ids_verified"] = verified
        stream = train_dir / f"{name}.tokens.bin"
        try:
            stat = stream.stat()
            current = {"bytes": stat.st_size, "device": stat.st_dev,
                       "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
            stream_verified = meta.get("stream_stat_identity") == current
        except OSError:
            stream_verified = False
        row["checks"]["published_stream_identity_verified"] = stream_verified
        if not verified or not stream_verified:
            row["status"] = "FAIL"
        rows.append(row)
    val_orders = root / "data/processed/val_sets"
    for order_path in sorted(val_orders.glob("*.parquet")):
        docs, tokens = AUDITOR.selected_val_contract(order_path, 20_000_000)
        rows.append(AUDITOR.audit_one(
            stream_dir=val_dir, name=f"val_{order_path.stem}", order_path=order_path,
            expected_docs=docs, expected_reference_tokens=tokens, snapshot=base,
            tokenizer_sha256=tokenizer_sha, tokenizer_gate=gate,
        ))
    config = yaml.safe_load((root / "pretrain/configs/pretrain_base.yaml").read_text())
    limits = [int(value) for value in config["data"]["stage_sample_limits"]]
    capacity = {}
    for index, name in enumerate(STAGES):
        row = next(value for value in rows if value["name"] == name)
        samples = (int(row.get("actual_total_tokens_with_eod", 0)) - 1) // int(config["data"]["seq_len"])
        capacity[name] = {"available_samples": samples, "required_samples": limits[index],
                          "pass": samples >= limits[index]}
    train_meta = {path.name.removesuffix(".meta.json") for path in train_dir.glob("*.meta.json")}
    train_binary = {path.name.removesuffix(".tokens.bin") for path in train_dir.glob("*.tokens.bin")}
    train_lengths = {path.name.removesuffix(".doc_lengths.u32")
                     for path in train_dir.glob("*.doc_lengths.u32")}
    val_meta = {path.name.removesuffix(".meta.json") for path in val_dir.glob("val_*.meta.json")}
    val_binary = {path.name.removesuffix(".tokens.bin") for path in val_dir.glob("val_*.tokens.bin")}
    expected_train = set(STAGES)
    expected_val = {f"val_{domain}" for domain in DOMAINS}
    expected = expected_train | expected_val
    roster_exact = (
        train_meta == expected_train == train_binary == train_lengths
        and val_meta == expected_val == val_binary
        and {f"val_{path.stem}" for path in val_orders.glob("*.parquet")} == expected_val
    )
    published = train_meta | train_binary | train_lengths | val_meta | val_binary
    train_lengths_audited = all(
        row["status"] == "PASS"
        and "doc_lengths_sha256_exact" in row.get("checks", {})
        for row in rows if row.get("name") in expected_train
    )
    train_source_rows_verified = all(
        row.get("checks", {}).get("source_document_ids_verified") is True
        for row in rows if row.get("name") in expected_train
    )
    status = ("PASS" if roster_exact and train_lengths_audited and train_source_rows_verified
              and all(row["status"] == "PASS" for row in rows)
              and all(value["pass"] for value in capacity.values()) else "FAIL")
    return {"schema_version": 3, "audit": "pretrain_1t_token_streams", "status": status,
            "audit_tool_sha256": AUDITOR.sha256_file(Path(__file__)),
            "shared_auditor_sha256": AUDITOR.sha256_file(Path(AUDITOR.__file__)),
            "source_binding_tool_sha256": AUDITOR.sha256_file(Path(source_document_binding.__file__)),
            "dataset_id": base["dataset_id"], "snapshot_id": base["snapshot_id"],
            "release_name": release["release_name"], "tokenizer_sha256": tokenizer_sha,
            "frozen_recipe_sha256": release["frozen_recipe_sha256"],
            "roster_exact": roster_exact, "expected_streams": sorted(expected),
            "train_doc_lengths_audited": train_lengths_audited,
            "train_source_rows_verified": train_source_rows_verified,
            "published_streams": sorted(published),
            "published_meta": sorted(train_meta | val_meta),
            "published_binaries": sorted(train_binary | val_binary),
            "published_doc_lengths": sorted(train_lengths),
            "stage_capacity": capacity, "streams": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/audits/pretrain_1t_token_streams.json")
    args = parser.parse_args()
    result = audit(args.root.resolve())
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "streams": len(result["streams"]),
                      "roster_exact": result["roster_exact"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
