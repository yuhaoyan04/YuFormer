#!/usr/bin/env python3
"""Independently hash and close the exact eight-train/nine-validation E3 roster."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[3]
AUDITOR_PATH = ROOT / "pretrain/scripts/audit_token_stream_release.py"
SPEC = importlib.util.spec_from_file_location("token_stream_auditor", AUDITOR_PATH)
AUDITOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(AUDITOR)
DOMAINS = ("web", "code", "math", "chinese", "academic", "books",
           "encyclopedia", "qa", "multilingual")
MATRIX = ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(orders_dir: Path, train_streams: Path, val_streams: Path,
          base_release_path: Path, tokenizer_path: Path, gate_path: Path) -> dict[str, Any]:
    derived = json.loads((orders_dir / "manifest.json").read_text(encoding="utf-8"))
    base = json.loads(base_release_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["_gate_sha256"] = AUDITOR.sha256_file(gate_path)
    tokenizer_sha = AUDITOR.sha256_file(tokenizer_path)
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    expected_arms = set(matrix.get("arms", {}))
    if derived.get("status") != "PASS" or base.get("status") != "PASS":
        raise ValueError("derived or base release is not PASS")
    if (set(derived.get("outputs", {})) != expected_arms or len(expected_arms) != 8
            or derived.get("matrix_sha256") != AUDITOR.sha256_file(MATRIX)):
        raise ValueError("derived E3 release is not bound to the exact matrix roster")
    if derived.get("snapshot_id") != base.get("snapshot_id") or gate.get("snapshot_id") != base.get("snapshot_id"):
        raise ValueError("data/tokenizer snapshot identities differ")
    if gate.get("tokenizer_sha256") != tokenizer_sha:
        raise ValueError("tokenizer gate does not bind tokenizer")
    rows = []
    for name, output in sorted(derived["outputs"].items()):
        rows.append(AUDITOR.audit_one(
            stream_dir=train_streams, name=name, order_path=orders_dir / output["file"],
            expected_docs=int(output["docs"]), expected_reference_tokens=int(output["total_tokens"]),
            snapshot=derived, tokenizer_sha256=tokenizer_sha, tokenizer_gate=gate,
        ))
    val_dir = ROOT / "data/processed/val_sets"
    for order_path in sorted(val_dir.glob("*.parquet")):
        docs, text_tokens = AUDITOR.selected_val_contract(order_path, 20_000_000)
        rows.append(AUDITOR.audit_one(
            stream_dir=val_streams, name=f"val_{order_path.stem}", order_path=order_path,
            expected_docs=docs, expected_reference_tokens=text_tokens,
            snapshot=base, tokenizer_sha256=tokenizer_sha, tokenizer_gate=gate,
        ))
    expected_val = {f"val_{domain}" for domain in DOMAINS}
    expected = set(derived["outputs"]) | expected_val
    train_meta = {path.name.removesuffix(".meta.json") for path in train_streams.glob("*.meta.json")}
    train_binary = {path.name.removesuffix(".tokens.bin") for path in train_streams.glob("*.tokens.bin")}
    val_meta = {path.name.removesuffix(".meta.json") for path in val_streams.glob("val_*.meta.json")}
    val_binary = {path.name.removesuffix(".tokens.bin") for path in val_streams.glob("val_*.tokens.bin")}
    expected_train = set(derived["outputs"])
    val_order_roster = {f"val_{path.stem}" for path in val_dir.glob("*.parquet")}
    published = train_meta | train_binary | val_meta | val_binary
    roster_exact = (
        train_meta == expected_train == train_binary
        and val_meta == expected_val == val_binary
        and val_order_roster == expected_val
        and len(expected_train) == 8
    )
    return {
        "schema_version": 2, "audit": "e3_formal_token_stream_release",
        "status": "PASS" if roster_exact and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "dataset_id": base.get("dataset_id"), "snapshot_id": base.get("snapshot_id"),
        "release_name": derived.get("release_name"), "tokenizer_sha256": tokenizer_sha,
        "roster_exact": roster_exact, "expected_streams": sorted(expected),
        "published_streams": sorted(published),
        "published_meta": sorted(train_meta | val_meta),
        "published_binaries": sorted(train_binary | val_binary),
        "streams": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=Path, default=ROOT / "data/experiments/e3_data_mix/orders")
    parser.add_argument("--train-streams", type=Path,
                        default=ROOT / "data/token_streams/e3_20261021_yf64k")
    parser.add_argument("--val-streams", type=Path,
                        default=ROOT / "data/token_streams/m3_20261020_yf64k")
    parser.add_argument("--base-release", type=Path,
                        default=ROOT / "data/processed/releases/mix_m3_e1_100b-20261019.release.json")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "data/TOKENIZER_READY.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/audits/e3_token_streams_20261021.json")
    args = parser.parse_args()
    result = audit(args.orders, args.train_streams, args.val_streams,
                   args.base_release, args.tokenizer, args.gate)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "streams": len(result["streams"]),
                      "roster_exact": result["roster_exact"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
