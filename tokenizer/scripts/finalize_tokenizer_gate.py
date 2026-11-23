#!/usr/bin/env python3
"""Freeze the selected tokenizer only when E0 evidence satisfies policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATES = {"yf-32k-bpe", "yf-64k-bpe", "yf-128k-bpe"}


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--selected", default="yf-64k-bpe")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metrics = load(args.metrics)
    eval_manifest = load(args.eval_manifest)
    audit = load(args.candidate_audit)
    data_ready = load(args.data_ready)
    candidates = {
        row["name"]: row for row in metrics.get("tokenizers", [])
        if row.get("kind") == "candidate"
    }
    if set(candidates) != EXPECTED_CANDIDATES:
        raise SystemExit(f"candidate roster mismatch: {sorted(candidates)}")
    selected = candidates[args.selected]
    small = candidates["yf-32k-bpe"]
    large = candidates["yf-128k-bpe"]

    selected_macro = selected["aggregate"]["utf8_bytes_per_token_domain_macro"]
    small_macro = small["aggregate"]["utf8_bytes_per_token_domain_macro"]
    selected_tokens = selected["aggregate"]["tokens"]
    small_tokens = small["aggregate"]["tokens"]
    large_tokens = large["aggregate"]["tokens"]
    selected_compute = selected["parameter_economics"]["dense_training_flops_per_utf8_byte_proxy"]
    small_compute = small["parameter_economics"]["dense_training_flops_per_utf8_byte_proxy"]
    large_compute = large["parameter_economics"]["dense_training_flops_per_utf8_byte_proxy"]
    fastest = max(row["aggregate"]["throughput_mb_s_median"] for row in candidates.values())

    evidence_checks = {
        "data_gate_pass": data_ready.get("status") == "PASS",
        "eval_corpus_pass": eval_manifest.get("status") == "PASS",
        "metrics_pass": metrics.get("status") == "PASS",
        "all_candidate_metrics_pass": all(row.get("status") == "PASS" for row in candidates.values()),
        "selected_artifact_audit_pass": audit.get("status") == "PASS",
        "dataset_id_aligned": eval_manifest.get("dataset_id") == data_ready.get("dataset_id"),
        "snapshot_id_aligned": eval_manifest.get("snapshot_id") == data_ready.get("snapshot_id"),
        "eval_manifest_hash_bound": metrics.get("corpus", {}).get("manifest_sha256") == sha256_file(args.eval_manifest),
        "selected_tokenizer_id_aligned": selected.get("tokenizer_id") == audit.get("tokenizer_id"),
        "selected_tokenizer_hash_aligned": selected.get("sha256") == (audit.get("identity") or {}).get("tokenizer_sha256"),
        "selected_zero_unknown": selected["aggregate"]["unknown_tokens"] == 0,
        "selected_exact_roundtrip": selected["aggregate"]["roundtrip_failures"] == 0,
    }
    policy_checks = {
        "compression_gain_vs_32k_at_least_5pct": selected_macro / small_macro - 1 >= 0.05,
        "token_reduction_vs_32k_at_least_5pct": 1 - selected_tokens / small_tokens >= 0.05,
        "compute_per_byte_penalty_vs_32k_at_most_2pct": selected_compute / small_compute - 1 <= 0.02,
        "total_parameters_below_500m": selected["parameter_economics"]["total_parameters"] < 500_000_000,
        "throughput_at_least_85pct_of_fastest_candidate": selected["aggregate"]["throughput_mb_s_median"] >= 0.85 * fastest,
        "128k_compute_penalty_exceeds_token_reduction": (
            large_compute / selected_compute - 1 > 1 - large_tokens / selected_tokens
        ),
        "all_domains_improve_over_32k": all(
            selected["per_domain"][domain]["utf8_bytes_per_token"]
            > small["per_domain"][domain]["utf8_bytes_per_token"]
            for domain in selected["per_domain"]
        ),
    }
    status = "PASS" if all(evidence_checks.values()) and all(policy_checks.values()) else "FAIL"
    identity_contract = {
        "data_gate_id": data_ready.get("gate_id"),
        "eval_corpus_id": eval_manifest.get("eval_corpus_id"),
        "metrics_sha256": sha256_file(args.metrics),
        "tokenizer_id": audit.get("tokenizer_id"),
        "tokenizer_sha256": (audit.get("identity") or {}).get("tokenizer_sha256"),
        "selected": args.selected,
        "policy_checks": policy_checks,
    }
    identity = hashlib.sha256(
        json.dumps(identity_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "gate": "tokenizer_ready",
        "status": status,
        "gate_id": f"yuformer-tokenizer-gate-{identity[:20]}",
        "selected": args.selected,
        "tokenizer_id": audit.get("tokenizer_id"),
        "tokenizer_sha256": (audit.get("identity") or {}).get("tokenizer_sha256"),
        "tokenizer_path": selected.get("path"),
        "vocab_size": selected.get("vocab_size"),
        "dataset_id": data_ready.get("dataset_id"),
        "snapshot_id": data_ready.get("snapshot_id"),
        "eval_corpus_id": eval_manifest.get("eval_corpus_id"),
        "evidence_checks": evidence_checks,
        "policy_checks": policy_checks,
        "decision_metrics": {
            "compression_gain_vs_32k": selected_macro / small_macro - 1,
            "token_reduction_vs_32k": 1 - selected_tokens / small_tokens,
            "compute_per_byte_penalty_vs_32k": selected_compute / small_compute - 1,
            "128k_token_reduction_vs_64k": 1 - large_tokens / selected_tokens,
            "128k_compute_per_byte_penalty_vs_64k": large_compute / selected_compute - 1,
            "selected_total_parameters": selected["parameter_economics"]["total_parameters"],
            "selected_throughput_mb_s": selected["aggregate"]["throughput_mb_s_median"],
        },
        "evidence": {
            "data_ready": {"path": str(args.data_ready), "sha256": sha256_file(args.data_ready)},
            "eval_manifest": {"path": str(args.eval_manifest), "sha256": sha256_file(args.eval_manifest)},
            "metrics": {"path": str(args.metrics), "sha256": sha256_file(args.metrics)},
            "candidate_audit": {"path": str(args.candidate_audit), "sha256": sha256_file(args.candidate_audit)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({"status": status, "gate_id": payload["gate_id"], "selected": args.selected}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
