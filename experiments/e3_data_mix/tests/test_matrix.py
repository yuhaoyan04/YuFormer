import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/e3_data_mix/scripts/audit_matrix.py"
SPEC = importlib.util.spec_from_file_location("audit_e3_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

MATERIALIZE_SPEC = importlib.util.spec_from_file_location(
    "materialize_e3_orders", ROOT / "experiments/e3_data_mix/scripts/materialize_orders.py"
)
MATERIALIZE = importlib.util.module_from_spec(MATERIALIZE_SPEC)
assert MATERIALIZE_SPEC and MATERIALIZE_SPEC.loader
MATERIALIZE_SPEC.loader.exec_module(MATERIALIZE)

ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "analyze_e3_results", ROOT / "experiments/e3_data_mix/scripts/analyze_results.py"
)
ANALYZE = importlib.util.module_from_spec(ANALYZE_SPEC)
assert ANALYZE_SPEC and ANALYZE_SPEC.loader
ANALYZE_SPEC.loader.exec_module(ANALYZE)

GENERATE_SPEC = importlib.util.spec_from_file_location(
    "generate_e3_configs", ROOT / "experiments/e3_data_mix/scripts/generate_configs.py"
)
GENERATE = importlib.util.module_from_spec(GENERATE_SPEC)
assert GENERATE_SPEC and GENERATE_SPEC.loader
GENERATE_SPEC.loader.exec_module(GENERATE)

TOKENIZE_SPEC = importlib.util.spec_from_file_location(
    "tokenize_e3_orders", ROOT / "experiments/e3_data_mix/scripts/tokenize_orders.py"
)
TOKENIZE = importlib.util.module_from_spec(TOKENIZE_SPEC)
assert TOKENIZE_SPEC and TOKENIZE_SPEC.loader
TOKENIZE_SPEC.loader.exec_module(TOKENIZE)


def test_tokenization_resume_is_bound_to_exact_input_identity(tmp_path):
    binary = tmp_path / "arm.tokens.bin"
    binary.write_bytes(b"\x00\x00\x01\x00")
    meta = {
        "schema_version": 2, "name": "arm", "dtype": "<u2", "bytes": 4,
        "total_tokens": 2, "order_sha256": "order", "order_bytes": 12,
        "data_snapshot_id": "snap", "data_release_name": "release",
        "tokenizer_sha256": "tokenizer", "tokenizer_gate_sha256": "gate",
        "edge_sha256": TOKENIZE.edge_sha256(binary),
    }
    (tmp_path / "arm.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    kwargs = {"order": {"sha256": "order", "size": 12},
              "snapshot": {"snapshot_id": "snap", "release_name": "release"},
              "tokenizer_sha256": "tokenizer", "tokenizer_gate_sha256": "gate"}
    assert TOKENIZE.valid_pair(tmp_path, "arm", **kwargs)
    binary.write_bytes(b"\x02\x00\x01\x00")
    assert not TOKENIZE.valid_pair(tmp_path, "arm", **kwargs)
    binary.write_bytes(b"\x00\x00\x01\x00")
    kwargs["order"] = {"sha256": "changed", "size": 12}
    assert not TOKENIZE.valid_pair(tmp_path, "arm", **kwargs)


def test_e3_matrix_is_exact_normalized_and_capacity_safe():
    result = MODULE.audit(ROOT, ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml")
    assert result["status"] == "PASS"
    assert result["defects"] == []
    assert {row["arm"] for row in result["arms"]} == MODULE.EXPECTED_ARMS
    assert all(sum(row["quotas"].values()) == 10_000_000_000 for row in result["arms"])


def test_e3_training_budget_is_declared_and_optimizer_step_aligned():
    declared = 10_000_000_000
    iters, budget, samples = GENERATE.aligned_matched_budget(
        declared, declared + 10_000_000, 4096, 64
    )
    assert budget == declared // (4096 * 64) * (4096 * 64)
    assert declared - budget < 4096 * 64
    assert samples >= iters * 64
    try:
        GENERATE.aligned_matched_budget(declared, budget - 4096, 4096, 64)
    except ValueError as exc:
        assert "declared matched budget" in str(exc)
    else:
        raise AssertionError("an undersized E3 stream was accepted")


def test_materialize_orders_uses_broad_and_quality_prefixes(tmp_path):
    processed = tmp_path / "processed"
    orders = processed / "orders"
    orders.mkdir(parents=True)
    members_path = processed / "members.parquet"
    source_order = orders / "source.parquet"
    rows = [
        ("00", "a", 0.1, 3), ("02", "a", 0.9, 3), ("04", "a", 0.8, 3),
        ("01", "b", 0.2, 3), ("03", "b", 0.7, 3), ("05", "b", 0.6, 3),
    ]
    pq.write_table(pa.table({
        "doc_id": [row[0] for row in rows], "domain": [row[1] for row in rows],
        "composite": pa.array([row[2] for row in rows], type=pa.float32()),
        "n_tokens": pa.array([row[3] for row in rows], type=pa.uint32()),
    }), members_path)
    by_id = sorted(rows)
    pq.write_table(pa.table({
        "position": pa.array(range(len(rows)), type=pa.int64()),
        "doc_id": [row[0] for row in by_id], "domain": [row[1] for row in by_id],
        "n_tokens": pa.array([row[3] for row in by_id], type=pa.uint32()),
    }), source_order)
    (orders / "source.manifest.json").write_text(
        json.dumps({"order_tables": ["source.parquet"]}), encoding="utf-8"
    )
    (processed / "release.json").write_text(
        json.dumps({"status": "PASS", "snapshot_id": "snapshot"}), encoding="utf-8"
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({
        "source_manifest": "processed/orders/source.manifest.json",
        "source_members": "processed/members.parquet",
        "source_release": "processed/release.json",
        "budget_tokens_per_arm": 12,
        "arms": {
            "broad": {"selection": "deterministic_broad", "weights": {"a": 0.5, "b": 0.5}},
            "quality": {"selection": "quality_prefix", "weights": {"a": 0.5, "b": 0.5}},
        },
    }), encoding="utf-8")
    report = MATERIALIZE.materialize(tmp_path, matrix, tmp_path / "output")
    assert report["status"] == "PASS"
    assert report["outputs"]["broad"]["total_tokens"] == 12
    broad = pq.read_table(tmp_path / "output/broad.parquet")["doc_id"].to_pylist()
    quality = pq.read_table(tmp_path / "output/quality.parquet")["doc_id"].to_pylist()
    assert broad == ["00", "01", "02", "03"]
    assert quality == ["02", "03", "04", "05"]
    assert MATERIALIZE.materialize(tmp_path, matrix, tmp_path / "output")["idempotent_skip"] is True


def test_e3_analysis_requires_all_domains_and_selects_lowest_loss(tmp_path):
    matrix_path = ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    tokens_per_step = 4096 * 4 * 16
    trained_tokens = ANALYZE.screen_budget(ROOT, matrix["budget_tokens_per_arm"]) // tokens_per_step * tokens_per_step
    for index, name in enumerate(sorted(matrix["arms"])):
        run = tmp_path / name
        run.mkdir()
        aggregate = 1.5 + index / 100
        metrics = {"validation_metrics/loss": {"last": aggregate},
                   "throughput(tps)": {"last": 1000.0}}
        metrics.update({f"validation_metrics/domain/{domain}/loss": {"last": aggregate + 0.1}
                        for domain in ANALYZE.DOMAINS})
        (run / "result.json").write_text(json.dumps({
            "status": "PASS", "run_name": name, "identity_sha256": name, "metrics": metrics,
            "hyperparameters": {"tokens_per_step": tokens_per_step,
                                "trained_steps": trained_tokens // tokens_per_step,
                                "trained_tokens": trained_tokens},
        }), encoding="utf-8")
    report = ANALYZE.analyze(matrix_path, tmp_path)
    assert report["status"] == "PASS"
    assert report["selection"]["run_name"] == sorted(matrix["arms"])[0]
    assert report["selection"]["empirical_best_margin_to_runner_up"] > 0
    assert len(report["response_slopes"]) == 5

    for path in tmp_path.glob("*/result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["execution_context"] = {"loss_only": True}
        value["metrics"].pop("throughput(tps)")
        path.write_text(json.dumps(value), encoding="utf-8")
    provisional = ANALYZE.analyze(matrix_path, tmp_path)
    assert provisional["status"] == "PASS_PROVISIONAL"

    best_loss = report["ranking"][0]["validation_loss"]
    baseline_path = tmp_path / "e3_m2_base/result.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["metrics"]["validation_metrics/loss"]["last"] = best_loss + 0.001
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    conservative = ANALYZE.analyze(matrix_path, tmp_path)
    assert conservative["selection"]["run_name"] == "e3_m2_base"
    assert len(conservative["practically_equivalent_arms"]) >= 2

    # An aggregate winner with a severe held-out domain regression is not a
    # production-safe mixture; M2 remains an always-defined safe fallback.
    unsafe_name = conservative["selection"]["empirical_best_run_name"]
    if unsafe_name == "e3_m2_base":
        unsafe_name = "e3_academic_2x"
    unsafe_path = tmp_path / f"{unsafe_name}/result.json"
    unsafe = json.loads(unsafe_path.read_text(encoding="utf-8"))
    unsafe["metrics"]["validation_metrics/loss"]["last"] = best_loss - 0.1
    baseline_web = baseline["metrics"]["validation_metrics/domain/web/loss"]["last"]
    unsafe["metrics"]["validation_metrics/domain/web/loss"]["last"] = baseline_web * 1.10
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
    guarded = ANALYZE.analyze(matrix_path, tmp_path)
    assert unsafe_name in guarded["domain_noninferiority"]["excluded_arms"]
    assert guarded["selection"]["run_name"] != unsafe_name


def test_e3_analysis_rejects_nonfinite_loss_and_unexpected_result(tmp_path):
    matrix_path = ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    tokens_per_step = 4096 * 4 * 16
    trained_tokens = ANALYZE.screen_budget(ROOT, matrix["budget_tokens_per_arm"]) // tokens_per_step * tokens_per_step
    for name in sorted(matrix["arms"]):
        run = tmp_path / name
        run.mkdir()
        metrics = {"validation_metrics/loss": {"last": 1.5},
                   "throughput(tps)": {"last": 1000.0}}
        metrics.update({f"validation_metrics/domain/{domain}/loss": {"last": 1.6}
                        for domain in ANALYZE.DOMAINS})
        if name == sorted(matrix["arms"])[0]:
            metrics["validation_metrics/loss"]["last"] = float("nan")
        (run / "result.json").write_text(json.dumps({
            "status": "PASS", "run_name": name, "identity_sha256": name, "metrics": metrics,
            "hyperparameters": {"tokens_per_step": tokens_per_step,
                                "trained_steps": trained_tokens // tokens_per_step,
                                "trained_tokens": trained_tokens},
        }), encoding="utf-8")
    extra = tmp_path / "retired"
    extra.mkdir()
    (extra / "result.json").write_text(json.dumps({
        "status": "PASS", "run_name": "retired", "identity_sha256": "retired", "metrics": {},
    }), encoding="utf-8")
    report = ANALYZE.analyze(matrix_path, tmp_path)
    assert report["status"] == "BLOCKED"
    assert report["unexpected"][0]["run_name"] == "retired"
    failed = next(row for row in report["runs"] if row["run_name"] == sorted(matrix["arms"])[0])
    assert "aggregate_loss_invalid" in failed["issues"]
