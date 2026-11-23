from __future__ import annotations

import importlib.util
import json
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/e4_curriculum/scripts/audit_1t_plan.py"
SPEC = importlib.util.spec_from_file_location("audit_1t_plan", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FREEZE_SCRIPT = ROOT / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_1t_recipe", FREEZE_SCRIPT)
assert FREEZE_SPEC and FREEZE_SPEC.loader
FREEZE = importlib.util.module_from_spec(FREEZE_SPEC)
FREEZE_SPEC.loader.exec_module(FREEZE)

OVERLAP_SCRIPT = ROOT / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
OVERLAP_SPEC = importlib.util.spec_from_file_location("audit_1t_overlap", OVERLAP_SCRIPT)
assert OVERLAP_SPEC and OVERLAP_SPEC.loader
OVERLAP = importlib.util.module_from_spec(OVERLAP_SPEC)
OVERLAP_SPEC.loader.exec_module(OVERLAP)

TOKENIZE_SCRIPT = ROOT / "experiments/e4_curriculum/scripts/tokenize_1t_orders.py"
TOKENIZE_SPEC = importlib.util.spec_from_file_location("tokenize_1t_orders", TOKENIZE_SCRIPT)
assert TOKENIZE_SPEC and TOKENIZE_SPEC.loader
TOKENIZE = importlib.util.module_from_spec(TOKENIZE_SPEC)
TOKENIZE_SPEC.loader.exec_module(TOKENIZE)

ORDER_RUNNER_SCRIPT = ROOT / "experiments/e4_curriculum/scripts/materialize_1t_orders.py"
ORDER_RUNNER_SPEC = importlib.util.spec_from_file_location("materialize_1t_orders", ORDER_RUNNER_SCRIPT)
assert ORDER_RUNNER_SPEC and ORDER_RUNNER_SPEC.loader
ORDER_RUNNER = importlib.util.module_from_spec(ORDER_RUNNER_SPEC)
ORDER_RUNNER_SPEC.loader.exec_module(ORDER_RUNNER)

RELEASE_SCRIPT = ROOT / "experiments/e4_curriculum/scripts/build_1t_order_release.py"
RELEASE_SPEC = importlib.util.spec_from_file_location("build_1t_order_release", RELEASE_SCRIPT)
assert RELEASE_SPEC and RELEASE_SPEC.loader
RELEASE = importlib.util.module_from_spec(RELEASE_SPEC)
RELEASE_SPEC.loader.exec_module(RELEASE)


@pytest.mark.parametrize("publisher,stamp", [(FREEZE.publish_recipe, "frozen_at"),
                                            (RELEASE.publish_release, "created_at")])
def test_identical_publication_preserves_checkpoint_bound_bytes_and_inode(tmp_path, publisher, stamp):
    path = tmp_path / "published.json"
    first = {"status": "PASS", "input_sha256": "original", stamp: "first"}
    publisher(path, first)
    content, inode, modified = path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns
    repeated = publisher(path, dict(first, **{stamp: "retry"}))
    assert repeated[stamp] == "first"
    assert path.read_bytes() == content
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (inode, modified)
    changed = publisher(path, dict(first, input_sha256="changed", **{stamp: "new_generation"}))
    assert changed[stamp] == "new_generation"
    assert path.read_bytes() != content


def test_final_tokenization_storage_preflight_accounts_for_merge_peak(monkeypatch, tmp_path):
    release = {"stages": {
        "warmup": {"target_final_tokens": 50},
        "stable": {"target_final_tokens": 800},
        "anneal": {"target_final_tokens": 150},
    }}
    usage = namedtuple("usage", "total used free")
    # uint16 remaining final output is 1,900 bytes after warmup. The stable
    # NPZ/raw/staging overlap contributes two additional 1,600-byte copies.
    monkeypatch.setattr(TOKENIZE.shutil, "disk_usage", lambda _: usage(10_000, 1_000, 9_000))
    monkeypatch.setattr(TOKENIZE, "STORAGE_RESERVE_BYTES", 100)
    report = TOKENIZE.storage_preflight(
        release, {"vocab_size": 65_536}, {"warmup"}, tmp_path
    )
    assert report["token_width_bytes"] == 2
    assert report["remaining_final_bytes"] == 1_900
    assert report["largest_remaining_stage_bytes"] == 1_600
    assert report["required_free_bytes"] == 5_710
    assert report["status"] == "PASS"

    monkeypatch.setattr(TOKENIZE.shutil, "disk_usage", lambda _: usage(10_000, 4_300, 5_700))
    assert TOKENIZE.storage_preflight(
        release, {"vocab_size": 65_536}, {"warmup"}, tmp_path
    )["status"] == "BLOCKED"


def test_final_tokenization_storage_preflight_uses_uint32_above_64k(monkeypatch, tmp_path):
    release = {"stages": {
        "warmup": {"target_final_tokens": 50},
        "stable": {"target_final_tokens": 800},
        "anneal": {"target_final_tokens": 150},
    }}
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(TOKENIZE.shutil, "disk_usage", lambda _: usage(100_000, 0, 100_000))
    monkeypatch.setattr(TOKENIZE, "STORAGE_RESERVE_BYTES", 0)
    report = TOKENIZE.storage_preflight(release, {"vocab_size": 65_537}, set(), tmp_path)
    assert report["token_width_bytes"] == 4
    assert report["remaining_final_bytes"] == 4_000
    assert report["largest_remaining_stage_bytes"] == 3_200
    assert report["required_free_bytes"] == 11_440


def test_final_tokenization_resume_rejects_stale_order(monkeypatch, tmp_path):
    monkeypatch.setattr(TOKENIZE, "STREAMS", tmp_path)
    (tmp_path / "warmup.tokens.bin").write_bytes(b"\x00\x00\x01\x00")
    meta = {
        "schema_version": 2, "name": "warmup", "dtype": "<u2", "bytes": 4,
        "total_tokens": 2, "order_sha256": "order", "order_bytes": 12,
        "data_snapshot_id": "snap", "data_release_name": "release",
        "tokenizer_sha256": "tokenizer", "tokenizer_gate_sha256": "gate",
        "edge_sha256": TOKENIZE.edge_sha256(tmp_path / "warmup.tokens.bin"),
    }
    (tmp_path / "warmup.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    stage = {"order": "orders/warmup.parquet"}
    release = {"snapshot_id": "snap", "release_name": "release", "artifacts": [
        {"path": "orders/warmup.parquet", "sha256": "order", "bytes": 12}
    ]}
    assert TOKENIZE.valid_pair(tmp_path, "warmup", stage=stage, release=release,
                               tokenizer_sha256="tokenizer", tokenizer_gate_sha256="gate")
    assert not TOKENIZE.valid_pair(
        tmp_path, "warmup", stage=stage, release=release,
        tokenizer_sha256="tokenizer", tokenizer_gate_sha256="gate",
        require_doc_lengths=True,
    )
    lengths = tmp_path / "warmup.doc_lengths.u32"
    np.asarray([2], dtype="<u4").tofile(lengths)
    stream_stat = (tmp_path / "warmup.tokens.bin").stat()
    meta.update({
        "schema_version": 3, "n_docs": 1,
        "stream_stat_identity": {"bytes": stream_stat.st_size, "device": stream_stat.st_dev,
                                 "inode": stream_stat.st_ino, "mtime_ns": stream_stat.st_mtime_ns},
        "source_document_binding": {"schema_version": 4, "expected_doc_ids_verified": True,
                                    "tokenizer_bytes_verified": True, "order_bytes_verified": True, "source_handles_verified": True,
                                    "roster_fingerprint": "a" * 64},
        "doc_lengths": {"file": lengths.name, "dtype": "<u4", "rows": 1,
                        "bytes": 4, "edge_sha256": TOKENIZE.edge_sha256(lengths)},
    })
    (tmp_path / "warmup.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert TOKENIZE.valid_pair(
        tmp_path, "warmup", stage=stage, release=release,
        tokenizer_sha256="tokenizer", tokenizer_gate_sha256="gate",
        require_doc_lengths=True,
    )
    (tmp_path / "warmup.tokens.bin").write_bytes(b"\x02\x00\x01\x00")
    assert not TOKENIZE.valid_pair(tmp_path, "warmup", stage=stage, release=release,
                                   tokenizer_sha256="tokenizer", tokenizer_gate_sha256="gate")
    (tmp_path / "warmup.tokens.bin").write_bytes(b"\x00\x00\x01\x00")
    release["artifacts"][0]["sha256"] = "changed"
    assert not TOKENIZE.valid_pair(tmp_path, "warmup", stage=stage, release=release,
                                   tokenizer_sha256="tokenizer", tokenizer_gate_sha256="gate")


def test_final_order_resume_requires_published_stat_identity(monkeypatch, tmp_path):
    processed = tmp_path / "data/processed"
    orders = processed / "mix_order"
    members_dir = processed / "mix_members"
    orders.mkdir(parents=True)
    members_dir.mkdir()
    order = orders / "mix_stage2.parquet"
    members = members_dir / "mix.parquet"
    order.write_bytes(b"order")
    members.write_bytes(b"members")

    def record(path):
        stat = path.stat()
        return {"path": path.relative_to(processed).as_posix(), "bytes": stat.st_size,
                "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
                "sha256": "unused-by-resume"}

    manifest = {"mix_name": "mix", "config_echo": "echo",
                "members": "mix_members/mix.parquet",
                "order_tables": [order.name],
                "published_artifacts": [record(order), record(members)]}
    (orders / "mix.manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(ORDER_RUNNER, "load_mix_config", lambda path: {"mix_name": "mix"})
    monkeypatch.setattr(ORDER_RUNNER, "config_echo", lambda cfg: "echo")
    valid, _ = ORDER_RUNNER.output_valid(tmp_path, tmp_path / "mix.yaml")
    assert valid
    order.write_bytes(b"changed")
    assert not ORDER_RUNNER.output_valid(tmp_path, tmp_path / "mix.yaml")[0]


def test_order_runner_requires_closed_upstream_snapshot_identity():
    value = {"upstream_snapshot_sha256": "a",
             "upstream_snapshot": {"snapshot_sha256": "a"}}
    assert ORDER_RUNNER.upstream_snapshot_identity(value) == "a"
    value["upstream_snapshot"]["snapshot_sha256"] = "b"
    assert ORDER_RUNNER.upstream_snapshot_identity(value) is None


@pytest.mark.parametrize("suffix", ["summary.json", "parquet", "ids.bin"])
@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_order_runner_rejects_validation_drift_before_materialization(tmp_path, suffix, mutation):
    manifest = tmp_path / ORDER_RUNNER.BASE_MANIFEST.relative_to(ORDER_RUNNER.ROOT)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"upstream_snapshot_sha256": "snapshot",
                                   "upstream_snapshot": {"snapshot_sha256": "snapshot"}}))
    validation = tmp_path / f"data/processed/val_sets/web.{suffix}"
    validation.parent.mkdir(parents=True)
    validation.write_bytes(b"original")
    artifacts = [{"path": str(path.relative_to(tmp_path)), "bytes": path.stat().st_size,
                  "sha256": ORDER_RUNNER.sha256(path)} for path in (manifest, validation)]
    release = tmp_path / ORDER_RUNNER.BASE_RELEASE.relative_to(ORDER_RUNNER.ROOT)
    release.parent.mkdir(parents=True)
    release.write_text(json.dumps({"status": "PASS", "artifacts": artifacts}))
    assert ORDER_RUNNER.base_upstream_snapshot(tmp_path) == "snapshot"
    if mutation == "replace":
        validation.write_bytes(b"modified")
    else:
        validation.unlink()
    with pytest.raises(ValueError, match="base validation artifact drift before S6"):
        ORDER_RUNNER.base_upstream_snapshot(tmp_path)


def test_frozen_release_requires_exact_validation_domain_roster():
    artifacts = [
        {"path": f"data/processed/val_sets/{domain}.parquet"}
        for domain in RELEASE.DOMAINS
    ]
    assert RELEASE.validation_parquet_domains(artifacts) == set(RELEASE.DOMAINS)
    artifacts[-1]["path"] = "data/processed/val_sets/wrong.parquet"
    assert RELEASE.validation_parquet_domains(artifacts) != set(RELEASE.DOMAINS)


def test_frozen_release_closes_upstream_snapshot_identity():
    manifest = {
        "upstream_snapshot_sha256": "snapshot-a",
        "upstream_snapshot": {"snapshot_sha256": "snapshot-a"},
    }
    assert RELEASE.upstream_snapshot_identity(manifest) == "snapshot-a"
    manifest["upstream_snapshot"]["snapshot_sha256"] = "snapshot-b"
    with pytest.raises(ValueError, match="unclosed upstream"):
        RELEASE.upstream_snapshot_identity(manifest)


def test_frozen_release_artifact_binds_hash_and_stat_identity(tmp_path):
    path = tmp_path / "order.parquet"
    path.write_bytes(b"immutable-order")
    record = RELEASE.artifact(tmp_path, path)
    stat = path.stat()
    assert record == {
        "path": "order.parquet", "bytes": stat.st_size,
        "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
        "sha256": RELEASE.sha256(path),
    }


def test_order_release_audit_closes_positions_and_token_sum(tmp_path):
    order = tmp_path / "order.parquet"
    pq.write_table(pa.table({
        "position": pa.array([0, 1, 2], type=pa.int64()),
        "doc_id": ["a" * 32, "b" * 32, "c" * 32], "domain": ["web", "web", "code"],
        "n_tokens": pa.array([2, 3, 5], type=pa.uint32()),
    }), order)
    assert RELEASE.audit_order(order, expected_docs=3, expected_tokens=10) == {
        "docs": 3, "reference_tokens": 10, "positions_contiguous": True,
        "doc_ids_strictly_sorted": True,
    }
    pq.write_table(pa.table({
        "position": pa.array([0, 2], type=pa.int64()),
        "doc_id": ["a" * 32, "b" * 32], "domain": ["web", "code"],
        "n_tokens": pa.array([2, 3], type=pa.uint32()),
    }), order)
    try:
        RELEASE.audit_order(order, expected_docs=2, expected_tokens=5)
    except ValueError as exc:
        assert "positions are not contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous order positions were accepted")


def test_order_release_rejects_non_strict_doc_id_order_before_tokenization(tmp_path):
    order = tmp_path / "order.parquet"
    pq.write_table(pa.table({
        "position": pa.array([0, 1, 2], type=pa.int64()),
        "doc_id": pa.array(["b" * 32, "a" * 32, "c" * 32], type=pa.large_string()),
        "domain": ["web", "web", "code"],
        "n_tokens": pa.array([2, 3, 5], type=pa.uint32()),
    }), order)
    try:
        RELEASE.audit_order(order, expected_docs=3, expected_tokens=10)
    except ValueError as exc:
        assert "strict-sorted" in str(exc)
    else:
        raise AssertionError("non-sorted order doc_ids were accepted")


def test_current_1t_plan_has_static_capacity_and_accounting_closure():
    report = MODULE.audit(
        ROOT,
        ROOT / "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml",
        ROOT / "experiments/e4_curriculum/tests/fixtures/base_s6_capacity.json",
    )
    assert report["status"] == "PASS_STATIC", report["errors"]
    budget = json.loads((ROOT/'experiments/e4_curriculum/results/effective_total_training_budget.json').read_bytes())
    assert report['checks']['pretrain_budget']['stage_tokens'] == budget['stages'][0]['tokens']
    assert report['checks']['global_training_budget']['actual_total_tokens'] == budget['actual_total_tokens']
    assert {
        arm["extension_tokens"]
        for arm in report["checks"]["context_scaling"]["matched_token_arms"].values()
    } == {sum(stage['tokens'] for stage in budget['stages'][1:])}
    assert all(
        domain["fits_pool"]
        for stage in report["stages"]
        for domain in stage["domains"].values()
    )
    assert all(
        domain["fits_measured_top10"] is not False
        for stage in report["stages"]
        for domain in stage["domains"].values()
    )
    exact_domains = {
        domain_name
        for stage in report["stages"]
        for domain_name, domain in stage["domains"].items()
        if domain["observed_candidate_is_exact_top10"]
    }
    assert exact_domains == {"web", "code", "math", "academic", "multilingual"}


def test_static_capacity_binds_candidate_counts_to_the_baseline_quality_band(monkeypatch):
    ledger_path = ROOT / "experiments/e4_curriculum/tests/fixtures/base_s6_capacity.json"
    ledger = json.loads(ledger_path.read_text())
    for row in ledger["per_domain"].values():
        row["candidate_tokens"] *= 10
    load = MODULE._load
    monkeypatch.setattr(MODULE, "_load", lambda path: ledger if path == ledger_path else load(path))
    report = MODULE.audit(ROOT, ROOT / "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml", ledger_path)
    baseline = json.loads((ROOT / "data/processed/mix_order/mix_m3_e1_100b.manifest.json").read_text())
    for stage in report["stages"]:
        for domain, row in stage["domains"].items():
            assert row["observed_candidate_tokens"] == baseline["selected"][domain]["candidate_tokens"]


def test_scale_architecture_preserves_e2_switches_and_tracks_target():
    source = {
        "hidden_size": 1024, "num_layers": 32, "intermediate_size": 2816,
        "num_heads": 16, "num_kv_heads": 8, "attention_every": 6,
        "attention_output_gate": False, "qk_norm": True,
    }
    target = 600_000_000
    scaled = FREEZE.scale_architecture(source, target)
    assert scaled["attention_output_gate"] is False
    assert scaled["num_heads"] * 64 == scaled["hidden_size"]
    assert scaled["num_heads"] == 2 * scaled["num_kv_heads"]
    assert abs(FREEZE.non_embedding_params(scaled) - target) / target < 0.15


def test_frozen_batch_recomputes_optimizer_aligned_wsd_boundaries():
    targets = [50_000_000_000, 800_000_000_000, 150_000_000_000]
    for tokens_per_step in (1_048_576, 2_097_152, 4_194_304):
        steps = FREEZE.optimizer_aligned_stage_steps(targets, tokens_per_step)
        assert sum(steps) == sum(targets) // tokens_per_step
        assert all(abs(step * tokens_per_step - target) < tokens_per_step
                   for step, target in zip(steps, targets, strict=True))


def test_frozen_learning_rate_applies_batch_pilot_factor():
    resolution = {"fit": {"log_intercept": 0.0, "exponent": -0.5},
                  "batch_lr_factor": 1.75}
    value, factor = FREEZE.resolved_learning_rate(resolution, 400_000_000)
    assert factor == 1.75
    assert abs(value - 1.75 / (400_000_000 ** 0.5)) < 1e-15


def test_validation_ratios_requires_exact_nine_domain_roster():
    audit = {"streams": [
        {"name": f"val_{domain}", "status": "PASS", "expected_reference_tokens": 100,
         "actual_total_tokens_with_eod": 91, "expected_docs": 1}
        for domain in FREEZE.DOMAINS
    ]}
    ratios = FREEZE.validation_ratios(audit)
    assert ratios == {domain: 0.9 for domain in FREEZE.DOMAINS}


def test_freeze_rejects_stale_experiment_identity_snapshot(tmp_path):
    evidence = tmp_path / "result.json"
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    audit = {
        "evidence_snapshot": {"result.json": FREEZE.sha256_bytes(evidence.read_bytes())},
        "evidence_snapshot_count": 1,
    }
    FREEZE.verify_identity_audit_snapshot(tmp_path, audit)
    evidence.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stale experiment identity evidence"):
        FREEZE.verify_identity_audit_snapshot(tmp_path, audit)


def test_freeze_rejects_input_changed_after_capture(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text('{"status":"PASS"}\n', encoding="utf-8")
    _, digest = FREEZE.load_json_bound(selection)
    inputs = {"selection.json": digest}
    FREEZE.verify_captured_inputs(tmp_path, inputs)
    selection.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed during generation"):
        FREEZE.verify_captured_inputs(tmp_path, inputs)


def test_overlap_extraction_and_native_merge(tmp_path):
    stage_rows = {
        "warmup": [("00", 2), ("02", 3)],
        "stable": [("00", 2), ("01", 5), ("03", 7)],
        "anneal": [("00", 2), ("03", 7), ("04", 11)],
    }
    raw_paths = []
    for stage, rows in stage_rows.items():
        # Production ids are fixed 32-byte hex strings.
        ids = [value.ljust(32, "0") for value, _ in rows]
        order = tmp_path / f"{stage}.parquet"
        pq.write_table(pa.table({"doc_id": pa.array(ids, type=pa.large_string()),
                                 "n_tokens": pa.array([value for _, value in rows], type=pa.uint32())}),
                       order)
        raw = tmp_path / f"{stage}.records"
        meta = OVERLAP.extract_sorted(order, raw, stage)
        assert meta["rows"] == len(rows)
        assert meta["sha256"] == OVERLAP.sha256(raw)
        raw_paths.append(raw)
    binary = tmp_path / "merge"
    subprocess.run(["cc", "-O3", "-std=c11",
                    str(ROOT / "experiments/e4_curriculum/scripts/overlap_merge.c"),
                    "-o", str(binary)], check=True)
    result = subprocess.run([str(binary), *map(str, raw_paths)], check=True,
                            capture_output=True, text=True)
    metrics = json.loads(result.stdout)
    assert metrics["unique_docs"] == 5
    assert metrics["occurrence_docs"] == 8
    assert metrics["triple_docs"] == 1
    assert metrics["token_mismatches"] == 0


def test_exact_consumed_overlap_uses_target_prefix_and_partial_documents(tmp_path):
    # Position order lengths [3,4,2]; targets [1,7) consume doc segments
    # [1,3), [0,4), excluding the third document.
    ids = ["00".ljust(32, "0"), "01".ljust(32, "0"), "02".ljust(32, "0")]
    order = tmp_path / "order.parquet"
    pq.write_table(pa.table({
        "doc_id": pa.array(ids, type=pa.large_string()),
        "position": pa.array([0, 1, 2], type=pa.int64()),
        "domain": ["web", "code", "math"],
    }), order)
    lengths = tmp_path / "lengths.u32"
    np.asarray([3, 4, 2], dtype="<u4").tofile(lengths)
    output = tmp_path / "consumed.records"
    meta = OVERLAP.extract_consumed(
        order, lengths, output, identity="test", trained_tokens=6
    )
    records = np.fromfile(output, dtype=OVERLAP.CONSUMED_DTYPE)
    assert meta["consumed_tokens"] == 6
    assert meta["domain_tokens"] == {"web": 2, "code": 4}
    assert records[["doc_tokens", "begin", "end"]].tolist() == [(3, 1, 3), (4, 0, 4)]

    binary = tmp_path / "consumed-merge"
    subprocess.run(["cc", "-O3", "-std=c11",
                    str(ROOT / "experiments/e4_curriculum/scripts/overlap_consumed_merge.c"),
                    "-o", str(binary)], check=True)
    merged = subprocess.run([str(binary), str(output), str(output), str(output)], check=True,
                            capture_output=True, text=True)
    metrics = json.loads(merged.stdout)
    assert metrics["occurrence_consumed_tokens"] == 18
    assert metrics["unique_consumed_tokens"] == 6
    assert metrics["token_mismatches"] == 0


def test_consumed_mixture_rejects_domain_drift_even_when_total_tokens_close():
    target = {"web": 0.9, "code": 0.099, "qa": 0.001}
    policy = {"absolute_tolerance": 0.002, "relative_tolerance": 0.10}
    assert OVERLAP.audit_consumed_mixture(
        {"web": 900000, "code": 99000, "qa": 1000}, target, **policy)["status"] == "PASS"
    # Rare-domain disappearance passes an absolute-only tolerance, so both
    # relative tolerance and the complete domain roster are necessary.
    assert OVERLAP.audit_consumed_mixture(
        {"web": 901000, "code": 99000}, target, **policy)["status"] == "FAIL"
    assert OVERLAP.audit_consumed_mixture(
        {"web": 880000, "code": 119000, "qa": 1000}, target, **policy)["status"] == "FAIL"


def _stage_mixture_fixture(root, targets):
    def write(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    script = root / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(OVERLAP_SCRIPT.read_bytes())
    order = root / "data/processed/mix_order/warmup.parquet"
    order.parent.mkdir(parents=True)
    pq.write_table(pa.table({"doc_id": [f"{i:032x}" for i in range(3)],
                             "position": [0, 1, 2], "domain": ["web", "code", "math"]}), order)
    lengths = root / "data/token_streams/pretrain-1t/warmup.doc_lengths.u32"
    lengths.parent.mkdir(parents=True)
    np.asarray([3, 4, 2], dtype="<u4").tofile(lengths)
    record = {"path": str(order.relative_to(root)), "sha256": TOKENIZE.sha256(order),
              **OVERLAP.stat_identity(order)}
    frozen = write("experiments/e4_curriculum/results/frozen_1t_recipe.json", {
        "status": "PASS", "stages": {"warmup": {"target_final_domain_weights": targets}}})
    write("pretrain/configs/pretrain_base.yaml", {
        "data": {"seq_len": 2, "stage_sample_limits": [3, 1, 1]}})
    write("experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml", {"release_gate": {
        "max_domain_weight_absolute_error": 0.002, "max_domain_weight_relative_error": 0.1}})
    write("data/token_streams/pretrain-1t/warmup.meta.json", {
        "source_document_binding": {"schema_version": 4, "expected_doc_ids_verified": True,
                                    "tokenizer_bytes_verified": True, "order_bytes_verified": True, "source_handles_verified": True,
                                    "roster_fingerprint": "a" * 64},
        "order_sha256": record["sha256"], "doc_lengths": {
            "file": lengths.name, "dtype": "<u4", "bytes": lengths.stat().st_size,
            "sha256": TOKENIZE.sha256(lengths)}})
    release = {"status": "PASS", "frozen_recipe_sha256": TOKENIZE.sha256(frozen),
               "stages": {"warmup": {"order": record["path"]}}, "artifacts": [record]}
    write("data/processed/releases/pretrain-1t-frozen.release.json", release)
    return release


@pytest.mark.parametrize("targets,status", [({"web": 1 / 3, "code": 2 / 3}, "PASS"),
                                            ({"web": 0.5, "code": 0.5}, "FAIL")])
def test_early_stage_mixture_uses_actual_causal_targets_and_reuses_final_cache(tmp_path, targets, status):
    release = _stage_mixture_fixture(tmp_path, targets)
    result = TOKENIZE.audit_stage_mixture(tmp_path, "warmup", release)
    assert result["status"] == status
    assert result["configured_target_tokens"] == 6
    assert result["consumed"]["domain_tokens"] == {"web": 2, "code": 4}
    cached = tmp_path / "data/processed/overlap_1t_cache/warmup.consumed.records"
    before = cached.stat().st_ino
    assert TOKENIZE.audit_stage_mixture(tmp_path, "warmup", release)["status"] == status
    assert cached.stat().st_ino == before


def test_early_stage_mixture_rejects_corrupt_length_ledger(tmp_path):
    release = _stage_mixture_fixture(tmp_path, {"web": 1 / 3, "code": 2 / 3})
    np.asarray([3, 3, 3], dtype="<u4").tofile(
        tmp_path / "data/token_streams/pretrain-1t/warmup.doc_lengths.u32")
    with pytest.raises(ValueError, match="unbound document-length ledger"):
        TOKENIZE.audit_stage_mixture(tmp_path, "warmup", release)


def test_early_stage_mixture_rejects_unverified_source_document_rows(tmp_path):
    release = _stage_mixture_fixture(tmp_path, {"web": 1 / 3, "code": 2 / 3})
    path = tmp_path / "data/token_streams/pretrain-1t/warmup.meta.json"
    meta = json.loads(path.read_text())
    del meta["source_document_binding"]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="requires verified source document rows"):
        TOKENIZE.audit_stage_mixture(tmp_path, "warmup", release)


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("failure", ["mixture", "capacity", "ledger"])
def test_tokenization_stops_before_stable_when_warmup_preflight_fails(tmp_path, monkeypatch, cached, failure):
    release = _stage_mixture_fixture(tmp_path, {"web": 0.5, "code": 0.5})
    for stage in ("stable", "anneal"):
        release["stages"][stage] = {"order": release["stages"]["warmup"]["order"]}
    paths = {name: tmp_path / getattr(TOKENIZE, name).relative_to(TOKENIZE.ROOT)
             for name in ("RELEASE", "BASE_RELEASE", "STATE", "STREAMS", "VAL_STREAMS", "VAL_ORDERS", "AUDIT")}
    paths["RELEASE"].write_text(json.dumps(release))
    paths["BASE_RELEASE"].write_text(json.dumps({"status": "PASS"}))
    tokenizer = tmp_path / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    (tmp_path / "data/TOKENIZER_READY.json").write_text(json.dumps({"vocab_size": 65536}))
    (tmp_path / "report/logs").mkdir(parents=True)
    for name, value in paths.items():
        monkeypatch.setattr(TOKENIZE, name, value)
    monkeypatch.setattr(TOKENIZE, "ROOT", tmp_path)
    ready = [cached]
    calls = []

    def materialize(command, **_kwargs):
        calls.append(command)
        ready[0] = True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(TOKENIZE, "valid_pair", lambda _out, name, **_kwargs: name == "warmup" and ready[0])
    monkeypatch.setattr(TOKENIZE, "storage_preflight", lambda *_args: {"status": "PASS"})
    monkeypatch.setattr(TOKENIZE.subprocess, "run", materialize)
    if failure == "capacity":
        config = tmp_path / "pretrain/configs/pretrain_base.yaml"
        config.write_text("data:\n  seq_len: 2\n  stage_sample_limits: [5, 1, 1]\n")
    elif failure == "ledger":
        (paths["STREAMS"] / "warmup.doc_lengths.u32").write_bytes(b"corrupt")
    result = TOKENIZE.run(2)
    expected = "BLOCKED_MIXTURE" if failure == "mixture" else "BLOCKED_PREFLIGHT"
    assert result["status"] == expected
    assert result["jobs"]["warmup"]["status"] == expected
    persisted = json.loads(paths["STATE"].read_text())
    assert persisted["status"] == expected
    if failure != "mixture":
        report = json.loads((tmp_path / "data/health/pretrain-1t-warmup-mixture-preflight.json").read_text())
        assert report["status"] == "BLOCKED_PREFLIGHT" and report["error"]
    assert result["jobs"]["stable"]["attempts"] == 0
    assert len(calls) == (0 if cached else 1)
