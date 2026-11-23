#!/usr/bin/env python3
"""Validate and bind a formal YAML/data contract before torchrun starts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


EXPECTED_VALIDATION_DOMAINS = {
    "val_web", "val_code", "val_math", "val_chinese", "val_academic",
    "val_books", "val_encyclopedia", "val_qa", "val_multilingual",
}
import numpy as np


CONFIG_FIELDS = {
    "root": {"run_name", "backend", "seed", "out_dir", "experiment_status", "model",
             "data", "train", "optim", "parallel", "compile", "attention", "loss",
             "research_compute"},
    "model": {"preset", "configs"},
    "data": {"seq_len", "epochs", "stages", "meta", "stage_sample_limits", "val_dir",
             "val_samples_per_domain", "val_max_tokens_per_domain", "tokenizer_gate",
             "release_manifest", "token_stream_audit", "require_release", "require_eod",
             "sampling_policy", "block_permutation_manifest"},
    "train": {"budget_tokens", "expected_iters", "micro_batch_size", "grad_accum",
              "save_every", "eval_every", "log_every", "keep_last_n", "keep_every",
              "activation_checkpointing", "bf16", "cuda_graphs", "deterministic"},
    "optim": {"optimizer", "lr", "clip_grad", "warmup_steps", "decay_steps",
              "lr_schedule", "min_lr_ratio", "adam_betas", "adam_eps", "weight_decay",
              "muon_momentum", "muon_nesterov", "muon_ns_steps", "lr_resolution",
              "batch_pilot_resolution"},
    "parallel": {"world_size", "sharding", "replicate_degree", "shard_degree"},
    "compile": {"enable", "components", "backend"},
    "attention": {"sdpa_backend"},
    "loss": {"num_chunks"},
}


class ContractLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


ContractLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _validate_config_fields(cfg: Any) -> None:
    for section, allowed in CONFIG_FIELDS.items():
        values = cfg if section == "root" else cfg.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported config fields in {section}: {sorted(unknown, key=str)}")
    if cfg.get('data', {}).get('sampling_policy', 'sequential') not in {'sequential', 'block_permutation'}:
        raise ValueError('unsupported sampling policy in formal contract')
    for section, keys in {
        "train": ("activation_checkpointing", "bf16", "cuda_graphs", "deterministic"),
        "compile": ("enable",), "data": ("require_release", "require_eod"),
    }.items():
        for key in keys:
            if key in cfg.get(section, {}) and type(cfg[section][key]) is not bool:
                raise ValueError(f"{section}.{key} must be a boolean")
    for section, keys in {
        "train": ("budget_tokens", "keep_every", "keep_last_n"),
        "optim": ("warmup_steps", "decay_steps"),
        "data": ("val_samples_per_domain", "val_max_tokens_per_domain"),
    }.items():
        for key in keys:
            if key in cfg.get(section, {}):
                value = cfg[section][key]
                if type(value) is not int or value < 0:
                    raise ValueError(f"{section}.{key} must be a non-negative integer")
    for value in cfg.get("data", {}).get("stage_sample_limits", []):
        if type(value) is not int or value < 1:
            raise ValueError("stage_sample_limits must contain positive integers")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _edge_sha256(path: Path, edge_bytes: int = 1 << 20) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as fh:
        h.update(fh.read(edge_bytes))
        if size > edge_bytes:
            fh.seek(max(0, size - edge_bytes))
            h.update(fh.read(edge_bytes))
    return h.hexdigest()


def _source_sha256(repo: Path) -> str:
    # Bind the selected model preset file separately in the run identity. This
    # keeps frozen production presets from invalidating completed experiments.
    roots = [repo / "model/src", repo / "pretrain/backends/torchtitan"]
    h = hashlib.sha256()
    for path in sorted(p for base in roots for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(repo)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    for relative in ("pretrain/scripts/run_with_lease.py", "pretrain/scripts/assert_run_lease.py",
                     "pretrain/src/yuformer_pretrain/lease.py"):
        path = repo / relative
        if path.is_file():
            h.update(relative.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def validate(repo: Path, config_path: Path, *, publish_identity: bool,
             require_production_order: bool = False) -> dict[str, Any]:
    rendered = os.path.expandvars(config_path.read_text(encoding="utf-8"))
    if "${" in rendered:
        raise ValueError("unresolved environment variable in training config")
    cfg = yaml.load(rendered, Loader=ContractLoader)
    _validate_config_fields(cfg)
    if cfg.get("backend") != "torchtitan-v0.3.0":
        raise ValueError("formal contract must use torchtitan-v0.3.0")
    if cfg.get("experiment_status", "active") not in {"active", "qualification"}:
        raise ValueError("non-formal experiment status cannot launch")
    if type(cfg.get("seed")) is not int or cfg["seed"] < 0:
        raise ValueError("formal contract requires a non-negative integer seed")

    data, train, parallel = cfg["data"], cfg["train"], cfg["parallel"]
    optim = cfg["optim"]
    for key in ("require_release", "require_eod"):
        if key in data and data[key] is not True:
            raise ValueError(f"formal training cannot disable data.{key}")
    if ("val_max_tokens_per_domain" in data
            and data["val_samples_per_domain"] * data["seq_len"] > data["val_max_tokens_per_domain"]):
        raise ValueError("validation sample budget exceeds val_max_tokens_per_domain")
    for key, value in {
        "seq_len": data["seq_len"], "micro_batch_size": train["micro_batch_size"],
        "grad_accum": train["grad_accum"], "world_size": parallel["world_size"],
        "expected_iters": train["expected_iters"], "epochs": data.get("epochs", 1),
        "eval_every": train["eval_every"], "log_every": train["log_every"],
        "save_every": train["save_every"],
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    loss_chunks = cfg.get("loss", {}).get("num_chunks", 8)
    if (not isinstance(loss_chunks, int) or isinstance(loss_chunks, bool)
            or loss_chunks < 1 or int(data["seq_len"]) % loss_chunks):
        raise ValueError("loss.num_chunks must be a positive divisor of sequence length")
    if optim.get("optimizer") != "adamw":
        raise ValueError("formal backend supports AdamW; other optimizers need separate qualification")
    sharding = parallel.get("sharding")
    replicate_degree = parallel.get("replicate_degree", 1)
    shard_degree = parallel.get("shard_degree", -1)
    if sharding not in {"full_shard", "hybrid_shard"}:
        raise ValueError("formal adapter supports full_shard and hybrid_shard")
    if not isinstance(replicate_degree, int) or isinstance(replicate_degree, bool) or replicate_degree < 1:
        raise ValueError("parallel.replicate_degree must be a positive integer")
    if not isinstance(shard_degree, int) or isinstance(shard_degree, bool) or shard_degree == 0 or shard_degree < -1:
        raise ValueError("parallel.shard_degree must be a positive integer or -1")
    resolved_shard_degree = int(parallel["world_size"]) // replicate_degree if shard_degree == -1 else shard_degree
    if replicate_degree * resolved_shard_degree != int(parallel["world_size"]):
        raise ValueError("replicate_degree * shard_degree must equal world_size")
    if sharding == "full_shard" and (replicate_degree != 1 or resolved_shard_degree != int(parallel["world_size"])):
        raise ValueError("full_shard requires replicate_degree=1 and shard_degree=world_size")
    if sharding == "hybrid_shard" and (replicate_degree <= 1 or resolved_shard_degree <= 1):
        raise ValueError("hybrid_shard requires replicate_degree>1 and shard_degree>1")
    if train.get("cuda_graphs", False):
        raise ValueError("CUDA graphs require separate qualification of the finite-gradient guard")
    compile_cfg = cfg.get("compile", {})
    if not isinstance(compile_cfg, dict):
        raise ValueError("compile must be a mapping")
    if compile_cfg.get("enable", False):
        components = compile_cfg.get("components", ["model", "loss"])
        if not isinstance(components, list) or not components or any(
            component not in {"model", "loss"} for component in components
        ):
            raise ValueError("compile.components must be a non-empty subset of ['model', 'loss']")
        if compile_cfg.get("backend", "inductor") != "inductor":
            raise ValueError("formal adapter only qualifies the inductor compile backend")
    attention_cfg = cfg.get("attention", {})
    if (not isinstance(attention_cfg, dict)
            or attention_cfg.get("sdpa_backend", "auto") not in {"auto", "cudnn", "cudnn_flex_swa"}):
        raise ValueError("attention.sdpa_backend must be auto, cudnn or cudnn_flex_swa")
    if (attention_cfg.get('sdpa_backend') == 'cudnn_flex_swa'
            and not cfg.get('compile', {}).get('enable')):
        raise ValueError('cudnn_flex_swa requires explicit torch.compile enablement')
    betas = optim.get("adam_betas", [0.9, 0.95])
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(not math.isfinite(float(value)) or not 0 <= float(value) < 1 for value in betas)):
        raise ValueError("Adam betas must be two finite values in [0, 1)")
    if not math.isfinite(float(optim.get("adam_eps", 1e-8))) or float(optim.get("adam_eps", 1e-8)) <= 0:
        raise ValueError("Adam epsilon must be finite and positive")
    if (not math.isfinite(float(optim.get("weight_decay", 0.1)))
            or float(optim.get("weight_decay", 0.1)) < 0):
        raise ValueError("weight decay must be finite and non-negative")
    for key in ("lr", "clip_grad"):
        if not math.isfinite(float(optim[key])) or float(optim[key]) <= 0:
            raise ValueError(f"optimizer {key} must be finite and positive")
    if not 0 <= int(optim["warmup_steps"]) <= int(train["expected_iters"]):
        raise ValueError("warmup_steps falls outside the training horizon")
    if optim.get("lr_schedule") not in {"cosine", "wsd"}:
        raise ValueError("unsupported LR schedule")
    if optim["lr_schedule"] == "wsd":
        if not 0 < int(optim["decay_steps"]) <= int(train["expected_iters"]) - int(optim["warmup_steps"]):
            raise ValueError("WSD warmup and decay exceed the training horizon")
    if not 0 <= float(optim.get("min_lr_ratio", 0)) <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1]")
    if int(train.get("keep_last_n", 2)) not in (0,) and int(train.get("keep_last_n", 2)) < 2:
        raise ValueError("keep_last_n must be zero or at least two for async checkpoint safety")
    if len(data["stages"]) != len(data["meta"]) or not data["stages"]:
        raise ValueError("training stages/meta must be non-empty and one-to-one")
    tps = int(data["seq_len"]) * int(train["micro_batch_size"]) * int(train["grad_accum"]) * int(parallel["world_size"])
    if int(train["expected_iters"]) != int(train["budget_tokens"]) // tps:
        raise ValueError("expected_iters does not close against the token budget")

    gate_path = Path(data["tokenizer_gate"])
    release_path = Path(data["release_manifest"])
    stream_audit_path = Path(data["token_stream_audit"])
    gate, release, stream_audit = (
        _json(gate_path), _json(release_path), _json(stream_audit_path)
    )
    model_document = yaml.load(
        Path(cfg["model"]["configs"]).read_text(encoding="utf-8"), Loader=ContractLoader
    )
    preset = model_document["presets"][cfg["model"]["preset"]]
    if train.get("activation_checkpointing", False) and preset.get("selective_recompute", "none") != "none":
        raise ValueError("full activation checkpointing and model selective recompute cannot both be enabled")
    if int(preset["vocab_size"]) != int(gate["vocab_size"]):
        raise ValueError("model vocabulary differs from the tokenizer gate")
    if int(data["seq_len"]) > int(preset["max_seq_len"]):
        raise ValueError("sequence length exceeds the model RoPE capacity")
    if gate.get("status") != "PASS" or release.get("status") != "PASS":
        raise ValueError("tokenizer/data gate is not PASS")
    for key in ("dataset_id", "snapshot_id"):
        if gate.get(key) != release.get(key):
            raise ValueError(f"tokenizer/data identity mismatch: {key}")
    if stream_audit.get("status") != "PASS" or stream_audit.get("roster_exact") is not True:
        raise ValueError("token stream full audit is not PASS with an exact roster")
    if (stream_audit.get("audit") == "pretrain_1t_token_streams"
            and stream_audit.get("train_doc_lengths_audited") is not True):
        raise ValueError("production token stream audit does not close document lengths")
    if stream_audit.get("audit") == "pretrain_1t_token_streams":
        stream_tool = repo / "experiments/e4_curriculum/scripts/audit_1t_token_streams.py"
        shared_tool = repo / "pretrain/scripts/audit_token_stream_release.py"
        release_tool = repo / "experiments/e4_curriculum/scripts/build_1t_order_release.py"
        if (stream_audit.get("audit_tool_sha256") != _sha256(stream_tool)
                or stream_audit.get("shared_auditor_sha256") != _sha256(shared_tool)):
            raise ValueError("production token stream audit uses stale audit code")
        if release.get("builder_tool_sha256") != _sha256(release_tool):
            raise ValueError("production derived release uses stale builder code")
        freeze_path = repo / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
        origin_path = repo/'experiments/e4_curriculum/scripts/training_data_origin.py'
        origin_spec = importlib.util.spec_from_file_location('_contract_data_origin', origin_path)
        origin_module = importlib.util.module_from_spec(origin_spec)
        origin_spec.loader.exec_module(origin_module)
        origin_module.resolve_data_origin(repo, release_path, freeze_path)
    if stream_audit.get("dataset_id") != release.get("dataset_id"):
        raise ValueError("token stream audit dataset identity mismatch")
    if stream_audit.get("snapshot_id") != release.get("snapshot_id"):
        raise ValueError("token stream audit snapshot identity mismatch")
    if stream_audit.get("tokenizer_sha256") != gate.get("tokenizer_sha256"):
        raise ValueError("token stream audit tokenizer identity mismatch")
    validation_domains = {
        str(row.get("name"))
        for row in stream_audit.get("streams", [])
        if isinstance(row, dict) and str(row.get("name", "")).startswith("val_")
        and row.get("status") == "PASS"
    }
    validation_samples = int(data.get("val_samples_per_domain", 0))
    validation_global_batch = (int(parallel["world_size"])
                               * int(train["micro_batch_size"]))
    if validation_domains != EXPECTED_VALIDATION_DOMAINS:
        raise ValueError(f"validation domain roster is not exact: {sorted(validation_domains)}")
    if (validation_samples < 1
            or validation_samples % validation_global_batch):
        raise ValueError(
            "validation roster is absent or each domain is not global-batch aligned"
        )

    tokenizer_value = gate.get("tokenizer_path")
    if not isinstance(tokenizer_value, str) or not tokenizer_value:
        raise ValueError("tokenizer gate is missing tokenizer_path")
    tokenizer_path = Path(tokenizer_value)
    tokenizer_path = (tokenizer_path if tokenizer_path.is_absolute()
                      else repo / tokenizer_path).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer artifact: {tokenizer_path}")
    tokenizer_artifact_sha256 = _sha256(tokenizer_path)
    if tokenizer_artifact_sha256 != gate.get("tokenizer_sha256"):
        raise ValueError(
            f"tokenizer artifact hash mismatch: {tokenizer_path}: "
            f"{tokenizer_artifact_sha256} != {gate.get('tokenizer_sha256')}"
        )

    def audited_path(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else repo / path).resolve()

    audited_streams = {
        str(audited_path(row["artifacts"]["stream"])): row
        for row in stream_audit.get("streams", [])
        if isinstance(row, dict) and isinstance(row.get("artifacts"), dict)
        and isinstance(row["artifacts"].get("stream"), str)
    }

    stage_limits = [int(value) for value in data.get("stage_sample_limits", [])]
    if stage_limits and len(stage_limits) != len(data["stages"]):
        raise ValueError("stage_sample_limits must be empty or one-to-one with stages")
    samples_per_optimizer_step = (int(train["micro_batch_size"])
                                  * int(train["grad_accum"]) * int(parallel["world_size"]))
    if stage_limits:
        if any(value < 1 or value % samples_per_optimizer_step for value in stage_limits):
            raise ValueError("every stage sample limit must be positive and optimizer-step aligned")
        if sum(stage_limits) != int(train["expected_iters"]) * samples_per_optimizer_step:
            raise ValueError("stage sample limits do not close against expected_iters")
        if len(stage_limits) == 3 and optim["lr_schedule"] == "wsd":
            if (stage_limits[0] != optim["warmup_steps"] * samples_per_optimizer_step
                    or stage_limits[-1] != optim["decay_steps"] * samples_per_optimizer_step):
                raise ValueError("three-stage WSD data boundaries must match LR warmup and decay")
    stream_rows, validation_rows = [], []
    pairs = [(index, binary, meta, True) for index, (binary, meta) in enumerate(
        zip(data["stages"], data["meta"], strict=True))]
    val_dir = Path(data["val_dir"])
    val_streams = sorted(val_dir.glob("val_*.tokens.bin"))
    if {path.name.removesuffix(".tokens.bin") for path in val_streams} != EXPECTED_VALIDATION_DOMAINS:
        raise ValueError("validation stream domain roster is not exact")
    pairs.extend((index, str(path), str(path.with_suffix("").with_suffix(".meta.json")), False)
                 for index, path in enumerate(val_streams))
    for stage_index, binary_name, meta_name, is_train in pairs:
        binary, meta_path = Path(binary_name), Path(meta_name)
        if not binary.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing stream pair: {binary}, {meta_path}")
        meta = _json(meta_path)
        if binary.stat().st_size != int(meta["bytes"]):
            raise ValueError(f"stream byte length mismatch: {binary}")
        dtype = np.dtype(meta.get("dtype", "<u2"))
        if dtype.kind != "u" or dtype.itemsize not in (2, 4) or int(meta["total_tokens"]) * dtype.itemsize != binary.stat().st_size:
            raise ValueError(f"stream dtype/token length mismatch: {binary}")
        if _edge_sha256(binary) != meta.get("edge_sha256"):
            raise ValueError(f"stream edge hash mismatch: {binary}")
        audited = audited_streams.get(str(binary.resolve()))
        if audited is None or audited.get("status") != "PASS":
            raise ValueError(f"stream is absent from the PASS full audit: {binary}")
        if audited.get("actual_sha256") != meta.get("sha256"):
            raise ValueError(f"stream full audit hash mismatch: {binary}")
        if int(audited.get("actual_bytes", -1)) != binary.stat().st_size:
            raise ValueError(f"stream full audit byte length mismatch: {binary}")
        if int(stream_audit.get("schema_version", 1)) >= 2:
            current_stat = binary.stat()
            audited_stat = (
                int(audited.get("actual_device", -1)),
                int(audited.get("actual_inode", -1)),
                int(audited.get("actual_mtime_ns", -1)),
            )
            current_identity = (
                current_stat.st_dev, current_stat.st_ino, current_stat.st_mtime_ns,
            )
            if audited_stat != current_identity:
                raise ValueError(f"stream changed after full audit: {binary}")
        if is_train and stream_audit.get("audit") == "pretrain_1t_token_streams":
            lengths_record = meta.get("doc_lengths") or {}
            lengths_value = lengths_record.get("file")
            if int(meta.get("schema_version", 0)) < 3 or not isinstance(lengths_value, str):
                raise ValueError(f"production stream lacks a document-length ledger: {meta_path}")
            lengths_path = meta_path.parent / lengths_value
            audited_lengths = (audited.get("artifacts") or {}).get("doc_lengths")
            if (not lengths_path.is_file()
                    or audited_lengths is None
                    or audited_path(audited_lengths) != lengths_path.resolve()
                    or lengths_path.stat().st_size != int(lengths_record.get("bytes", -1))
                    or int(audited.get("actual_doc_lengths_bytes", -2)) != lengths_path.stat().st_size
                    or audited.get("actual_doc_lengths_sha256") != lengths_record.get("sha256")
                    or _edge_sha256(lengths_path) != lengths_record.get("edge_sha256")):
                raise ValueError(f"production document-length audit mismatch: {lengths_path}")
            lengths_stat = lengths_path.stat()
            audited_lengths_stat = (
                int(audited.get("actual_doc_lengths_device", -1)),
                int(audited.get("actual_doc_lengths_inode", -1)),
                int(audited.get("actual_doc_lengths_mtime_ns", -1)),
            )
            if audited_lengths_stat != (
                    lengths_stat.st_dev, lengths_stat.st_ino, lengths_stat.st_mtime_ns):
                raise ValueError(f"document-length ledger changed after full audit: {lengths_path}")
        capacity_samples = (int(meta["total_tokens"]) - 1) // int(data["seq_len"])
        if is_train and stage_limits and stage_limits[stage_index] > capacity_samples:
            raise ValueError(
                f"stage sample limit exceeds stream capacity: {binary}: "
                f"{stage_limits[stage_index]} > {capacity_samples}"
            )
        if meta.get("tokenizer_sha256") != gate.get("tokenizer_sha256"):
            raise ValueError(f"stream tokenizer mismatch: {meta_path}")
        if int(meta.get("vocab_size", -1)) != int(gate.get("vocab_size", -2)):
            raise ValueError(f"stream vocabulary mismatch: {meta_path}")
        if meta.get("data_snapshot_id") != release.get("snapshot_id"):
            raise ValueError(f"stream snapshot mismatch: {meta_path}")
        boundary = meta.get("document_boundary") or {}
        if boundary.get("policy") != "append_eod" or int(boundary.get("count", -1)) != int(meta.get("n_docs", -2)):
            raise ValueError(f"stream EOD boundary mismatch: {meta_path}")
        if int(boundary.get('token_id', -1)) != int(preset.get('eos_token_id', 0)):
            raise ValueError(f'model EOS and stream EOD ids differ: {meta_path}')
        if not is_train and int(data["val_samples_per_domain"]) > capacity_samples:
            raise ValueError(f"validation sample budget exceeds stream capacity: {binary}")
        (stream_rows if is_train else validation_rows).append({
            "binary": str(binary), "meta": str(meta_path),
            "edge_sha256": meta["edge_sha256"], "meta_sha256": _sha256(meta_path),
            "capacity_samples": capacity_samples,
            "sample_limit": (stage_limits[stage_index]
                             if is_train and stage_limits else capacity_samples),
        })

    capacity = sum(row["sample_limit"] for row in stream_rows) * int(data.get("epochs", 1))
    required = int(train["expected_iters"]) * samples_per_optimizer_step
    if capacity < required:
        raise ValueError(f"training stream capacity is below the declared budget: {capacity} < {required}")

    order_path = Path(__file__).with_name('yuformer_ext')/'block_permutation.py'
    order_spec = importlib.util.spec_from_file_location('_contract_block_permutation', order_path)
    order_module = importlib.util.module_from_spec(order_spec)
    order_spec.loader.exec_module(order_module)
    order_identity = order_module.configuration_order_identity(
        cfg, require_production=require_production_order)

    identity = {
        "schema_version": 1,
        "runtime_state_schema_version": 2,
        "training_sample_order": order_identity,
        "run_name": cfg["run_name"],
        "backend": cfg["backend"],
        "resolved_config_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "source_sha256": _source_sha256(repo),
        "model_configs": str(Path(cfg["model"]["configs"]).resolve()),
        "model_configs_sha256": _sha256(Path(cfg["model"]["configs"])),
        "tokenizer_gate": str(gate_path.resolve()),
        "tokenizer_gate_sha256": _sha256(gate_path),
        "release_manifest": str(release_path.resolve()),
        "release_manifest_sha256": _sha256(release_path),
        "dataset_id": release["dataset_id"],
        "snapshot_id": release["snapshot_id"],
        "tokenizer_id": gate["tokenizer_id"],
        "tokenizer_sha256": gate["tokenizer_sha256"],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
        "token_stream_audit": str(stream_audit_path),
        "token_stream_audit_sha256": _sha256(stream_audit_path),
        "streams": stream_rows,
        "validation_streams": validation_rows,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    out_dir = Path(cfg["out_dir"])
    if publish_identity:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "run_identity.json"
        if path.exists():
            old = _json(path)
            if old != identity:
                raise ValueError(f"output directory is bound to a different run identity: {path}")
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(identity, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
        resolved_path = out_dir / "resolved_config.yaml"
        if resolved_path.exists():
            if resolved_path.read_text(encoding="utf-8") != rendered:
                raise ValueError(
                    f"output directory contains a different resolved config: {resolved_path}"
                )
        else:
            fd = os.open(resolved_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(rendered)
                fh.flush()
                os.fsync(fh.fileno())
    return {"status": "PASS", "out_dir": str(out_dir), "identity": identity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--print-out-dir", action="store_true")
    args = parser.parse_args()
    result = validate(args.repo.resolve(), args.config.resolve(), publish_identity=not args.print_out_dir)
    print(result["out_dir"] if args.print_out_dir else json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
