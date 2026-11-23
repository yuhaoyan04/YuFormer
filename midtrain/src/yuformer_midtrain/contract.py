"""Machine-checkable contract for progressive context extension."""
from __future__ import annotations

import re
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATASET_ID = re.compile(r"^yuformer-dataset-[0-9a-f]{20}$")


def audit_ladder(config: Any) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    if not isinstance(config, dict):
        return {"schema_version": 1, "status": "FAIL", "errors": ["config must be an object"], "blockers": []}
    if config.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if config.get("backend") != "torchtitan-v0.3.0":
        errors.append("backend must equal torchtitan-v0.3.0")
    checkpoint = config.get("base_checkpoint") or {}
    if not HEX64.fullmatch(str(checkpoint.get("manifest_sha256") or "")):
        blockers.append("base_checkpoint.manifest_sha256 is not locked")
    if not DATASET_ID.fullmatch(str(config.get("dataset_id") or "")):
        blockers.append("dataset_id is not a PASS release identity")
    if config.get("optimizer_transition") != "reset":
        errors.append("optimizer_transition must be reset for weight-only midtraining")

    target = config.get("target_tokens_per_update")
    required_world = config.get("required_world_size")
    if required_world is not None and (type(required_world) is not int or required_world < 1):
        errors.append("required_world_size must be a positive integer")
    if not isinstance(target, int) or target <= 0:
        errors.append("target_tokens_per_update must be a positive integer")
        target = None
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
        stages = []
    previous_seq = int(config.get("pretrain_context", 0) or 0)
    names: set[str] = set()
    for index, stage in enumerate(stages):
        prefix = f"stages[{index}]"
        if not isinstance(stage, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = stage.get("name")
        if not isinstance(name, str) or not name or name in names:
            errors.append(f"{prefix}.name must be non-empty and unique")
        else:
            names.add(name)
        seq = stage.get("seq_len")
        if not isinstance(seq, int) or seq <= previous_seq:
            errors.append(f"{prefix}.seq_len must strictly increase from {previous_seq}")
        else:
            previous_seq = seq
        if target is not None and stage.get("tokens_per_update") != target:
            errors.append(f"{prefix}.tokens_per_update must equal {target}")
        topology = (stage.get("world_size"), stage.get("micro_batch_size"), stage.get("grad_accum"))
        if required_world is not None and topology[0] != required_world:
            errors.append(f"{prefix}.world_size must equal required_world_size={required_world}")
        if not all(isinstance(value, int) and value > 0 for value in topology):
            errors.append(f"{prefix}.world_size/micro_batch_size/grad_accum must be positive integers")
        elif isinstance(seq, int):
            derived = seq * topology[0] * topology[1] * topology[2]
            if stage.get("tokens_per_update") != derived:
                errors.append(
                    f"{prefix}.tokens_per_update={stage.get('tokens_per_update')} "
                    f"does not match seq_len*world_size*micro_batch_size*grad_accum={derived}"
                )
        replay = stage.get("short_replay_fraction")
        if not isinstance(replay, (int, float)) or isinstance(replay, bool) or not 0.1 <= replay <= 0.5:
            errors.append(f"{prefix}.short_replay_fraction must be within [0.1, 0.5]")
        kernel = stage.get("attention_kernel")
        if kernel not in {"flash_attention_2", "flash_attention_3", "cudnn_flex_swa", "flash_flex_swa"}:
            errors.append(f"{prefix}.attention_kernel must use a qualified linear-memory backend")
        if kernel in {'cudnn_flex_swa', 'flash_flex_swa'}:
            if kernel == 'flash_flex_swa' and stage.get('deterministic') is not True:
                errors.append(f'{prefix}.flash_flex_swa requires explicit deterministic execution')
            if stage.get('compile_model') is not True:
                errors.append(f'{prefix}.cudnn_flex_swa requires explicit model compilation')
            if type(stage.get('activation_checkpointing')) is not bool:
                errors.append(f'{prefix}.activation_checkpointing must be an explicit boolean')
            if stage.get('selective_recompute') not in {'none','mlp','attn'}:
                errors.append(f'{prefix}.selective_recompute is not explicitly configured')
            chunks = stage.get('loss_num_chunks')
            if type(chunks) is not int or chunks <= 0 or type(seq) is not int or seq % chunks:
                errors.append(f'{prefix}.loss_num_chunks must divide the sequence length')
            if type(seq) is int and seq >= 65536:
                if (stage.get('activation_checkpointing') is not False
                        or stage.get('selective_recompute') != 'mlp'):
                    errors.append(f'{prefix}.64K cuDNN/Flex requires the tested selective MLP policy; full checkpointing is unqualified')
                if type(chunks) is not int or chunks <= 0 or seq//chunks > 2048:
                    errors.append(f'{prefix}.64K loss chunks must contain at most 2048 sequence positions')
                if stage.get('cuda_allocator_conf') != 'expandable_segments:True,garbage_collection_threshold:0.7':
                    errors.append(f'{prefix}.64K requires the tested allocator headroom policy for checkpoint saves')
        if stage.get("init") != "previous_stage_weights_only":
            errors.append(f"{prefix}.init must be previous_stage_weights_only")
        if not isinstance(stage.get("budget_tokens"), int) or stage.get("budget_tokens", 0) <= 0:
            errors.append(f"{prefix}.budget_tokens must be positive")
        evaluation = stage.get("evaluation") or {}
        if evaluation.get("long_context") is not True or evaluation.get("short_regression") is not True:
            errors.append(f"{prefix}.evaluation must require long_context and short_regression")

    status = "FAIL" if errors else "BLOCKED" if blockers else "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "stage_count": len(stages),
        "errors": errors,
        "blockers": blockers,
    }
