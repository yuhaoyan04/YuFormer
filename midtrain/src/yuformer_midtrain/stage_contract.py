"""Validate stage boundaries independently of the pretrain launcher."""
from __future__ import annotations


def validate_stage_geometry(config, budget):
    stage = config.get('midtrain', {})
    if stage.get('scope') not in ('PRODUCTION', 'QUALIFICATION'):
        raise ValueError('midtrain scope must explicitly distinguish production from qualification')
    seq = config['data'].get('seq_len')
    if seq not in (16384, 65536):
        raise ValueError('midtrain supports the declared 16K/64K ladder only')
    budget_stage = next((s for s in budget.get('stages', []) if s.get('seq_len') == seq), None)
    if (budget.get('status') != 'ACTIVE_USER_APPROVED_TOTAL_1T_INCLUDING_LONG_CONTEXT'
            or budget.get('world_size') != 8 or budget.get('tokens_per_update') != 4194304
            or budget_stage is None or budget_stage.get('short_replay_fraction') != .25):
        raise ValueError('midtrain needs the active total-budget allocation and short-replay policy')
    train, parallel = config['train'], config['parallel']
    values = [parallel.get('world_size'), train.get('micro_batch_size'), train.get('grad_accum')]
    if (any(type(v) is not int or v <= 0 for v in values) or values[0] != 8
            or seq*values[0]*values[1]*values[2] != 4194304):
        raise ValueError('midtrain must use eight GPUs and constant4194304targetlabels/update')
    if (parallel.get('sharding') != 'hybrid_shard'
            or (parallel.get('replicate_degree'),parallel.get('shard_degree')) != (2,4)
            or parallel.get('fsdp_reshard_after_forward') != 'never'):
        raise ValueError('long stages must use the measured hybrid2x4/never sharding policy')
    steps, tokens = train.get('expected_iters'), train.get('budget_tokens')
    if type(steps) is not int or steps <= 0 or type(tokens) is not int or tokens != steps*4194304:
        raise ValueError('midtrain optimizer steps and target-label budget do not close')
    if (stage['scope'] == 'PRODUCTION' and (steps != budget_stage.get('updates') or tokens != budget_stage.get('tokens'))
            or stage['scope'] == 'QUALIFICATION' and steps > budget_stage.get('updates', 0)):
        raise ValueError('midtrain budget exceeds or differs from its declared production allocation')
    if train.get('deterministic') is not True or train.get('bf16') is not True:
        raise ValueError('current midtrain policy requires explicit deterministic BF16 execution')
    if config.get('attention', {}).get('sdpa_backend') != 'flash_flex_swa':
        raise ValueError('midtrain must use the qualified deterministic hybrid Flash backend')
    if (config.get('compile', {}).get('enable') is not True
            or 'model' not in config['compile'].get('components', [])):
        raise ValueError('hybrid midtrain attention requires explicit model compilation')
    chunks = config.get('loss', {}).get('num_chunks')
    if type(chunks) is not int or chunks <= 0 or seq % chunks:
        raise ValueError('midtrain loss chunks must divide the sequence length')
    if seq == 65536:
        if (train.get('activation_checkpointing') is not False
                or stage.get('selective_recompute') != 'mlp' or seq//chunks > 2048
                or stage.get('cuda_allocator_conf') != 'expandable_segments:True,garbage_collection_threshold:0.7'):
            raise ValueError('64K requires the qualified selectiveMLP/chunks/allocator policy')
    if stage.get('optimizer_transition') != 'reset' or stage.get('initial_load_model_only') is not True:
        raise ValueError('new context stages must load weights only and reset optimizer/scheduler/data/RNG')
    if config['optim'].get('lr_schedule') != 'cosine':
        raise ValueError('context extension needs its independently declared cosine schedule')
    import math
    optim=config['optim']
    if (optim.get('optimizer')!='adamw' or type(optim.get('warmup_steps')) is not int
            or not 0<=optim['warmup_steps']<steps
            or type(optim.get('lr')) not in (int,float) or not math.isfinite(optim['lr']) or optim['lr']<=0
            or type(optim.get('min_lr_ratio')) not in (int,float) or not 0<=optim['min_lr_ratio']<=1):
        raise ValueError('long-stage AdamW/cosine learning rate and warmup horizon must be explicit and finite')
    if stage.get('evaluation') != {'long_context': True, 'short_regression': True}:
        raise ValueError('stage progression must require both long capability and short regression evaluation')
    return {'seq_len': seq, 'updates': steps, 'tokens': tokens,
            'samples': tokens//seq, 'scope': stage['scope']}
