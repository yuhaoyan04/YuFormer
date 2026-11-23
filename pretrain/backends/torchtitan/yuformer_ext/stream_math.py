"""Pure stream arithmetic shared by the TorchTitan data adapter and tests."""


def sample_count(total_tokens: int, seq_len: int) -> int:
    if total_tokens < 0 or seq_len < 1:
        raise ValueError("total_tokens must be >=0 and seq_len must be >=1")
    return max(0, (total_tokens - 1) // seq_len)


def sample_offset(sample_index: int, seq_len: int) -> int:
    if sample_index < 0 or seq_len < 1:
        raise ValueError("sample_index must be >=0 and seq_len must be >=1")
    return sample_index * seq_len


def round_robin_location(sample_index: int, stage_samples: list[int]) -> tuple[int, int]:
    """Map a logical validation sample to a domain and its local sample.

    Cycling domains before advancing the local sample keeps every prefix
    balanced.  The usable cycle count is bounded by the smallest domain so no
    domain silently wraps while another still advances.
    """
    if sample_index < 0 or not stage_samples or any(n < 1 for n in stage_samples):
        raise ValueError("invalid round-robin sample request")
    stage = sample_index % len(stage_samples)
    local = sample_index // len(stage_samples)
    if local >= min(stage_samples):
        raise IndexError("round-robin validation capacity exhausted")
    return stage, local


def domain_round_robin_location(
    sample_index: int, stage_samples: list[int], global_batch_size: int
) -> tuple[int, int]:
    """Round-robin whole global batches so per-domain loss stays identifiable."""
    if (sample_index < 0 or global_batch_size < 1 or not stage_samples
            or any(n < 1 for n in stage_samples)):
        raise ValueError("invalid domain-batched round-robin request")
    batch, within_batch = divmod(sample_index, global_batch_size)
    stage = batch % len(stage_samples)
    domain_batch = batch // len(stage_samples)
    local = domain_batch * global_batch_size + within_batch
    if local >= stage_samples[stage]:
        raise IndexError("domain-batched round-robin capacity exhausted")
    return stage, local


def resume_step(consumed_samples: int, dp_world_size: int, local_batch_size: int) -> int:
    if consumed_samples < 0 or dp_world_size < 1 or local_batch_size < 1:
        raise ValueError("invalid cursor/topology")
    global_batch = dp_world_size * local_batch_size
    if consumed_samples % global_batch:
        raise ValueError("cursor is not aligned to the new global batch")
    return consumed_samples // global_batch


def validation_steps(samples_per_domain: int, domain_count: int,
                     dp_world_size: int, local_batch_size: int) -> int:
    """Return steps that consume whole, domain-pure global batches exactly."""
    if (samples_per_domain < 1 or domain_count < 1 or dp_world_size < 1
            or local_batch_size < 1):
        raise ValueError("validation dimensions must be positive")
    global_batch = dp_world_size * local_batch_size
    # Domain validation advances in whole global batches.  Checking only the
    # aggregate roster is insufficient when domain_count and global_batch
    # share factors: an individual domain could end in a partial batch.
    if samples_per_domain % global_batch:
        raise ValueError("each validation domain must be global-batch aligned")
    return domain_count * (samples_per_domain // global_batch)


def should_validate_step(step: int, frequency: int, final_step: int) -> bool:
    if step < 1 or frequency < 1 or final_step < 1:
        raise ValueError("validation schedule values must be positive")
    return step == 1 or step % frequency == 0 or step == final_step
