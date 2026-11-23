import importlib.util
from pathlib import Path

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "backends/torchtitan/yuformer_ext/stream_math.py"
)
SPEC = importlib.util.spec_from_file_location("stream_math", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_causal_windows_share_only_the_boundary_input():
    seq = 4
    assert MODULE.sample_count(13, seq) == 3
    windows = [range(MODULE.sample_offset(i, seq), MODULE.sample_offset(i, seq) + seq + 1) for i in range(3)]
    assert list(windows[0]) == [0, 1, 2, 3, 4]
    assert list(windows[1]) == [4, 5, 6, 7, 8]
    targets = [token for window in windows for token in list(window)[1:]]
    assert targets == list(range(1, 13))


def test_cursor_reshards_only_on_exact_new_batch_boundary():
    consumed = 4 * 8 * 17
    assert MODULE.resume_step(consumed, dp_world_size=2, local_batch_size=8) == 34
    assert MODULE.resume_step(consumed, dp_world_size=8, local_batch_size=4) == 17
    with pytest.raises(ValueError, match="aligned"):
        MODULE.resume_step(consumed + 1, dp_world_size=8, local_batch_size=4)


def test_round_robin_validation_prefix_is_domain_balanced():
    counts = [5, 3, 4]
    locations = [MODULE.round_robin_location(i, counts) for i in range(9)]
    assert locations == [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
    ]
    with pytest.raises(IndexError, match="capacity"):
        MODULE.round_robin_location(9, counts)


def test_domain_round_robin_keeps_global_batches_pure():
    counts = [12, 12, 12]
    locations = [MODULE.domain_round_robin_location(i, counts, 4) for i in range(24)]
    assert locations[:4] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert locations[4:8] == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert locations[8:12] == [(2, 0), (2, 1), (2, 2), (2, 3)]
    assert locations[12:16] == [(0, 4), (0, 5), (0, 6), (0, 7)]
    with pytest.raises(IndexError, match="capacity"):
        MODULE.domain_round_robin_location(36, counts, 4)


def test_validation_steps_include_local_batch_in_global_batch():
    assert MODULE.validation_steps(64, 9, 4, 4) == 36
    assert MODULE.validation_steps(64, 9, 1, 4) == 144
    with pytest.raises(ValueError, match="aligned"):
        MODULE.validation_steps(63, 9, 4, 4)
    # Aggregate 18 is divisible by six, while each two-sample domain cannot
    # form a domain-pure global batch.
    with pytest.raises(ValueError, match="each validation domain"):
        MODULE.validation_steps(2, 9, 2, 3)


def test_validation_schedule_always_includes_final_optimizer_step():
    selected = [step for step in range(1, 1908)
                if MODULE.should_validate_step(step, 500, 1907)]
    assert selected == [1, 500, 1000, 1500, 1907]
