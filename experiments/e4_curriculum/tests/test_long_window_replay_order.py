import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'experiments/e4_curriculum/scripts'/file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_auditor_reproduces_the_actual_producer_byte_separator():
    producer=load('_real_long_window_producer','prepare_long_window_release_v2.py')
    auditor=load('_independent_long_window_auditor','audit_long_window_release_v2.py')
    seed='frozen-test-seed'
    for seq in (16384,65536):
        expected=sorted(range(64),key=lambda j:producer.key(seed,f'short:{seq}:{j}'))
        assert auditor.replay_permutation(seq,64,seed)==expected
        assert sorted(expected)==list(range(64))


def test_replay_permutation_does_not_reuse_a_different_context_stage_order():
    auditor=load('_independent_length_order_auditor','audit_long_window_release_v2.py')
    assert auditor.replay_permutation(16384,64,'seed')!=auditor.replay_permutation(65536,64,'seed')
