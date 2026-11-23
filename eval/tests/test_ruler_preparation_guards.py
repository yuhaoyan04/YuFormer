import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('ruler_case_preparation_guards',
    ROOT/'eval/scripts/prepare_ruler_cases.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.mark.parametrize('tasks,lengths', [(['niah_single_1']*13, [4096]),
                                        (['niah_single_1'], [4096, 4096]),
                                        ([], [4096]), (['niah_single_1'], [])])
def test_duplicate_or_empty_rosters_are_rejected_before_publishing_cases(tmp_path, tasks, lengths):
    output = tmp_path/'cases'
    with pytest.raises(ValueError, match='supported context roster'):
        PREPARE.prepare(tmp_path, tmp_path/'upstream', output,
            lengths=lengths, tasks=tasks, samples=500, seed=42)
    assert not output.exists()


def test_essay_sentence_model_must_be_explicit_and_complete(tmp_path, monkeypatch):
    class Distribution:
        version = 'test'
        files = []
    monkeypatch.setattr(PREPARE.importlib.metadata, 'distribution', lambda _: Distribution())
    with pytest.raises(ValueError, match='explicit frozen NLTK'):
        PREPARE.runtime_dependencies(None, needs_sentences=True)
    with pytest.raises(ValueError, match='asset is absent'):
        PREPARE.runtime_dependencies(tmp_path, needs_sentences=True)


class CountingTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(range(len(text))))


def case(index, length):
    return {'index': index, 'input': 'a'*length, 'outputs': ['reference']}


def test_diagnostic_selection_preserves_first_valid_original_identities():
    rows = [case(0, 20), case(1, 80), case(2, 85), case(3, 90)]
    selected, lengths = PREPARE.select_length_conditioned_cases(rows, CountingTokenizer(),
        seq=100, generation_tokens=10, samples=2)
    assert selected == rows[1:3]
    assert [r['index'] for r in selected] == [1, 2]
    assert lengths == [80, 85]


def test_oversampling_cannot_relax_the_original_length_floor():
    with pytest.raises(ValueError, match='unchanged75pct'):
        PREPARE.select_length_conditioned_cases([case(0, 74), case(1, 80)], CountingTokenizer(),
            seq=100, generation_tokens=10, samples=2)


def test_over_budget_raw_case_is_rejected_even_after_enough_cases_were_selected():
    with pytest.raises(ValueError, match='hard generation budget'):
        PREPARE.select_length_conditioned_cases([case(0, 80), case(1, 99)], CountingTokenizer(),
            seq=100, generation_tokens=10, samples=1)
