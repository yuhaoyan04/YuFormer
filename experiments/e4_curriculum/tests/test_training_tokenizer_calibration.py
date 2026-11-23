import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'pretrain/scripts'))
import training_tokenizer_calibration as contract

SPEC = importlib.util.spec_from_file_location('training_tokenizer_calibration',
    ROOT / 'experiments/e4_curriculum/scripts/calibrate_training_tokens.py')
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def test_independent_population_sample_measures_actual_training_tokens_including_eod(tmp_path):
    rows = [{'doc_id': i.to_bytes(16, 'big'), 'domain': domain, 'n_tokens': 10}
            for i, domain in enumerate(list(CAL.DOMAINS) * 4)]
    parent, sample = tmp_path / 'parent.parquet', tmp_path / 'sample.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parent)
    selection = CAL.sample_order(parent, 15, sample,
        population={d: {'selected_docs': 4, 'selected_tokens': 40} for d in CAL.DOMAINS},
        expected_parent_sha256=CAL.digest(parent), seed='independent-unit', phase='warmup')
    table = pq.read_table(sample)
    assert table.num_rows == 18
    assert table['position'].to_pylist() == list(range(18))
    assert set(selection['reference_tokens_by_domain'].values()) == {20}
    lengths = tmp_path / 'lengths.u32'
    np.array([21] * 18, dtype='<u4').tofile(lengths)
    measured = CAL.measure(sample, lengths)
    assert all(row == {'docs': 2, 'reference_tokens': 20,
                       'actual_tokens_with_eod': 42, 'ratio': 2.1} for row in measured.values())


def test_sample_rejects_missing_domains_and_unsorted_parent(tmp_path):
    parent, sample = tmp_path / 'parent.parquet', tmp_path / 'sample.parquet'
    pq.write_table(pa.table({'doc_id': [b'a'], 'domain': ['web'], 'n_tokens': [10]}), parent)
    with pytest.raises(ValueError, match='insufficient'):
        CAL.sample_order(parent, 5, sample,
            population={d: {'selected_docs': int(d == 'web'), 'selected_tokens': 10*int(d == 'web')}
                        for d in CAL.DOMAINS}, expected_parent_sha256=CAL.digest(parent),
            seed='independent-unit', phase='warmup')
    rows = [{'doc_id': (100-i).to_bytes(16, 'big'), 'domain': domain, 'n_tokens': 10}
            for i, domain in enumerate(CAL.DOMAINS)]
    pq.write_table(pa.Table.from_pylist(rows), parent)
    with pytest.raises(ValueError, match='strictly sorted'):
        CAL.sample_order(parent, 5, sample,
            population={d: {'selected_docs': 1, 'selected_tokens': 10} for d in CAL.DOMAINS},
            expected_parent_sha256=CAL.digest(parent), seed='independent-unit', phase='warmup')


def test_measured_ledger_rejects_zero_lengths(tmp_path):
    order, ledger = tmp_path / 'order.parquet', tmp_path / 'lengths.u32'
    pq.write_table(pa.table({'domain': list(CAL.DOMAINS), 'n_tokens': [10]*9}), order)
    np.array([0]*9, dtype='<u4').tofile(ledger)
    with pytest.raises(ValueError, match='cover every document'):
        CAL.measure(order, ledger)


@pytest.mark.parametrize('target', ['order', 'ledger'])
def test_measurement_consumes_the_hashed_fd_during_path_rollback(tmp_path, monkeypatch, target):
    order, ledger = tmp_path / 'order.parquet', tmp_path / 'lengths.u32'
    alternate, saved = tmp_path / 'alternate', tmp_path / 'saved'
    pq.write_table(pa.table({'domain': list(CAL.DOMAINS), 'n_tokens': [10]*9}), order)
    np.full(9, 21, dtype='<u4').tofile(ledger)
    if target == 'order':
        pq.write_table(pa.table({'domain': list(CAL.DOMAINS), 'n_tokens': [20]*9}), alternate)
        original, path, owner, name = pq.read_table, order, CAL.pq, 'read_table'
    else:
        np.full(9, 42, dtype='<u4').tofile(alternate)
        original, path, owner, name = np.fromfile, ledger, CAL.np, 'fromfile'
    expected_order, expected_lengths = CAL.digest(order), CAL.digest(ledger)
    def rollback(handle, *args, **kwargs):
        assert hasattr(handle, 'fileno')
        os.replace(path, saved)
        os.replace(alternate, path)
        try:
            return original(handle, *args, **kwargs)
        finally:
            os.replace(path, alternate)
            os.replace(saved, path)
    monkeypatch.setattr(owner, name, rollback)
    measurements = CAL.measure(order, ledger, expected_order_sha256=expected_order,
                               expected_lengths_sha256=expected_lengths)
    assert all(row['ratio'] == 2.1 for row in measurements.values())


def test_quality_policy_excludes_budget_but_binds_quality_and_sources():
    config = {'domains': {d: {'weight': 1/9, 'top_k': .4} for d in CAL.DOMAINS},
              'included_sources': ['source'], 'score_field': 'hybrid_v2', 'total_tokens': 10}
    before = CAL.quality_policy(config)
    config['total_tokens'] = 20
    config['domains']['web']['weight'] = .5
    assert CAL.quality_policy(config) == before
    config['domains']['web']['top_k'] = .5
    assert CAL.quality_policy(config) != before


@pytest.fixture
def measured_report(tmp_path, monkeypatch):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return path
    for relative in (contract.TOOL, contract.SAMPLER, 'pretrain/scripts/training_tokenizer_calibration.py',
                     'pretrain/scripts/materialize_tokens.py', 'pretrain/scripts/source_document_binding.py',
                     'pretrain/scripts/merge_token_parts.c'):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('tokenizer bytes')
    tokenizer_sha = CAL.digest(tokenizer)
    write('data/TOKENIZER_READY.json', {'status': 'PASS', 'tokenizer_sha256': tokenizer_sha})
    fingerprint = '0' * 64
    monkeypatch.setattr(CAL.materializer, 'source_roster_fingerprint', lambda *args: (fingerprint, {}))
    parent = {'status': 'PASS', 'artifacts': [], 'stages': {}}
    report = {'schema_version': 2, 'status': 'PASS', 'purpose': 'TRAINING_TOKENIZER_CALIBRATION',
              'includes_appended_eod': True, 'sampling': contract.SAMPLING, 'seed': 'independent-unit',
              'sampler_tool_sha256': CAL.digest(tmp_path/contract.SAMPLER),
              'tool_sha256': CAL.digest(tmp_path / contract.TOOL), 'tokenizer_sha256': tokenizer_sha,
              'minimum_reference_tokens_per_domain': 100_000_000, 'stages': {}}
    for stage, ratio in [('warmup', .8), ('stable', 1.2), ('anneal', 1.5)]:
        directory = tmp_path / stage
        directory.mkdir()
        config = {'domains': {d: {'weight': 1/9, 'top_k': .4} for d in CAL.DOMAINS}}
        config_path = tmp_path / f'{stage}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        manifest = write(stage+'-population.json', {'selected': {
            d: {'selected_docs': 100, 'selected_tokens': 100_000_000} for d in CAL.DOMAINS}})
        parent['stages'][stage] = {'config': config_path.name, 'order': f'{stage}-parent.parquet',
                                  'manifest': manifest.name}
        parent['artifacts'].append({'path': f'{stage}-parent.parquet', 'sha256': 'a' * 64})
        parent['artifacts'].append({'path': manifest.name, 'sha256': CAL.digest(manifest)})
        order, lengths = directory / 'sample.parquet', directory / 'sample.doc_lengths.u32'
        pq.write_table(pa.table({'domain': list(CAL.DOMAINS)*100, 'n_tokens': [1_000_000]*900}), order)
        np.full(900, int(ratio*1_000_000), dtype='<u4').tofile(lengths)
        meta_path = write(f'{stage}/sample.meta.json', {
            'tokenizer': str(tokenizer), 'tokenizer_sha256': tokenizer_sha,
            'total_tokens': int(ratio*900_000_000),
            'materializer_tool_sha256': CAL.digest(tmp_path / 'pretrain/scripts/materialize_tokens.py'),
            'native_merge_tool_sha256': CAL.digest(tmp_path / 'pretrain/scripts/merge_token_parts.c'),
            'order_sha256': CAL.digest(order), 'doc_lengths': {'sha256': CAL.digest(lengths)},
            'source_document_binding': {'schema_version': 4, 'roster_fingerprint': fingerprint,
                'expected_doc_ids_verified': True, 'tokenizer_bytes_verified': True,
                'order_bytes_verified': True, 'source_handles_verified': True}})
        selection = write(f'{stage}/selection.json', {'sample_sha256': CAL.digest(order),
            'parent_order_sha256': 'a'*64, 'minimum_reference_tokens_per_domain': 100_000_000,
            'sampling': contract.SAMPLING, 'seed': report['seed'], 'phase': stage, 'docs': 900,
            'sampler_tool_sha256': report['sampler_tool_sha256'],
            'population_docs_by_domain': {d: 100 for d in CAL.DOMAINS},
            'population_reference_tokens_by_domain': {d: 100_000_000 for d in CAL.DOMAINS}})
        report['stages'][stage] = {'status': 'PASS', 'quality_policy': CAL.quality_policy(config),
            'source_roster_fingerprint': fingerprint, 'domains': CAL.measure(order, lengths),
            'evidence': {str(p.relative_to(tmp_path)): CAL.digest(p) for p in (order, lengths, meta_path, selection)}}
    parent_path = write('parent.release.json', parent)
    report['parent_release'] = parent_path.name
    report['parent_release_sha256'] = CAL.digest(parent_path)
    write(contract.REPORT, report)
    return tmp_path, report, write


def test_contract_uses_distinct_training_phase_ratios_and_binds_evidence(measured_report):
    root, report, _ = measured_report
    inputs = {}
    ratios = contract.load_training_ratios(root, inputs)
    assert [ratios[s]['web'] for s in ('warmup', 'stable', 'anneal')] == [.8, 1.2, 1.5]
    assert contract.REPORT in inputs and 'stable/sample.doc_lengths.u32' in inputs


@pytest.mark.parametrize('fault', ['missing_domain', 'small_sample', 'fabricated_ratio', 'stale_tool',
                                   'tokenizer', 'ledger', 'source_roster', 'quality_policy'])
def test_contract_rejects_stale_or_incomplete_calibration(measured_report, fault):
    root, report, write = measured_report
    row = report['stages']['warmup']
    if fault == 'missing_domain':
        row['domains'].pop('web')
    elif fault == 'small_sample':
        report['minimum_reference_tokens_per_domain'] = 50_000_000
    elif fault == 'fabricated_ratio':
        row['domains']['web']['ratio'] = 99.
    elif fault == 'stale_tool':
        report['tool_sha256'] = 'b'*64
    elif fault == 'tokenizer':
        (root / 'tokenizer.json').write_text('different tokenizer')
    elif fault == 'ledger':
        (root / 'warmup/sample.doc_lengths.u32').write_bytes(b'corrupt')
    elif fault == 'source_roster':
        report['stages']['stable']['source_roster_fingerprint'] = 'c'*64
    elif fault == 'quality_policy':
        config = yaml.safe_load((root / 'warmup.yaml').read_text())
        config['domains']['web']['top_k'] = .5
        (root / 'warmup.yaml').write_text(yaml.safe_dump(config))
    write(contract.REPORT, report)
    with pytest.raises(ValueError):
        contract.load_training_ratios(root, {})


def test_historical_producer_requires_exact_bound_origin_bytes(measured_report):
    root, report, write = measured_report
    parent_path = root/report['parent_release']
    parent = json.loads(parent_path.read_text())
    old_origin = b'original data recipe'
    import hashlib
    origin = hashlib.sha256(old_origin).hexdigest()
    parent['frozen_recipe_sha256'] = origin
    write(report['parent_release'], parent)
    report['parent_release_sha256'] = CAL.digest(parent_path)
    write(contract.REPORT, report)
    name = 'pretrain/scripts/merge_token_parts.c'
    old_bytes = (root/name).read_bytes()
    old_sha = CAL.digest(root/name)
    prefix = f'data/processed/recipe_origins/{origin}'
    rows = {}
    for relative, payload, digest in ((name, old_bytes, old_sha),
            ('experiments/e4_curriculum/results/frozen_1t_recipe.json', old_origin, origin)):
        asset = f'{prefix}/objects/{digest}'
        path = root/asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows[relative] = {'archive': asset, 'sha256': digest}
    write(prefix+'/manifest.json', {'status': 'PASS_IMMUTABLE_ORIGIN_ARCHIVE',
        'frozen_recipe_sha256': origin, 'files': rows})
    (root/name).write_text('new checked writeback implementation')
    inputs = {}
    assert contract.load_training_ratios(root, inputs)['stable']['web'] == 1.2
    assert rows[name]['archive'] in inputs
    (root/rows[name]['archive']).write_bytes(b'corrupt original producer')
    with pytest.raises(ValueError, match='producer bytes'):
        contract.load_training_ratios(root, {})


def test_changed_producer_without_parent_origin_is_rejected(measured_report):
    root, _, _ = measured_report
    (root/'pretrain/scripts/merge_token_parts.c').write_text('changed')
    with pytest.raises(ValueError, match='bound parent data origin'):
        contract.load_training_ratios(root, {})
