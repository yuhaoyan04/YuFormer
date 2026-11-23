import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/e4_curriculum/scripts'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('_staged_uniform_calibration_builder', ROOT/'experiments/e4_curriculum/prototypes/calibrate_training_tokens.py')
contract = load('_staged_uniform_calibration_loader', ROOT/'experiments/e4_curriculum/prototypes/pretrain_scripts/training_tokenizer_calibration.py')
fixtures = load('_legacy_calibration_test_fixture', Path(__file__).with_name('test_training_tokenizer_calibration.py'))
measured_report = fixtures.measured_report
from test_uniform_selected_sample import population_file


@pytest.fixture
def independent_report(measured_report):
    root, report, write = measured_report
    shutil.copyfile(ROOT/'experiments/e4_curriculum/prototypes/calibrate_training_tokens.py', root/contract.TOOL)
    (root/contract.SAMPLER).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/contract.SAMPLER, root/contract.SAMPLER)
    report.update(schema_version=2, sampling=contract.SAMPLING, seed='independent-regression',
                  tool_sha256=builder.digest(root/contract.TOOL), sampler_tool_sha256=builder.digest(root/contract.SAMPLER))
    parent_path = root/report['parent_release']
    parent = json.loads(parent_path.read_bytes())
    for phase, row in report['stages'].items():
        manifest = write(phase+'-population.json', {'selected': {
            d: {'selected_docs': 100, 'selected_tokens': 100_000_000} for d in builder.DOMAINS}})
        parent['stages'][phase]['manifest'] = manifest.name
        parent['artifacts'].append({'path': manifest.name, 'sha256': builder.digest(manifest)})
        path = root/phase/'selection.json'
        selection = json.loads(path.read_bytes())
        selection.update(sampling=contract.SAMPLING, seed=report['seed'], phase=phase,
            sampler_tool_sha256=report['sampler_tool_sha256'], docs=900,
            population_docs_by_domain={d: 100 for d in builder.DOMAINS},
            population_reference_tokens_by_domain={d: 100_000_000 for d in builder.DOMAINS})
        path.write_text(json.dumps(selection, sort_keys=True))
        row['evidence'][str(path.relative_to(root))] = builder.digest(path)
    parent_path.write_text(json.dumps(parent, sort_keys=True))
    report['parent_release_sha256'] = builder.digest(parent_path)
    write(contract.REPORT, report)
    return root, report, write


def test_staged_builder_samples_across_entire_population_instead_of_docid_prefix(tmp_path):
    parent, population = population_file(tmp_path, n=1000)
    destination = tmp_path/'independent.parquet'
    result = builder.sample_order(parent, 100, destination, population=population,
        expected_parent_sha256=builder.digest(parent), seed='independent-regression', phase='warmup')
    table = pq.read_table(destination)
    ranks = [int.from_bytes(v, 'big')//9 for v in table['doc_id'].to_pylist()]
    assert min(ranks) < 100 and max(ranks) > 900
    assert result['sampling'] == contract.SAMPLING
    assert all(n == 1000 for n in result['population_docs_by_domain'].values())
    with pytest.raises(TypeError):
        builder.sample_order(parent, 100, tmp_path/'prefix.parquet')


def test_staged_contract_accepts_source_bound_independent_samples(independent_report):
    root, _, _ = independent_report
    captured = {}
    result = contract.load_training_ratios(root, captured)
    assert [result[p]['web'] for p in ('warmup', 'stable', 'anneal')] == [.8, 1.2, 1.5]
    assert contract.SAMPLER in captured
    assert 'stable-population.json' in captured


@pytest.mark.parametrize('fault', ['prefix_report', 'schema1', 'missing_seed', 'wrong_sampler',
                                   'prefix_selection', 'wrong_phase', 'wrong_population'])
def test_staged_contract_rejects_prefix_or_unbound_sampling(independent_report, fault):
    root, report, write = independent_report
    if fault == 'prefix_report': report['sampling'] = 'per-domain-hash-prefix'
    elif fault == 'schema1': report['schema_version'] = 1
    elif fault == 'missing_seed': report.pop('seed')
    elif fault == 'wrong_sampler': report['sampler_tool_sha256'] = '0'*64
    else:
        path = root/'warmup/selection.json'
        selection = json.loads(path.read_bytes())
        if fault == 'prefix_selection': selection.pop('sampling')
        elif fault == 'wrong_phase': selection['phase'] = 'stable'
        else: selection['population_docs_by_domain']['web'] += 1
        path.write_text(json.dumps(selection))
        report['stages']['warmup']['evidence'][str(path.relative_to(root))] = builder.digest(path)
    write(contract.REPORT, report)
    with pytest.raises(ValueError):
        contract.load_training_ratios(root, {})
