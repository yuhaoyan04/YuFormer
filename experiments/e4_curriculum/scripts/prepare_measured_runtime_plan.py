"""Bind actual stage speeds and derive an execution budget before quality admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'pretrain/scripts'), str(ROOT/'experiments/e2_architecture/scripts')]
import benchmark_model_sizes as bench
from adopt_training_recipe import training_execution_contract
from audit_training_progress import audit, verify_reference_schedule
from allocator_retry_review import require_no_allocator_retries
from sustained_memory_review import review_memory_log
from freeze_1t_recipe import source_sha256
from frozen_training_evidence import publish_checked
from runtime_limited_plan import derive_runtime_limited_plan, resolve_effective_budget


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('preserve prior immutable measured runtime proposals')
    health_path = root/'data/health/final-deterministic-production-controls.json'
    health = json.loads(health_path.read_bytes())
    if health.get('status') != 'COMPLETE_REQUIRES_PAIRED_PRIMARY_AND_EXPANDED_QUALITY_REVIEW':
        raise ValueError('complete both actual200-update controls before deriving the execution budget')
    long_path = root/'data/health/deterministic-flash-long-profiles.json'
    long = json.loads(long_path.read_bytes())
    if long.get('status') != 'COMPLETE_SYSTEM_PROFILES_NOT_LONG_CAPABILITY':
        raise ValueError('both current long-stage system profiles must finish')
    budget_path = root/'data/health/user-total-pretrain-and-length-budget.json'
    nominal = json.loads(budget_path.read_bytes())
    source = source_sha256(root)
    sources = {str(p.relative_to(root)): digest(p) for p in (health_path, long_path, budget_path)}
    # Preserve producer bytes independently of later orchestration edits.
    producer_files = [Path(__file__), Path(bench.__file__),
        root/'experiments/e4_curriculum/scripts/runtime_limited_plan.py',
        root/'experiments/e4_curriculum/scripts/global_training_budget.py',
        root/'experiments/e4_curriculum/scripts/adopt_training_recipe.py',
        root/'pretrain/scripts/audit_training_progress.py']
    provenance = {}
    for path in producer_files:
        sha = digest(path)
        snapshot = root/'report/source_snapshots'/(path.stem+'-'+sha+path.suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        publish_checked(snapshot, path.read_bytes(), immutable=True)
        sources[str(snapshot.relative_to(root))] = sha
        provenance[str(path.relative_to(root))] = str(snapshot.relative_to(root))
    measurements, reviews = {}, {}
    for stage in nominal['stages']:
        seq = stage['seq_len']
        name = 'larger_lr1e4_b8' if seq == 4096 else f'context{seq}_det_flash_24'
        row = health['cases'][name] if seq == 4096 else long['cases'][name]
        cohort = 'final_deterministic_production_controls_20261115' if seq == 4096 else 'deterministic_flash_long_profiles_20261115'
        run = root/'data/experiments'/cohort/name
        identity = json.loads((run/'run_identity.json').read_bytes())
        result = json.loads((run/'result.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        steps, skip = (200, 50) if seq == 4096 else (24, 8)
        evaluated = bench.evaluate(result, steps=steps, skip_steps=skip)
        if (row['evaluation']['status'] != 'PASS_SCREENING' or evaluated != row['evaluation']
                or digest(run/'result.json') != row['result_sha256']
                or identity.get('source_sha256') != source or config['parallel']['world_size'] != 8
                or config['data']['seq_len'] != seq or row['parameters'] != 1516600064
                or config['train'].get('deterministic') is not True
                or config['attention']['sdpa_backend'] != 'flash_flex_swa'):
            raise ValueError('actual runtime point has another result/source/model/numeric policy')
        log = Path(row['log'])
        raw = log.read_bytes()
        review_memory_log(raw, steps=steps, skip_steps=skip)
        require_no_allocator_retries(raw)
        runtime = audit(run, check_checkpoint=True)
        runtime['reference_scheduler'] = verify_reference_schedule(config)
        if (runtime['checkpoint']['status'] != 'PASS' or runtime['checkpoint']['step'] != steps
                or runtime['checkpoint']['global_tokens_seen'] != steps*4194304):
            raise ValueError('actual runtime checkpoint/cursor failed')
        model_path = Path(identity['model_configs'])
        if digest(model_path) != identity['model_configs_sha256']:
            raise ValueError('actual measured model bytes changed')
        model = yaml.safe_load(model_path.read_bytes())['presets'][config['model']['preset']]
        measurements[stage['name']] = {'seq_len': seq, 'world_size': 8, 'parameters': row['parameters'],
            'p10_tps_per_device': evaluated['p10_tps_per_device'], 'run_dir': str(run.relative_to(root)),
            'memory_log': str(log.relative_to(root)), 'execution_contract': training_execution_contract(config),
            'parallel': config['parallel'], 'model': model}
        reviews[stage['name']] = runtime
        for path in (run/'result.json', run/'run_identity.json', run/'resolved_config.yaml', model_path,
                     log, run/f'checkpoint/step-{steps}/.metadata'):
            sources[str(path.relative_to(root))] = digest(path)
    proposal = derive_runtime_limited_plan(nominal, measurements)
    resolve_effective_budget(nominal, proposal)
    if source_sha256(root) != source or any(digest(root/name) != sha for name, sha in sources.items()):
        raise ValueError('runtime inputs changed while independently deriving the budget')
    document = {'schema_version': 1, 'status': 'PREPARED_MEASURED_RUNTIME_ALLOCATION_REQUIRES_QUALITY_AND_DATA_ADMISSION',
        'production_qualified': False, 'formal_pretrain_started': False,
        'nominal_budget_sha256': digest(budget_path), 'training_source_sha256': source,
        'proposal': proposal, 'runtime_reviews': reviews, 'source_inputs': sources,
        'producer_provenance': provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_checked(output, (json.dumps(document, sort_keys=True, indent=2)+'\n').encode(), immutable=True)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.root, args.output)
    print(json.dumps({key: report['proposal'][key] for key in
        ('proposed_total_tokens', 'token_reduction_fraction', 'total_days', 'planning_ceiling_days')}, sort_keys=True))
