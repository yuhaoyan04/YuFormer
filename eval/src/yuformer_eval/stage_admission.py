"""Issue stage capability decisions only from complete paired native evidence."""
import json
import os
from pathlib import Path

from .measurements import digest, load_measurements, merge_pins
from .repoqa_scoring import scorer
from .stage_scores import code_scores, decide, intrinsic_scores, require_policy, ruler_scores

ROOT = Path(__file__).resolve().parents[3]
SUITES = {'ruler', 'long_bpb', 'long_qa_code', 'short_core'}


def bound_json(binding, pins):
    path = Path(binding['file'])
    if not path.is_absolute() or digest(path) != binding['sha256']:
        raise ValueError('stage evaluation input byte binding changed')
    merge_pins(pins, {str(path): binding['sha256']})
    return json.loads(path.read_bytes())


def complete_checkpoint(binding, pins, *, seq_len, budget):
    report = bound_json(binding, pins)
    producer = ROOT/'eval/scripts/audit_completed_native_checkpoint.py'
    selected = [stage for stage in budget['stages'] if stage['seq_len'] == seq_len]
    if (len(selected) != 1
            or report.get('status') != 'PASS_ACTUAL_COMPLETE_PRODUCTION_CHECKPOINT_FINAL_VALIDATION_AND_NATIVE_COUNTERS'
            or report.get('seq_len') != seq_len or report.get('step') != selected[0]['updates']
            or report.get('tokens') != selected[0]['tokens']
            or report.get('producer') != str(producer) or report.get('producer_sha256') != digest(producer)):
        raise ValueError('stage capability requires its actual complete production checkpoint audit')
    source_inputs = report.get('source_inputs')
    if not isinstance(source_inputs, dict) or source_inputs.get(str(producer)) != digest(producer):
        raise ValueError('complete checkpoint audit lacks its actual native counter verifier')
    merge_pins(pins, source_inputs)
    handoff = Path(report['checkpoint_handoff_file'])
    if digest(handoff) != report['checkpoint_handoff_sha256']:
        raise ValueError('complete production checkpoint handoff changed')
    checkpoint = Path(report['checkpoint'])
    artifacts = report['checkpoint_artifacts']
    if (not checkpoint.is_absolute() or not artifacts
            or {p.name for p in checkpoint.iterdir() if p.is_file()} != {row['file'] for row in artifacts}):
        raise ValueError('complete checkpoint native artifact roster changed')
    for row in artifacts:
        name = row['file']
        if Path(name).name != name:
            raise ValueError('checkpoint artifact filename escapes its checkpoint')
        stat = (checkpoint/name).stat()
        actual = {'bytes': stat.st_size, 'device': stat.st_dev, 'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}
        if actual != row['stat']:
            raise ValueError('actual native checkpoint changed after its complete byte audit')
    return report


def create_admission(spec_file):
    spec_file = Path(spec_file).resolve()
    spec = json.loads(spec_file.read_bytes())
    seq = spec['seq_len']
    if seq not in (16384, 65536) or set(spec['baseline']['receipts']) != SUITES or set(spec['candidate']['receipts']) != SUITES:
        raise ValueError('production context admission requires both actual checkpoints and all four suites')
    pins = {str(spec_file): digest(spec_file)}
    cohort = bound_json(spec['cohort'], pins)
    if (cohort.get('status') != 'FROZEN_CONTEXT_STAGE_CASES_AND_QUANTITATIVE_POLICY_NOT_CAPABILITY_ADMISSION'
            or set(cohort['cases']) != SUITES):
        raise ValueError('actual cases and quantitative policy must have been frozen before measurements')
    policy = bound_json(cohort['policy'], pins)
    require_policy(policy)
    budget = bound_json(cohort['approved_budget'], pins)
    plan = bound_json_or_yaml(cohort['production_stage_plan'], pins)
    if (plan['stage_progression_evaluation']['short_loss_relative_regression_margin'] != 0.
            or set(plan['stage_progression_evaluation']['required_suites']) != SUITES):
        raise ValueError('frozen stage progression requirements differ from capability policy')
    for binding in cohort['independent_case_admissions']:
        audit = bound_json(binding, pins)
        if not audit.get('status', '').startswith('PASS_') or not audit.get('source_inputs'):
            raise ValueError('all actual evaluation case sources require independent admission')
        merge_pins(pins, audit['source_inputs'])
    baseline = complete_checkpoint(spec['baseline']['completed_audit'], pins,
        seq_len=4096 if seq == 16384 else 16384, budget=budget)
    candidate = complete_checkpoint(spec['candidate']['completed_audit'], pins, seq_len=seq, budget=budget)
    if (baseline['checkpoint_handoff_sha256'] == candidate['checkpoint_handoff_sha256']
            or baseline['run_identity_sha256'] == candidate['run_identity_sha256']
            or baseline['tokenizer_sha256'] != candidate['tokenizer_sha256']):
        raise ValueError('paired capability comparison requires distinct actual baseline/candidate and the same tokenizer')
    source = Path(spec['repoqa_source']).resolve()
    score_function, scoring_pins = scorer(source)
    merge_pins(pins, scoring_pins)
    dataset_path = source/'repoqa-2024-06-23.json'
    dataset = json.loads(dataset_path.read_bytes())
    merge_pins(pins, {str(dataset_path): digest(dataset_path)})
    metrics = {'baseline': {}, 'candidate': {}}
    for role, checkpoint in (('baseline', baseline), ('candidate', candidate)):
        for suite in sorted(SUITES):
            case_binding = cohort['cases'][suite]
            case_manifest = bound_json(case_binding, pins)
            if case_manifest['tokenizer_sha256'] != checkpoint['tokenizer_sha256']:
                raise ValueError('actual checkpoint tokenizer differs from the fixed evaluation cases')
            manifest, kind, rows, source_inputs = load_measurements(case_binding['file'], spec[role]['receipts'][suite],
                seq_len=seq, handoff_file=checkpoint['checkpoint_handoff_file'], expected_workers=policy['workers'])
            merge_pins(pins, source_inputs)
            if suite == 'ruler':
                if kind != 'ruler' or manifest['revision'] != policy[suite]['revision']:
                    raise ValueError('RULER original source revision mismatch')
                value = ruler_scores(rows, samples_per_task=policy[suite]['samples_per_task'])
            elif suite in ('short_core', 'long_bpb'):
                if kind != 'nll':
                    raise ValueError('intrinsic evaluation must use actual native NLL measurements')
                value = intrinsic_scores(rows, domains=policy[suite]['domains'],
                    samples_per_domain=policy[suite]['samples_per_domain'], byte_normalized=suite == 'long_bpb',
                    target_tokens=policy[suite]['target_tokens_per_case'] if suite == 'short_core' else None)
            else:
                if (kind != 'repoqa' or manifest['source_revision'] != policy[suite]['revision']
                        or manifest.get('reference_self_consistency_required') is not True
                        or manifest['score_protocol']['similarity_threshold'] != .8):
                    raise ValueError('RepoQA requires the original scorer and scorable immutable reference cohort')
                value = code_scores(rows, languages=policy[suite]['languages'],
                    samples_per_language=policy[suite]['samples_per_language'], dataset=dataset,
                    score_function=score_function)
            metrics[role][suite] = value
    suites = decide(policy, metrics['baseline'], metrics['candidate'])
    for path in (Path(__file__), ROOT/'eval/src/yuformer_eval/stage_scores.py',
                 ROOT/'eval/scripts/admit_context_stage.py'):
        merge_pins(pins, {str(path.resolve()): digest(path)})
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('actual paired capability evidence changed during admission')
    return {'status': 'PASS_STAGE_LONG_AND_SHORT_EVALUATION' if all(row['status'] == 'PASS' for row in suites.values())
            else 'FAIL_STAGE_LONG_AND_SHORT_EVALUATION', 'scope': 'PRODUCTION',
        'protocol_scope': policy['scope'], 'seq_len': seq, 'step': candidate['step'],
        'checkpoint_handoff_sha256': candidate['checkpoint_handoff_sha256'],
        'run_identity_sha256': candidate['run_identity_sha256'], 'baseline_checkpoint_bound': True,
        'baseline_checkpoint_handoff_sha256': baseline['checkpoint_handoff_sha256'],
        'baseline_run_identity_sha256': baseline['run_identity_sha256'],
        'cohort_sha256': spec['cohort']['sha256'], 'policy_sha256': cohort['policy']['sha256'],
        'spec_file': str(spec_file), 'spec_sha256': digest(spec_file), 'suites': suites, 'source_inputs': pins}


def bound_json_or_yaml(binding, pins):
    import yaml
    path = Path(binding['file'])
    if not path.is_absolute() or digest(path) != binding['sha256']:
        raise ValueError('production stage plan byte binding changed')
    merge_pins(pins, {str(path): binding['sha256']})
    return yaml.safe_load(path.read_bytes())


def publish_admission(spec_file, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve previous capability measurements and admissions')
    report = create_admission(spec_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write((json.dumps(report, sort_keys=True, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    return report


def verify_admission(path):
    report = json.loads(Path(path).read_bytes())
    if report.get('status') != 'PASS_STAGE_LONG_AND_SHORT_EVALUATION':
        raise ValueError('stage capability admission has not passed')
    if report != create_admission(report['spec_file']):
        raise ValueError('capability admission differs from a complete recomputation of actual paired evidence')
    return report
