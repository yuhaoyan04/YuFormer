#!/usr/bin/env python3
"""Wait for real stream gates, then audit and qualify the selected training graph."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root/'pretrain/scripts'), str(root/'pretrain/src')]
    from qualification_identity import qualification_identity, report_matches
    from yuformer_pretrain.lease import RunLease
    health = root/'data/health/post-encoding-qualification.json'
    python = Path('/opt/yuformer-torchtitan/bin/python')
    data_python = Path('/usr/bin/python')
    paths = [root/row['path'] for row in qualification_identity(root)['files']
             if not row.get('missing') and row['path'] != 'data/processed/audits/pretrain_1t_token_streams.json']
    paths += [root/name for name in (
        'experiments/e4_curriculum/scripts/audit_1t_token_streams.py',
        'experiments/e4_curriculum/scripts/publish_training_block_manifest.py',
        'experiments/e4_curriculum/scripts/prepare_training_blocks.py',
        'experiments/e4_curriculum/scripts/training_block_permutation.py',
        'experiments/e4_curriculum/scripts/audit_training_block_permutation.py',
        'experiments/e4_curriculum/scripts/calibrate_training_tokens.py',
        'experiments/e4_curriculum/scripts/adopt_training_recipe.py',
        'experiments/e4_curriculum/scripts/frozen_training_evidence.py',
        'experiments/e4_curriculum/scripts/sustained_memory_review.py',
        'experiments/e4_curriculum/scripts/allocator_retry_review.py',
        'experiments/e4_curriculum/scripts/validation_source_proof.py',
        'experiments/e4_curriculum/scripts/training_source_windows.py',
        'experiments/e4_curriculum/scripts/temporal_order_admission.py',
        'experiments/e4_curriculum/scripts/qualify_native_merge.py',
        'experiments/e4_curriculum/scripts/overlap_training_windows.py',
        'experiments/e4_curriculum/scripts/audit_1t_overlap.py',
        'pretrain/scripts/merge_token_parts.c',
        'data/health/validation-source-proof-archive.json',
        'data/processed/audits/training_tokenizer_calibration.json',
        'experiments/e2_architecture/results/model-scale-production-selection.json',
        'data/health/pretrain-user-full-pretrain-authorization.json',
        'pretrain/scripts/run_production_recovery.py')]
    paths.append(Path(__file__).resolve())
    paths.append(Path(__file__).with_name('audit_production_stage_order.py'))
    pinned = {str(path): digest(path) for path in paths}
    state = {'schema_version': 1, 'pid': os.getpid(), 'status': 'WAITING_REAL_TOKENIZATION_PASS',
             'formal_pretrain_started': False, 'launches_formal_pretrain': False,
             'pinned_inputs': pinned, 'stages': {}}

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = health.with_name(health.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2)+'\n')
        os.replace(temporary, health)

    def unchanged():
        for name, expected in pinned.items():
            if digest(Path(name)) != expected:
                raise ValueError('pinned prelaunch input changed: '+name)

    def run(stage, interpreter, relative, *extra):
        unchanged()
        log = root/f'report/logs/post-encoding-{stage}.log'
        state['stages'][stage] = {'status': 'RUNNING', 'log': str(log)}
        save(status='RUNNING', stage=stage)
        with log.open('a') as output:
            child = subprocess.Popen([str(interpreter), str(root/relative), *map(str, extra)],
                                     cwd=root, stdout=output, stderr=subprocess.STDOUT)
            state['stages'][stage]['pid'] = child.pid
            save()
            while child.poll() is None:
                save()
                time.sleep(15)
        state['stages'][stage].update(status='PASS' if child.returncode == 0 else 'FAIL',
                                      returncode=child.returncode)
        save()
        if child.returncode:
            raise ValueError('real gate failed: '+stage+'; see '+str(log))
        unchanged()

    with RunLease(root/'data/health/post-encoding-qualification-lease',
                  'selected-model-post-encoding-qualification'):
        try:
            while True:
                unchanged()
                tokens = json.loads((root/'data/health/pretrain-1t-tokenization.json').read_bytes())
                status = tokens.get('status')
                save(tokenization_status=status,
                     tokenization_jobs={k: v.get('status') for k, v in tokens.get('jobs', {}).items()})
                if status == 'PASS':
                    expected = {'warmup', 'stable', 'anneal'} | {
                        'val_'+name for name in ('web', 'code', 'math', 'chinese', 'academic',
                                                'books', 'encyclopedia', 'qa', 'multilingual')}
                    if (set(tokens['expected_jobs']) != expected or
                            any(tokens['jobs'].get(name, {}).get('status') != 'PASS' for name in expected)):
                        raise ValueError('tokenization PASS lacks all twelve actual stream jobs')
                    break
                if status not in {'IN_PROGRESS'}:
                    raise ValueError('tokenization gate requires attention: '+str(status))
                time.sleep(45)
            run('canonical_native_writeback', data_python,
                'experiments/e4_curriculum/scripts/qualify_native_merge.py', '--root', root)
            run('full_byte_audit', data_python, 'experiments/e4_curriculum/scripts/audit_1t_token_streams.py')
            sys.path.insert(0, str(root/'experiments/e4_curriculum/scripts'))
            from validation_source_proof import require_validation_source_proof
            physical = json.loads((root/'data/processed/audits/pretrain_1t_token_streams.json').read_bytes())
            warmup_meta = json.loads((root/'data/token_streams/pretrain-1t/warmup.meta.json').read_bytes())
            _, validation_archive = require_validation_source_proof(
                root, physical, warmup_meta['source_document_binding']['roster_fingerprint'])
            state['stages']['validation_source_proof'] = {'status': 'PASS', 'archive': validation_archive}
            save()
            run('temporal_order_recount', data_python,
                '/opt/yuformer-build/audit_production_stage_order.py', '--root', root)
            run('production_block_manifest', data_python,
                'experiments/e4_curriculum/scripts/publish_training_block_manifest.py', '--directory',
                root/'data/processed/training_permutations/pretrain-1t-dd284efd7c715544-b8')
            run('temporal_order_admission', data_python,
                'experiments/e4_curriculum/scripts/temporal_order_admission.py', '--root', root, '--master',
                root/'data/processed/training_permutations/pretrain-1t-dd284efd7c715544-b8/manifest.json')
            deadline = time.monotonic()+600
            while True:
                inventory = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                           capture_output=True, text=True, check=True, timeout=15)
                if not inventory.stdout.strip():
                    break
                save(status='WAITING_GPU_IDLE', gpu_compute_pids=inventory.stdout.strip())
                if time.monotonic() >= deadline:
                    raise ValueError('GPUs remain occupied; no process was interrupted')
                time.sleep(15)
            run('gpu_qualification', python, 'pretrain/scripts/run_torchtitan_qualification.py',
                '--gpus', '0,1,2,3,4,5,6,7')
            qualification = json.loads((root/'data/health/torchtitan-qualification.json').read_bytes())
            if qualification.get('status') != 'PASS' or not report_matches(qualification, qualification_identity(root)):
                raise ValueError('GPU qualification output is stale or incomplete')
            run('production_recovery', python, 'pretrain/scripts/run_production_recovery.py')
            from run_production_recovery import validate_report
            validate_report(root)
            save(status='PASS_AWAITING_TRAINING_RECIPE_ADOPTION', stage='qualified',
                 qualification_identity=qualification_identity(root)['sha256'])
            return 0
        except Exception as error:
            save(status='FAIL_CLOSED', error=repr(error))
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
