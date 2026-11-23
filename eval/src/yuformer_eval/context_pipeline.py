"""Coordinate the already-authorized frozen 4K/16K/64K production chain."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from .measurements import digest, load_measurements
from .stage_admission import complete_checkpoint


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def formal_completion_state(run):
    run = Path(run)
    complete_file, attempt_file = run/'training_complete.json', run/'current_attempt.json'
    if not attempt_file.is_file():
        raise ValueError('authorized formal pretrain has no actual current attempt')
    attempt = json.loads(attempt_file.read_bytes())
    if attempt.get('status') == 'RUNNING':
        return 'WAITING_FOR_ACTUAL_4K_COMPLETION'
    if not complete_file.is_file():
        if attempt.get('status') != 'RUNNING':
            raise ValueError('formal pretrain stopped without a successful terminal artifact')
        return 'WAITING_FOR_ACTUAL_4K_COMPLETION'
    complete = json.loads(complete_file.read_bytes())
    if (complete.get('status') != 'PASS' or complete.get('returncode') != 0
            or complete.get('attempt_id') != attempt.get('attempt_id')
            or complete.get('identity_sha256') != attempt.get('identity_sha256')
            or attempt.get('status') != 'PASS'):
        raise ValueError('formal pretrain terminal state is failed, stale or belongs to another attempt')
    return 'ACTUAL_4K_PROCESS_COMPLETED_REQUIRES_RESULT_AND_NATIVE_AUDIT'


class ContextPipeline:
    def __init__(self, config_file):
        self.config_file = Path(config_file).resolve()
        self.config_sha = digest(self.config_file)
        self.config = json.loads(self.config_file.read_bytes())
        self.lease = None
        cfg = self.config
        self.root, self.output = Path(cfg['root']), Path(cfg['output'])
        if (not self.root.is_absolute() or not self.output.is_absolute() or cfg['gpus'] != list(range(8))
                or not all(Path(cfg[name]).is_absolute() for name in
                           ('training_python', 'scoring_python', 'formal_run', 'base_env', 'repoqa_source'))):
            raise ValueError('chain dependencies and the exact eight-GPU roster must be declared explicitly')
        self.cohort = self.bound(cfg['cohort'])
        self.budget = self.bound(self.cohort['approved_budget'])
        import yaml
        plan_file = Path(self.cohort['production_stage_plan']['file'])
        if digest(plan_file) != self.cohort['production_stage_plan']['sha256']:
            raise ValueError('frozen production stage plan changed')
        self.plan = yaml.safe_load(plan_file.read_bytes())
        self.base_env = {}
        for line in Path(cfg['base_env']).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                if key in self.base_env or '${' in value:
                    raise ValueError('base dependency environment is ambiguous or unresolved')
                self.base_env[key] = value
        if self.base_env.get('YUFORMER_PYTHON') != cfg['training_python']:
            raise ValueError('chain training interpreter differs from the frozen training environment')
        self.state = {'status': 'INITIALIZING_FROZEN_CONTEXT_CHAIN', 'pid': os.getpid(),
            'scope': 'ACTUAL_4K_WSD_PLUS_16K_PLUS_64K_WITH_REAL_CAPABILITY_GATES',
            'config_file': str(self.config_file), 'config_sha256': self.config_sha,
            'formal_run': cfg['formal_run'], 'total_target_labels': self.budget['actual_total_tokens'],
            'runtime_ceiling_days': 33, 'stages': {}}
        self.guard()

    @staticmethod
    def bound(binding):
        path = Path(binding['file'])
        if not path.is_absolute() or digest(path) != binding['sha256']:
            raise ValueError('frozen chain input binding failed')
        return json.loads(path.read_bytes())

    def guard(self):
        if self.lease is not None:
            self.lease.renew()
        if digest(self.config_file) != self.config_sha:
            raise ValueError('active chain controller configuration changed')
        if not self.config.get('source_inputs'):
            raise ValueError('chain controller requires its full declared execution source closure')
        for name, sha in self.config['source_inputs'].items():
            if not Path(name).is_absolute() or digest(name) != sha:
                raise ValueError('frozen chain execution source changed: '+name)

    def publish(self, status=None, **fields):
        if status is not None:
            self.state['status'] = status
        self.state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
        atomic(self.config['health'], self.state)

    def environment(self, *, cuda=''):
        env = dict(os.environ)
        for key in ('YUFORMER_RUN_LEASE_HELD', 'YUFORMER_RUN_LEASE_TOKEN', 'YUFORMER_RUN_LEASE_GENERATION',
                    'YUFORMER_ATTEMPT_ID', 'YUFORMER_OVERRIDE_GPUS_PER_NODE', 'YUFORMER_ALLOW_TOPOLOGY_RESHARD',
                    'EXTRA_ARGS', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
            env.pop(key, None)
        env.update(self.base_env)
        env.pop('YUFORMER_TRAINING_BLOCK_MANIFEST', None)
        env.update(CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env['PYTHONPATH'] = os.pathsep.join(str(self.root/p) for p in
            ('pretrain/backends/torchtitan', 'model/src', 'midtrain/src', 'pretrain/src', 'eval/src'))
        return env

    def command(self, name, argv, *, env=None, accepted=(0,), training=False):
        self.guard()
        log = self.output/'logs'/(name+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as handle:
            process = subprocess.Popen(list(map(str, argv)), cwd=self.root,
                env=self.environment() if env is None else env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.publish(active_command=name, active_pid=process.pid, active_log=str(log))
            last_publish = time.monotonic()
            deadline = None if training else time.monotonic()+self.config['auxiliary_timeout_seconds']
            try:
                while process.poll() is None:
                    self.guard()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ValueError(name+' exceeded its declared auxiliary wallclock bound')
                    if time.monotonic()-last_publish >= 30:
                        self.publish(active_log_bytes=log.stat().st_size)
                        last_publish = time.monotonic()
                    time.sleep(5)
                if process.returncode not in accepted:
                    raise ValueError(f'{name} failed with rc={process.returncode}; see {log}')
            except BaseException:
                if process.poll() is None:
                    self.terminate_owned(process)
                raise
        self.publish(active_command=None, active_pid=None)

    def terminate_owned(self, process):
        if process.poll() is not None:
            return
        import sys
        sys.path.insert(0, str(self.root/'pretrain/scripts'))
        from process_control import terminate_process_tree
        terminate_process_tree(process.pid)
        process.wait(timeout=30)

    def cleanup_owned(self):
        import sys
        sys.path.insert(0, str(self.root/'pretrain/scripts'))
        from process_control import descendant_processes, terminate_process_tree
        targets = descendant_processes(os.getpid())
        for pid, _ in reversed(targets):
            if pid != os.getpid():
                terminate_process_tree(pid)

    def wait_idle(self):
        import sys
        sys.path.insert(0, str(self.root/'pretrain/scripts'))
        from gpu_resource_policy import allowed_gpu_indices
        deadline = time.monotonic()+self.config['idle_timeout_seconds']
        while True:
            self.guard()
            if allowed_gpu_indices(self.root/'data/health') != set(range(8)):
                raise ValueError('all eight physical GPUs must remain authorized')
            apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=30, check=True)
            inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=30, check=True)
            rows = [tuple(int(value.strip()) for value in line.split(','))
                    for line in inventory.stdout.splitlines() if line.strip()]
            if not apps.stdout.strip() and {r[0] for r in rows} == set(range(8)) and all(r[1] <= 1024 for r in rows):
                return
            self.publish(waiting_for_eight_idle_gpus=True, gpu_inventory=rows)
            if time.monotonic() >= deadline:
                raise ValueError('exclusive eight-GPU allocation did not become idle within the declared timeout')
            time.sleep(30)

    def ensure_result(self, run):
        result = run/'result.json'
        if not result.exists():
            self.command('collect-'+run.name, [self.config['training_python'],
                self.root/'pretrain/scripts/collect_torchtitan_result.py', run])
        if json.loads(result.read_bytes()).get('status') != 'PASS':
            raise ValueError('completed training result did not pass actual final metrics: '+str(run))

    def audit_checkpoint(self, run, seq):
        self.ensure_result(run)
        stage = next(s for s in self.budget['stages'] if s['seq_len'] == seq)
        checkpoint = run/'checkpoint'/('step-'+str(stage['updates']))
        handoff = self.output/'native-checkpoints'/(str(seq)+'.handoff.json')
        audit = handoff.with_name(str(seq)+'.complete-audit.json')
        if not handoff.exists():
            self.command('handoff-'+str(seq), [self.config['training_python'],
                self.root/'midtrain/scripts/create_checkpoint_handoff.py', '--checkpoint', checkpoint, '--output', handoff])
        if not audit.exists():
            binding = self.cohort['approved_budget']
            self.command('audit-native-'+str(seq), [self.config['training_python'],
                self.root/'eval/scripts/audit_completed_native_checkpoint.py', '--handoff', handoff,
                '--budget', binding['file'], '--budget-sha256', binding['sha256'], '--seq-len', seq, '--output', audit])
        binding = {'file': str(audit), 'sha256': digest(audit)}
        actual = complete_checkpoint(binding, {}, seq_len=seq, budget=self.budget)
        if any(digest(p) != sha for p, sha in actual['source_inputs'].items()):
            raise ValueError('completed native audit source changed')
        return binding, actual

    def train_long(self, seq, parent, *, admission=None):
        self.publish('PREPARING_PRODUCTION_'+str(seq), stage_seq_len=seq)
        run = self.output/('training-'+str(seq))
        prepared = self.output/'stage-configs'/('pretrain-final-context-'+str(seq)+'.preparation.json')
        if not prepared.exists():
            release = self.config['window_releases'][str(seq)]
            if digest(release['file']) != release['sha256']:
                raise ValueError('complete admitted production long-window release changed')
            plan = self.cohort['production_stage_plan']
            argv = [self.config['training_python'], self.root/'experiments/e4_curriculum/scripts/prepare_production_context_stage.py',
                '--root', self.root, '--plan', plan['file'], '--plan-sha256', plan['sha256'], '--seq-len', seq,
                '--parent-handoff', parent['checkpoint_handoff_file'], '--window-release', release['file'],
                '--output', prepared.parent, '--run-dir', run]
            if admission is not None:
                argv += ['--evaluation-admission', admission]
            self.command('prepare-stage-'+str(seq), argv)
        preparation = json.loads(prepared.read_bytes())
        if any(digest(preparation[key]) != preparation[key+'_sha256'] for key in ('config', 'env')):
            raise ValueError('prepared production context config/environment changed')
        complete = run/'training_complete.json'
        if not complete.exists() or json.loads(complete.read_bytes()).get('status') != 'PASS':
            self.wait_idle()
            self.publish('TRAINING_PRODUCTION_'+str(seq), stage_seq_len=seq, stage_run=str(run))
            self.command('train-stage-'+str(seq), [self.config['training_python'],
                self.root/'midtrain/scripts/launch.py', '--env', preparation['env']],
                env=self.environment(cuda='0,1,2,3,4,5,6,7'), training=True)
        return self.audit_checkpoint(run, seq)

    def measure(self, seq, role, checkpoint):
        self.publish('MEASURING_'+role.upper()+'_'+str(seq), stage_seq_len=seq)
        stage = next(s for s in self.plan['stages'] if s['seq_len'] == seq)
        model = self.root/stage['qualified_model']['file']
        preset = stage['qualified_model']['preset']
        receipts = {}
        for suite in ('long_bpb', 'short_core', 'ruler', 'long_qa_code'):
            case_file = self.cohort['cases'][suite]['file']
            result_dir = self.output/'measurements'/str(seq)/role/suite
            completed = result_dir/'completed-worker-set.json'
            if completed.exists():
                chosen = json.loads(completed.read_bytes())['receipts']
            else:
                self.wait_idle()
                attempt = result_dir/('attempt-'+secrets.token_hex(8))
                attempt.mkdir(parents=True)
                jobs, handles = [], []
                deadline = time.monotonic()+self.config['evaluation_timeout_seconds']
                try:
                    for rank in range(8):
                        worker_out = attempt/('worker-'+str(rank))
                        handle = (attempt/('worker-'+str(rank)+'.log')).open('x')
                        handles.append(handle)
                        argv = [self.config['training_python'], self.root/'eval/scripts/evaluate_native_checkpoint.py',
                            '--cases', case_file, '--handoff', checkpoint['checkpoint_handoff_file'],
                            '--model', model, '--preset', preset, '--seq-len', str(seq),
                            '--tokenizer', self.config['tokenizer'], '--output', worker_out,
                            '--worker-index', str(rank), '--workers', '8']
                        jobs.append(subprocess.Popen(list(map(str, argv)), cwd=self.root,
                            env=self.environment(cuda=str(rank)), stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True))
                    self.publish(active_evaluation_suite=suite, active_worker_pids=[p.pid for p in jobs],
                                 active_evaluation_attempt=str(attempt))
                    last_publish = time.monotonic()
                    while any(p.poll() is None for p in jobs):
                        self.guard()
                        if any(p.poll() not in (None, 0) for p in jobs):
                            raise ValueError('native evaluation worker failed; preserve '+str(attempt))
                        if time.monotonic() >= deadline:
                            raise ValueError('actual evaluation exceeded its declared wallclock bound')
                        if time.monotonic()-last_publish >= 30:
                            self.publish(active_worker_returncodes=[p.poll() for p in jobs],
                                active_worker_prediction_bytes=[p.stat().st_size if p.exists() else 0
                                    for p in [attempt/('worker-'+str(rank))/'measurements.jsonl' for rank in range(8)]])
                            last_publish = time.monotonic()
                        time.sleep(5)
                    if any(p.returncode != 0 for p in jobs):
                        raise ValueError('native evaluation worker failed; preserve '+str(attempt))
                    chosen = [str(attempt/('worker-'+str(rank))/'receipt.json') for rank in range(8)]
                finally:
                    for process in jobs:
                        if process.poll() is None:
                            self.terminate_owned(process)
                    for handle in handles:
                        handle.close()
            load_measurements(case_file, chosen, seq_len=seq, handoff_file=checkpoint['checkpoint_handoff_file'])
            if not completed.exists():
                seal(completed, {'status': 'PASS_INDEPENDENT_COMPLETE_EIGHT_WORKER_MEASUREMENTS_NOT_CAPABILITY',
                    'receipts': chosen, 'source_inputs': {p: digest(p) for p in chosen}})
            receipts[suite] = chosen
        self.publish(active_evaluation_suite=None, active_worker_pids=[], active_evaluation_attempt=None)
        return receipts

    def admit(self, seq, baseline, candidate, baseline_receipts, candidate_receipts):
        directory = self.output/'capability'/str(seq)
        spec, report = directory/'spec.json', directory/'admission.json'
        if not spec.exists():
            seal(spec, {'seq_len': seq, 'cohort': self.config['cohort'], 'repoqa_source': self.config['repoqa_source'],
                'baseline': {'completed_audit': baseline, 'receipts': baseline_receipts},
                'candidate': {'completed_audit': candidate, 'receipts': candidate_receipts}})
        if not report.exists():
            self.command('admit-'+str(seq), [self.config['scoring_python'], self.root/'eval/scripts/admit_context_stage.py',
                '--spec', spec, '--output', report])
        self.command('recompute-admission-'+str(seq), [self.config['scoring_python'],
            self.root/'eval/scripts/admit_context_stage.py', '--verify', report])
        return report

    def run(self):
        self.publish('WAITING_FOR_ACTUAL_4K_COMPLETION')
        while formal_completion_state(self.config['formal_run']) == 'WAITING_FOR_ACTUAL_4K_COMPLETION':
            self.guard()
            self.publish('WAITING_FOR_ACTUAL_4K_COMPLETION')
            time.sleep(30)
        parent_binding, parent = self.audit_checkpoint(Path(self.config['formal_run']), 4096)
        admission = None
        for seq in (16384, 65536):
            candidate_binding, candidate = self.train_long(seq, parent, admission=admission)
            left = self.measure(seq, 'baseline', parent)
            right = self.measure(seq, 'candidate', candidate)
            admission = self.admit(seq, parent_binding, candidate_binding, left, right)
            self.state['stages'][str(seq)] = {'status': 'PASS_TRAINED_AND_ACTUAL_LONG_SHORT_CAPABILITY',
                'complete_checkpoint_audit': candidate_binding, 'admission': str(admission),
                'admission_sha256': digest(admission)}
            self.publish()
            parent_binding, parent = candidate_binding, candidate
        self.publish('PASS_COMPLETE_4K_WSD_16K_64K_TRAINING_AND_ACTUAL_CAPABILITY_GATES')
