"""Confirm the fixed candidate on identical actual production warmup samples."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys

import yaml

import benchmark_model_sizes as bench

ROOT = bench.ROOT
sys.path[:0] = [str(ROOT/'pretrain/src'), str(ROOT/'model/src'),
                str(ROOT/'experiments/e4_curriculum/scripts'), str(ROOT/'pretrain/scripts')]
from checkpoint_durability_evidence import require_checkpoint_directory_durability
from freeze_1t_recipe import source_sha256
from production_probe_contracts import build_probe
from qualification_identity import qualification_identity as production_identity
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries
from yuformer_pretrain.lease import RunLease

bench.CONFIGS = ROOT/'experiments/e2_architecture/configs/production_matched_controls_20261115'
bench.RESULTS = ROOT/'experiments/e2_architecture/results/production_matched_controls_20261115'
bench.RUNS = ROOT/'data/experiments/production_matched_controls_20261115'
bench.HEALTH = ROOT/'data/health/production-matched-controls.json'
PROTOCOL = bench.RESULTS/'protocol.json'


def parse_env(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#'))


def require_declared_source(protocol):
    if source_sha256(ROOT) != protocol['training_source_sha256']:
        raise ValueError('the exact predeclared checkpoint-hardened source is required')
    for name, expected in protocol['expected_training_source_files'].items():
        if bench.digest(ROOT/name) != expected:
            raise ValueError('predeclared final training source differs: '+name)
    path = ROOT/protocol['production_order_manifest']
    if bench.digest(path) != protocol['production_order_manifest_sha256']:
        raise ValueError('actual production order changed since predeclaration')
    return require_checkpoint_directory_durability(ROOT)


def prepare(protocol):
    base = yaml.safe_load((ROOT/'pretrain/configs/pretrain_base.yaml').read_bytes())
    env = parse_env(ROOT/'pretrain/configs/pretrain_base.env')
    template, probe = build_probe(ROOT, base, env, role='canary', publish=True)
    if (template['train']['expected_iters'] != 200 or probe['tokens_per_update'] != 4194304
            or probe['parent_order_manifest_sha256'] != protocol['production_order_manifest_sha256']
            or base['optim']['lr'] != .0001):
        raise ValueError('the paired controls require the unchanged prepared 1e-4 production recipe')
    variants = []
    for index, row in enumerate(protocol['cases']):
        name = row['case']
        old = ROOT/'data/experiments/rope_source_controls_20261114'/name
        identity = json.loads((old/'run_identity.json').read_bytes())
        config = yaml.safe_load((old/'resolved_config.yaml').read_bytes())
        model_path = Path(identity['model_configs'])
        if bench.digest(model_path) != identity['model_configs_sha256']:
            raise ValueError('the actually measured source architecture changed')
        model = yaml.safe_load(model_path.read_bytes())['presets'][config['model']['preset']]
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        import torch
        with torch.device('meta'):
            actual_model = YuFormerModel(ModelConfig(**model))
        if actual_model.num_parameters() != row['parameters']:
            raise ValueError('actual production-control parameter count differs')
        del actual_model
        mp, cp, ep = [bench.CONFIGS/(name+suffix) for suffix in ('.model.yaml','.yaml','.env')]
        if any(path.exists() for path in (mp,cp,ep)) or (bench.RUNS/name).exists():
            raise ValueError('actual production confirmation must preserve all earlier attempts')
        mp.write_text(yaml.safe_dump({'presets':{'production-control':model}},sort_keys=True))
        private = copy.deepcopy(template)
        private.update(run_name=name, seed=protocol['seed'], out_dir='${YUFORMER_PRETRAIN_OUT}',
                       model=dict(preset='production-control',configs=str(mp)))
        private['optim'].update(lr=row['peak_lr'],warmup_steps=10,min_lr_ratio=.1)
        private['train'].update(log_every=1,eval_every=100,save_every=100,keep_last_n=2)
        private['data']['val_samples_per_domain'] = 64
        if (private['train']['micro_batch_size'] != 8 or private['train']['grad_accum'] != 16
                or private['parallel']['world_size'] != 8 or private['data']['seq_len'] != 4096
                or private['data']['sampling_policy'] != 'block_permutation'):
            raise ValueError('actual production-control geometry/data order changed')
        cp.write_text(yaml.safe_dump(private,sort_keys=False))
        variables = dict(env, YUFORMER_CONFIG=str(cp), YUFORMER_MODEL_CONFIGS=str(mp),
                         YUFORMER_PRETRAIN_OUT=str(bench.RUNS/name), MASTER_PORT=str(29921+index),
                         CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        ep.write_text(''.join(f'{key}={value}\n' for key,value in variables.items()))
        candidate = dict(name=name,parameters=row['parameters'],model_path=str(mp),model_sha256=bench.digest(mp))
        variant = dict(name=name,microbatch=8,grad_accum=16,config=str(cp),env=str(ep),
                       config_sha256=bench.digest(cp),env_sha256=bench.digest(ep))
        variants.append((candidate,variant))
    return variants,probe


def input_identity(probe, producer_inputs):
    rows = {row['path']:row for row in production_identity(ROOT)['files']}
    paths = [PROTOCOL, Path(probe['probe_order_manifest']),
             *bench.CONFIGS.glob('*.yaml'), *bench.CONFIGS.glob('*.env')]
    for path in paths:
        name = str(path.relative_to(ROOT))
        rows[name] = dict(path=name,bytes=path.stat().st_size,sha256=bench.digest(path))
    for name, expected in producer_inputs.items():
        path = Path(name)
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        rows[label] = dict(path=label,bytes=path.stat().st_size,sha256=bench.digest(path))
        if rows[label]['sha256'] != expected:
            raise ValueError('production-control producer changed')
    files = [rows[name] for name in sorted(rows)]
    return dict(schema_version=1,purpose='ACTUAL_PRODUCTION_WARMUP_MATCHED_CONTROL_NOT_FORMAL_TRAINING',
                files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def main():
    protocol = json.loads(PROTOCOL.read_bytes())
    inputs = require_declared_source(protocol)
    pinned = {str(path):bench.digest(path) for path in (Path(__file__), Path(bench.__file__), PROTOCOL,
        ROOT/'pretrain/scripts/production_probe_contracts.py', ROOT/'pretrain/scripts/derive_block_prefix_probe.py',
        ROOT/'experiments/e4_curriculum/scripts/sustained_memory_review.py',
        ROOT/'experiments/e4_curriculum/scripts/allocator_retry_review.py')}
    pinned.update({str(ROOT/name):expected for name,expected in inputs.items()})
    bench.CONFIGS.mkdir(parents=True,exist_ok=True)
    state = dict(pid=os.getpid(),status='PREPARING',cases={},production_qualified=False,formal_pretrain_started=False,
                 purpose=protocol['purpose'],loaded_source_inputs=pinned,protocol_sha256=bench.digest(PROTOCOL))
    with RunLease(bench.RESULTS,'actual-production-matched-current-backend-controls'):
        try:
            bench.require_idle()
            variants,probe = prepare(protocol)
            bench.qualification_identity = lambda root:input_identity(probe,pinned)
            state.update(status='RUNNING',source_qualification_identity=bench.qualification_identity(ROOT),probe=probe)
            bench.publish(bench.RESULTS/'plan.json',dict(protocol_sha256=bench.digest(PROTOCOL),controls=variants,
                source_inputs=pinned,source_qualification_identity=state['source_qualification_identity'],probe=probe))
            evaluate = bench.evaluate
            bench.evaluate = lambda result:evaluate(result,steps=200,skip_steps=50)
            for candidate,variant in variants:
                require_declared_source(protocol)
                result = bench.run_variant(candidate,variant,state)
                raw = Path(result['log']).read_bytes()
                review_memory_log(raw,steps=200,skip_steps=50)
                require_no_allocator_retries(raw)
                state['cases'][variant['name']] = result
                if result['evaluation']['status'] != 'PASS_SCREENING':
                    raise ValueError('actual production confirmation system gate failed')
                if candidate['name'] == protocol['candidate'] and not result['evaluation']['within_33_days']:
                    raise ValueError('actual production-window candidate exceeds 33 days')
                bench.publish(bench.HEALTH,state)
            require_declared_source(protocol)
            state['status'] = 'COMPLETE_REQUIRES_PAIRED_PRIMARY_AND_EXPANDED_QUALITY_REVIEW'
        except Exception as error:
            state.update(status='FAIL_CLOSED_PRODUCTION_MATCHED_CONTROLS',error=repr(error))
            raise
        finally:
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH,state)


if __name__ == '__main__':
    main()
