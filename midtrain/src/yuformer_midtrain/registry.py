"""Use the explicitly declared shared TorchTitan model runtime for long stages."""
import json
import os
from pathlib import Path

from .stage_contract import validate_stage_geometry
from .window_release import bound_json, load_window_release
from .checkpoint_contract import validate_parent, verify_parent_handoff
from .evaluation_contract import require_parent_evaluation


def validate_midtrain_contract(config):
    stage = config['midtrain']
    budget = bound_json(stage['approved_budget_file'], stage['approved_budget_sha256'])
    geometry = validate_stage_geometry(config, budget)
    parent = stage['initial_checkpoint']
    checkpoint = Path(parent['path'])
    import hashlib
    if (not checkpoint.is_absolute() or not (checkpoint/'.metadata').is_file()
            or hashlib.sha256((checkpoint/'.metadata').read_bytes()).hexdigest() != parent.get('metadata_sha256')):
        raise ValueError('initial midtrain checkpoint is absent or its frozen metadata changed')
    gate = json.loads(Path(config['data']['tokenizer_gate']).read_bytes())
    if gate.get('status') != 'PASS':
        raise ValueError('midtrain tokenizer gate must PASS')
    physical_release = json.loads(Path(config['data']['release_manifest']).read_bytes())
    if (physical_release.get('status') != 'PASS'
            or physical_release.get('dataset_id') != gate.get('dataset_id')
            or physical_release.get('snapshot_id') != gate.get('snapshot_id')):
        raise ValueError('shared nine-domain validation must retain its actual physical data release')
    # Validate before building a model or creating a CUDA context. Each worker
    # repeats these immutable byte/order checks before consuming its first row.
    records = load_window_release(stage['window_release'], expected_sha256=stage['window_release_sha256'],
        source_root=stage['source_root'], seq_len=geometry['seq_len'], tokenizer_sha256=gate['tokenizer_sha256'])
    release = bound_json(stage['window_release'], stage['window_release_sha256'])
    if release.get('scope') != geometry['scope']:
        raise ValueError('long-window release scope must match the declared training scope')
    if len(records.index) < geometry['samples']:
        raise ValueError('qualified long-window release cannot cover the complete stage budget')
    if geometry['scope'] == 'PRODUCTION' and len(records.index) != geometry['samples']:
        raise ValueError('production stage must consume exactly the admitted long-window record roster')
    allocator = stage.get('cuda_allocator_conf')
    if allocator is not None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != allocator:
        raise ValueError('midtrain allocator environment differs from the qualified stage policy')
    model_config = json.loads(json.dumps(config['model']))
    import yaml
    model = yaml.safe_load(Path(model_config['configs']).read_bytes())['presets'][model_config['preset']]
    if model.get('max_seq_len', 0) < geometry['seq_len'] or model.get('selective_recompute', 'none') != stage['selective_recompute']:
        raise ValueError('midtrain model context/recomputation differs from the declared stage')
    validate_parent(parent, budget=budget, next_seq=geometry['seq_len'], scope=geometry['scope'],
                    model=model, tokenizer_sha256=gate['tokenizer_sha256'])
    verify_parent_handoff(parent)
    require_parent_evaluation(parent,next_seq=geometry['seq_len'],scope=geometry['scope'])
    return geometry, gate, records


def midtrain_from_yaml():
    # The launcher declares this dependency through YUFORMER_SHARED_BACKEND_ROOT
    # and PYTHONPATH. The pretrain source tree remains unchanged during training.
    from yuformer_ext.config_registry import _load_contract, yuformer_from_yaml
    from .data import WindowDataLoader
    config = _load_contract()
    geometry, gate, _ = validate_midtrain_contract(config)
    stage = config['midtrain']
    checkpoint = Path(stage['initial_checkpoint']['path'])
    job = yuformer_from_yaml()
    job.dataloader = WindowDataLoader.Config(release_manifest=stage['window_release'],
        release_manifest_sha256=stage['window_release_sha256'], source_root=stage['source_root'],
        tokenizer_sha256=gate['tokenizer_sha256'], samples=geometry['samples'])
    job.checkpoint.initial_load_path = str(checkpoint)
    job.checkpoint.initial_load_model_only = True
    job.checkpoint.initial_load_in_hf = False
    # Upstream DCP ignores these initial_* flags when a valid local step>0
    # exists, restoring the full optimizer/scheduler/cursor/RNG on local resume.
    from .runtime import MidtrainTrainer
    from dataclasses import fields
    if geometry['seq_len']==65536:
        from .checkpoint import RankBoundCheckpointManager
        job.checkpoint=RankBoundCheckpointManager.Config(**{
            field.name:getattr(job.checkpoint,field.name) for field in fields(job.checkpoint)})
    return MidtrainTrainer.Config(**{field.name:getattr(job,field.name) for field in fields(job)})
