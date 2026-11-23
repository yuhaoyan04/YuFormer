"""Immutable report snapshots so later report exports cannot invalidate a run."""
import hashlib
import json
import os
from pathlib import Path
import sys

from training_data_origin import read_bound


def publish_checked(path, raw, *, immutable=False):
    """Publish proof bytes only after checked file and directory writeback."""
    path = Path(path)
    if immutable and path.exists():
        existing, _ = read_bound(path)
        if existing != raw:
            raise ValueError('refusing to overwrite different immutable training evidence')
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
        with temporary.open('wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def production_output_dir(root):
    root = Path(root).resolve()
    environment = dict(line.split('=', 1) for line in
        (root/'pretrain/configs/pretrain_base.env').read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#'))
    value = environment.get('YUFORMER_PRETRAIN_OUT')
    if not isinstance(value, str) or not value:
        raise ValueError('production output directory is not explicitly declared')
    for _ in range(len(environment)+1):
        previous = value
        for key, replacement in environment.items():
            value = value.replace('${'+key+'}', replacement)
        if value == previous:
            break
    if '${' in value:
        raise ValueError('unresolved production output directory')
    path = Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def production_mutation_lease(root, operation):
    sys.path.insert(0, str(Path(root)/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    return RunLease(production_output_dir(root), operation)


def snapshot_files(root, bindings):
    root = Path(root).resolve()
    folder = root/'data/processed/training_evidence'
    objects = folder/'objects'
    manifests = folder/'manifests'
    objects.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(exist_ok=True)
    records, dependencies = {}, {}
    for name, expected in sorted(bindings.items()):
        path = Path(name)
        path = path if path.is_absolute() else root/path
        if not path.resolve().is_relative_to(root):
            raise ValueError('training report snapshots must be repository-relative')
        label = str(path.relative_to(root))
        raw, digest = read_bound(path, expected)
        if len(raw) > 128*1024*1024:
            raise ValueError('training evidence snapshots contain reports/configuration, not model/token binaries')
        target = objects/digest
        if target.exists():
            read_bound(target, digest)
        publish_checked(target, raw, immutable=True)
        archived = str(target.relative_to(root))
        dependencies[archived] = digest
        records[label] = {'archive': archived, 'sha256': digest, 'bytes': len(raw)}
    manifest = {'schema_version': 1, 'purpose': 'IMMUTABLE_TRAINING_QUALIFICATION_EVIDENCE',
                'files': records}
    raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = manifests/(digest+'.json')
    if path.exists():
        read_bound(path, digest)
    publish_checked(path, raw, immutable=True)
    relative = str(path.relative_to(root))
    dependencies[relative] = digest
    return {'manifest': relative, 'manifest_sha256': digest, 'files': records,
            'dependencies': dependencies}


def freeze_report_bindings(root, inputs):
    def mutable(name):
        return (name.startswith(('data/experiments/flex_sustained_20261114/',
                                 'data/experiments/window_source_controls_20261114/',
                                 'experiments/e2_architecture/configs/flex_sustained_20261114/',
                                 'experiments/e2_architecture/configs/window_source_controls_20261114/',
                                 'experiments/e2_architecture/results/'))
                or name in {'data/health/torchtitan-qualification.json',
                    'data/health/production-recovery-qualification.json',
                    'data/health/model-flex-sustained-controls.json',
                    'data/health/model-window-source-controls.json',
                    'data/health/production-stage-order-recount.json',
                    'data/health/production-stage-order-admission.json',
                    'data/health/native-merge-writeback-qualification.json',
                    'data/health/native-merge-writeback-qualification.xml',
                    'data/health/warmup-training-block-permutation.json',
                    'data/health/stable-training-block-permutation.json',
                    'data/health/anneal-training-block-permutation.json',
                    'data/health/flex-module-parity.json',
                    'data/processed/audits/training_tokenizer_calibration.json',
                    'report/experiment_identity_audit_latest.json'})
    selected = {name: value for name, value in inputs.items() if mutable(name)}
    archive = snapshot_files(root, selected)
    bound = {name: value for name, value in inputs.items() if name not in selected}
    bound.update(archive['dependencies'])
    return bound, archive
