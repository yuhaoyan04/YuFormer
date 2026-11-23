"""Archive new matched research runs alongside the existing adoption evidence."""
import hashlib
from pathlib import Path

from frozen_training_evidence import freeze_report_bindings, snapshot_files


def freeze_adoption_research_evidence(root, inputs):
    root = Path(root).resolve()
    bound, archive = freeze_report_bindings(root, inputs)
    research = {name: expected for name, expected in bound.items()
                if name.startswith(('data/experiments/rope_source_controls_20261114/',
                                    'experiments/e2_architecture/configs/rope_source_controls_20261114/',
                                    'experiments/e2_architecture/scripts/',
                                    'data/experiments/production_matched_controls_20261115/',
                                    'experiments/e2_architecture/configs/production_matched_controls_20261115/',
                                    'data/experiments/final_deterministic_production_controls_20261115/',
                                    'experiments/e2_architecture/configs/final_deterministic_production_controls_20261115/',
                                    'data/experiments/deterministic_flash_long_profiles_20261115/',
                                    'report/patches/deterministic-flash-long-profiles/',
                                    'data/experiments/production_shape_lr_controls_20261115',
                                    'experiments/e2_architecture/configs/production_shape_lr_controls_20261115',
                                    'data/health/nearby-lr-', 'data/health/lower-lr-'))
                or name in {'data/health/model-rope-source-controls.json',
                            'data/health/production-matched-controls.json',
                            'data/health/production-matched-development-review.json',
                            'data/health/final-deterministic-production-controls.json',
                            'data/health/final-deterministic-production-review.json',
                            'data/health/production-shape-lr-controls.json',
                            'data/health/production-shape-lr-contract-preview.json',
                            'data/health/checkpoint-directory-durability-qualification.json'}}
    source_paths = set()
    for base in (root/'model/src', root/'pretrain/backends/torchtitan'):
        if base.is_dir():
            source_paths.update(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('pretrain/scripts/run_with_lease.py', 'pretrain/scripts/assert_run_lease.py',
                 'pretrain/src/yuformer_pretrain/lease.py', 'pretrain/configs/pretrain_base.yaml',
                 'pretrain/configs/pretrain_base.env', 'model/configs/pretrain_final_model_configs.yaml'):
        path = root/name
        if path.is_file():
            source_paths.add(path)
    source_backups = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for name, expected in source_backups.items():
        if name in bound and bound[name] != expected:
            raise ValueError('training source backup differs from its admitted binding: '+name)
    if not research and not source_backups:
        return bound, archive
    originals = {name: row['sha256'] for name, row in archive['files'].items()}
    complete = snapshot_files(root, {**originals, **research, **source_backups})
    bound = {name: expected for name, expected in bound.items() if name not in research}
    # Preserve live-code guards and immutable recovery copies together.
    bound.update(source_backups)
    bound.update(complete['dependencies'])
    return bound, complete
