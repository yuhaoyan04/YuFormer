"""Archive new matched research runs alongside the existing adoption evidence."""
from frozen_training_evidence import freeze_report_bindings, snapshot_files


def freeze_adoption_research_evidence(root, inputs):
    bound, archive = freeze_report_bindings(root, inputs)
    research = {name: expected for name, expected in bound.items()
                if name.startswith(('data/experiments/rope_source_controls_20261114/',
                                    'experiments/e2_architecture/configs/rope_source_controls_20261114/'))
                or name == 'data/health/model-rope-source-controls.json'}
    if not research:
        return bound, archive
    originals = {name: row['sha256'] for name, row in archive['files'].items()}
    complete = snapshot_files(root, {**originals, **research})
    bound = {name: expected for name, expected in bound.items() if name not in research}
    bound.update(complete['dependencies'])
    return bound, complete
