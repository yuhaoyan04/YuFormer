import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('repository_maintenance', Path(__file__).resolve().parents[1]/'maintain_pretrain_repository.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicationGuards(unittest.TestCase):
    def test_attention_cannot_publish_without_current_analysis(self):
        review = {'status': MODULE.ATTENTION, 'reviewed_at': '2026-11-26T12:46:00', 'regressed_domains': {'code': .3}}
        for note in [None, 'stale analysis2026-11-25']:
            with self.assertRaises(ValueError): MODULE.validate_observation(review, note)
        MODULE.validate_observation(review, 'independent analysis2026-11-26T12:46:00')

    def test_unrecognized_failure_is_not_an_observation(self):
        with self.assertRaises(ValueError):
            MODULE.validate_observation({'status': 'FAIL', 'reviewed_at': 'now'}, 'analysisnow')

    def test_unexpected_source_frozen_path_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                MODULE.validate_paths({'pretrain/trainer.py'}, set(), root, {})
            with self.assertRaises(ValueError):
                MODULE.validate_paths({'pretrain/trainer.py'}, {'pretrain/trainer.py'}, root,
                                      {str(root/'pretrain/trainer.py'): {}})
            (root/'report').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.validate_paths({'report/outside.md'}, set(), root, {})
