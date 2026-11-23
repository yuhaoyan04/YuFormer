import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('cleanup', Path(__file__).resolve().parents[1]/'clean_stale_artifacts.py')
CLEAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEAN)


class CleanupTests(unittest.TestCase):
    def test_protected_ancestor_and_descendant_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = {root/'data', root/'frozen/receipt.json'}
            for candidate in [root, root/'data', root/'data/cache', root/'frozen', root/'frozen/receipt.json']:
                with self.assertRaises(ValueError): CLEAN.assert_safe(candidate, root, protected)
            CLEAN.assert_safe(root/'obsolete', root, protected)

    def test_symlink_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'link').symlink_to('/tmp', target_is_directory=True)
            for candidate in [root/'link', root/'../outside', root/'link/something']:
                with self.assertRaises(ValueError): CLEAN.assert_safe(candidate, root, set())

    def test_generic_current_checkpoints_never_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ['data/runs/current/checkpoint', 'report/other/checkpoint', 'model/__pycache__', '.ipynb_checkpoints']:
                (root/path).mkdir(parents=True)
            actual = {p.relative_to(root).as_posix() for p in CLEAN.candidates(root)}
            self.assertEqual(actual, {'model/__pycache__', '.ipynb_checkpoints'})


if __name__ == '__main__': unittest.main()
