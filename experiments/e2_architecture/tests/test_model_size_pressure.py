import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_model_size_sweep.py'
spec = importlib.util.spec_from_file_location('size_analysis', path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PressureTests(unittest.TestCase):
    def inspect(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log'
            p.write_text(text)
            return analysis.memory_pressure(p)

    def test_compile_warning_does_not_poison_steady_state(self):
        r = self.inspect('memory allocation failed with OOM\nstep: 8 loss: 2\nstep: 24 loss: 1\n')
        self.assertEqual(r['allocation_retry_warnings_total'], 1)
        self.assertEqual(r['allocation_retry_warnings_steady'], 0)
        self.assertFalse(r['requires_smaller_batch_comparison'])

    def test_colored_steady_warning_requires_batch_comparison(self):
        r = self.inspect('\x1b[31mstep: 8  \x1b[32mloss: 2\nmemory allocation failed with OOM\nstep: 9 loss: 1\n')
        self.assertEqual(r['allocation_retry_warnings_steady'], 1)
        self.assertTrue(r['requires_smaller_batch_comparison'])

    def test_final_checkpoint_warning_not_in_training_window(self):
        r = self.inspect('step: 24 loss: 1\nmemory allocation failed with OOM\n')
        self.assertEqual(r['allocation_retry_warnings_steady'], 0)

    def test_frontier_can_rank_absolute_budget_without_repeating_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            health = root / 'data/health'
            health.mkdir(parents=True)
            log = root / 'case.log'
            log.write_text('step: 24 loss: 1\n')
            case = {'candidate': 'frontier', 'parameters': 1230000000,
                    'variant': {'name': 'frontier_b8', 'microbatch': 8, 'grad_accum': 16},
                    'log': str(log), 'evaluation': {'status': 'PASS_SCREENING',
                        'p10_tps_per_device': 55000, 'projected_1t_days_duty_09': 29.3}}
            (health / 'model-width-frontier.json').write_text(json.dumps({
                'status': 'COMPLETE_SCREENING', 'cases': {'frontier_b8': case}}))
            r = analysis.analyze(root)
            self.assertEqual(r['largest_within_30_days']['candidate'], 'frontier')
            self.assertIsNone(r['largest_under_10pct_slowdown'])

    def test_pressure_candidate_is_excluded_even_when_projection_fits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            health = root / 'data/health'
            health.mkdir(parents=True)
            log = root / 'case.log'
            log.write_text('step: 8 loss: 2\nmemory allocation failed with OOM\nstep: 24 loss: 1\n')
            case = {'candidate': 'frontier', 'parameters': 1230000000,
                    'variant': {'name': 'frontier_b8', 'microbatch': 8, 'grad_accum': 16},
                    'log': str(log), 'evaluation': {'status': 'PASS_SCREENING',
                        'p10_tps_per_device': 55000, 'projected_1t_days_duty_09': 29.3}}
            (health / 'model-width-frontier.json').write_text(json.dumps({
                'status': 'COMPLETE_SCREENING', 'cases': {'frontier_b8': case}}))
            self.assertIsNone(analysis.analyze(root)['largest_within_30_days'])


if __name__ == '__main__':
    unittest.main()
