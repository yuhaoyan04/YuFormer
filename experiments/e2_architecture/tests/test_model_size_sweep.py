import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_model_sizes.py'
spec = importlib.util.spec_from_file_location('size_sweep', path)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def result():
    return {'status': 'PASS', 'issues': [], 'hyperparameters': {
        'world_size': 8, 'tokens_per_step': 4194304, 'trained_steps': 24,
        'trained_tokens': 24 * 4194304}, 'series': {
        'throughput(tps)': [{'step': i, 'value': 1 if i <= 8 else 75000} for i in range(1, 25)],
        'mfu(%)': [{'step': i, 'value': 30} for i in range(1, 25)]}}


class SweepEvaluationTests(unittest.TestCase):
    def test_sustained_horizon_counts_all_150_steady_updates(self):
        r = result()
        r['hyperparameters'].update(trained_steps=200, trained_tokens=200 * 4194304)
        r['series']['throughput(tps)'] = [
            {'step': i, 'value': 1 if i <= 50 else 75000} for i in range(1, 201)]
        actual = sweep.evaluate(r, steps=200, skip_steps=50)
        self.assertEqual(actual['status'], 'PASS_SCREENING')
        self.assertEqual(actual['steady_points'], 150)
        self.assertEqual(actual['p10_tps_per_device'], 75000)

    def test_skip_compile_and_use_eight_gpu_duty_adjusted_forecast(self):
        actual = sweep.evaluate(result())
        self.assertEqual(actual['status'], 'PASS_SCREENING')
        self.assertEqual(actual['p10_tps_per_device'], 75000)
        self.assertAlmostEqual(actual['projected_1t_days_duty_09'], 1e12 / (8 * 75000 * .9 * 86400))
        self.assertFalse(actual['production_qualified'])

    def test_slow_model_is_measured_but_outside_time_budget(self):
        r = result()
        for point in r['series']['throughput(tps)']:
            point['value'] = 30000
        actual = sweep.evaluate(r)
        self.assertEqual(actual['status'], 'PASS_SCREENING')
        self.assertFalse(actual['within_30_days'])
        self.assertFalse(actual['within_33_days'])

    def test_missing_or_duplicate_step_cannot_fake_sufficient_samples(self):
        r = result()
        r['series']['throughput(tps)'][-1]['step'] = 23
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')

    def test_nonfinite_throughput_rejected(self):
        r = result()
        r['series']['throughput(tps)'][-1]['value'] = float('nan')
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')

    def test_different_global_batch_cannot_enter_comparison(self):
        r = result()
        r['hyperparameters']['tokens_per_step'] //= 2
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')

    def test_different_world_cannot_enter_comparison(self):
        r = result()
        r['hyperparameters']['world_size'] = 4
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')

    def test_incomplete_training_cannot_enter_comparison(self):
        r = result()
        r['hyperparameters']['trained_tokens'] -= 4194304
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')

    def test_failed_collector_cannot_enter_comparison(self):
        r = result()
        r['status'] = 'FAIL'
        self.assertEqual(sweep.evaluate(r)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
