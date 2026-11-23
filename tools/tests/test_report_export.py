import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('report_export', Path(__file__).resolve().parents[1]/'export_pretrain_report.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
DOMAINS = ['academic', 'books', 'chinese', 'code', 'encyclopedia', 'math', 'multilingual', 'qa', 'web']


def receipt():
    return {'step': 2000, 'status': 'PASS_ACTUAL_NINE_DOMAIN_VALIDATION_AND_IMMUTABLE_EVENT_PREFIX',
        'metrics': {key: {'step': 2000, 'value': 2.0} for key in
        [*(f'validation_metrics/domain/{domain}/loss' for domain in DOMAINS), 'validation_metrics/loss']}}


class ExportTests(unittest.TestCase):
    def test_missing_domain_rejected(self):
        value = receipt()
        del value['metrics']['validation_metrics/domain/books/loss']
        with self.assertRaises(ValueError): EXPORT.loss_values(value)

    def test_nonfinite_and_inconsistent_aggregate_rejected(self):
        for value in [float('nan'), float('inf'), -1, 4]:
            data = receipt()
            data['metrics']['validation_metrics/loss']['value'] = value
            with self.assertRaises(ValueError): EXPORT.loss_values(data)

    def test_wrong_metric_step_rejected(self):
        value = receipt()
        value['metrics']['validation_metrics/domain/books/loss']['step'] = 4000
        with self.assertRaises(ValueError): EXPORT.loss_values(value)

    def test_different_attempt_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            target = root/'data/runs/pretrain-final-1t-8gpu/current_attempt.json'
            target.write_text(json.dumps({'attempt_id': 'wrong', 'identity_sha256': 'identity'}))
            with self.assertRaises(ValueError): EXPORT.export(root)
            self.assertEqual((root/'README.md').read_text(), 'existing')
            self.assertFalse((root/'report/metrics/latest_pretrain_status.json').exists())

    def test_attention_remains_visible_and_frozen_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            frozen = root/'experiments/e4_curriculum/results/effective_total_training_budget.json'
            before = frozen.read_bytes()
            result = EXPORT.export(root)
            self.assertEqual(result['validation_rows'], 1)
            snapshot = json.loads((root/'report/metrics/latest_pretrain_status.json').read_text())
            self.assertEqual(snapshot['status'], 'HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION')
            self.assertEqual(snapshot['regressed_domains'], {'books': .12})
            self.assertFalse(snapshot['final_capability_completed'])
            self.assertFalse(snapshot['sft_rl_training_completed'])
            self.assertEqual(frozen.read_bytes(), before)
            source = json.loads((root/'report/metrics/provenance.json').read_text())
            self.assertIn('pretrain_validation.csv', source['exports'])

    @staticmethod
    def fixture(root):
        def write(name, value):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        review = {'status': 'HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION',
            'scope': 'INTERMEDIATE_ONLY', 'reviewed_at': '2026-11-23T00:46:00+00:00',
            'next_review_at': '2026-11-23T12:46:00+00:00', 'attempt_id': 'attempt',
            'training_metrics': {'step': 2000, 'loss': 2, 'tps_per_device': 48000, 'memory_gib': 67.46},
            'labels_at_last_logged_step': 2000*4194304, 'phase': 'stable',
            'data_cursor_phase': {}, 'regressed_domains': {'books': .12},
            'projected_total_days': 32, 'whole_chain_remaining_days': 24,
            'projected_completion_utc': '2026-12-17T00:00:00+00:00',
            'validation_losses': {'2000': {}}, 'checkpoint_receipts_rechecked': [250],
            'lr_contracts': [{'observations_checked': 200}]}
        write('data/health/assistant-half-day-pretrain-review-20261123T0046.json', review)
        write('experiments/e4_curriculum/results/effective_total_training_budget.json',
              {'actual_total_tokens': 966254395392, 'stages': [{'name': '4k-pretrain', 'tokens': 916258291712, 'updates': 218453}]})
        write('data/runs/pretrain-final-1t-8gpu/run_identity.json', {'identity_sha256': 'identity'})
        write('data/runs/pretrain-final-1t-8gpu/current_attempt.json', {'attempt_id': 'attempt', 'identity_sha256': 'identity'})
        write('data/health/all-production-nine-domain-validation-receipts-20261115/4096-attempt-step-2000.json', receipt())
        write('data/health/model-size-sweep.json', {'cases': {'sample': {'parameters': 1516600064, 'evaluation': {}}}})
        write('data/health/production-throughput-canary.json', {'evaluation': {'p10_tps_per_device': 48648.0625, 'projected_total_training_days': 32.745}})
        (root/'README.md').write_text('existing')
        return root


if __name__ == '__main__':
    unittest.main()
