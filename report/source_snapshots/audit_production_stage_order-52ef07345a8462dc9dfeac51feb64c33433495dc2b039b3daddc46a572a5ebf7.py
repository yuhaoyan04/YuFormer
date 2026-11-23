#!/usr/bin/env python3
"""Recount all actual phase tokens in the proposed production block order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'experiments/e4_curriculum/scripts'))
    import audit_training_block_permutation as audit
    from publish_training_block_manifest import shift_prepared_permutation
    audit.ROOT = root
    audit.WORK = root/'data/processed/training_permutations/e134-11326994-v1'
    directory = root/'data/processed/training_permutations/pretrain-1t-dd284efd7c715544-b8'
    plan = json.loads((directory/'mapping_plan.json').read_bytes())
    physical = json.loads((root/'data/processed/audits/pretrain_1t_token_streams.json').read_bytes())
    if (physical.get('status') != 'PASS' or physical.get('train_doc_lengths_audited') is not True
            or physical.get('train_source_rows_verified') is not True):
        raise ValueError('current full physical audit must pass before temporal proof reuse')
    physical_rows = {row['name']:row for row in physical['streams']}
    def digest(path):
        value=hashlib.sha256()
        with Path(path).open('rb') as handle:
            while chunk:=handle.read(8<<20):value.update(chunk)
        return value.hexdigest()
    def recount(phase):
        path = root/f'data/health/{phase}-training-block-permutation.json'
        cached = json.loads(path.read_bytes()) if path.is_file() else None
        meta = json.loads((root/f'data/token_streams/pretrain-1t/{phase}.meta.json').read_bytes())
        proof = physical_rows[phase]
        if (proof.get('status') != 'PASS' or proof.get('actual_order_sha256') != meta['order_sha256']
                or proof.get('actual_doc_lengths_sha256') != meta['doc_lengths']['sha256']):
            raise ValueError('full physical audit does not close temporal source/ledger identity: '+phase)
        current = (cached and cached.get('audit_tool_sha256') == digest(audit.__file__)
                   and cached.get('permutation_tool_sha256') == digest(audit.blocks.__file__)
                   and cached.get('source_order_sha256') == meta['order_sha256']
                   and cached.get('doc_lengths_sha256') == meta['doc_lengths']['sha256']
                   and cached.get('evidence_sha256')
                   and all(digest(root/name)==sha for name,sha in cached['evidence_sha256'].items()))
        if current:
            for file_key, hash_key in (('permutation_file','permutation_sha256'),
                                       ('block_domain_tokens_file','block_domain_tokens_sha256')):
                if digest(root/cached[file_key]) != cached[hash_key]:
                    raise ValueError('current temporal proof artifact is corrupt: '+phase)
            return cached, 'REUSED_IDENTICAL_SOURCE_INPUTS_AND_FULLY_HASHED_ARTIFACTS'
        if cached:
            from frozen_training_evidence import publish_checked
            raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
            publish_checked(root/f'data/health/_history/{phase}-training-block-permutation-{sha}.json',raw,immutable=True)
        return audit.audit(phase, plan['geometry']['block_samples']), 'FRESH_ACTUAL_RECOUNT'
    rows = []
    for expected in plan['stages']:
        phase = expected['phase']
        result, recount_origin = recount(phase)
        mixture = json.loads((root/f'data/health/pretrain-1t-{phase}-mixture-preflight.json').read_bytes())
        counts = {name: value['tokens'] for name, value in mixture['mixture']['domains'].items()}
        expected_array = shift_prepared_permutation(directory, expected,
            source_sample_offset=result['source_sample_offset'], block_samples=plan['geometry']['block_samples'])
        meta_path = root/f'data/token_streams/pretrain-1t/{phase}.meta.json'
        meta = json.loads(meta_path.read_bytes())
        if (result['status'] != 'PASS_DIAGNOSTIC'
                or result['whole_domain_token_counts'] != counts
                or result['permutation_sha256'] != expected_array['sha256']
                or result['base_permutation_sha256'] != expected['permutation']['sha256']
                or result['source_sample_offset'] != mixture['source_window']['source_sample_offset']
                or result['temporal_mixing']['pass'] is not True
                or result['doc_lengths_sha256'] != meta['doc_lengths']['sha256']
                or result['stage_sample_limit'] != expected['stage_sample_limit']
                or result['trained_target_tokens'] != mixture['configured_target_tokens']
                or not result['whole_domain_token_counts_unchanged']
                or not result['every_causal_block_exactly_once']):
            raise ValueError('temporal recount disagrees with actual production consumed tokens/order: '+phase)
        path = root/f'data/health/{phase}-training-block-permutation.json'
        rows.append({'phase': phase, 'report': str(path.relative_to(root)),
            'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'recount_origin': recount_origin,
            'source_sample_offset': result['source_sample_offset'],
            'temporal_mixing_pass': result['temporal_mixing']['pass'],
            'original_max_window_absolute_error': result['original_max_window_absolute_error'],
            'permuted_max_window_absolute_error': result['permuted_max_window_absolute_error'],
            'whole_consumed_counts_match_admitted_window': True,
            'permutation_matches_production_window_array': True})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    value = {'status': 'PASS_EXACT_COVERAGE_AWAITING_TEMPORAL_REVIEW',
        'production_qualified': False, 'phases': rows,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_tool_sha256': hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
        'note': 'All original domain tolerances remain unchanged; temporal balance is separately reviewed.'}
    path = root/'data/health/production-stage-order-recount.json'
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    os.replace(temporary, path)


if __name__ == '__main__':
    main()
