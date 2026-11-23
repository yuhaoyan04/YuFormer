#!/usr/bin/env python3
"""Qualify canonical native writeback behavior after all encoders finish."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from frozen_training_evidence import publish_checked
from training_data_origin import read_bound

ROOT = Path(__file__).resolve().parents[3]
REPORT = 'data/health/native-merge-writeback-qualification.json'
XML = 'data/health/native-merge-writeback-qualification.xml'
SOURCE = 'pretrain/scripts/merge_token_parts.c'
TESTS = ('pretrain/tests/test_native_merge_durability.py',
         'pretrain/tests/test_materialize_memory_bounds.py',
         'experiments/e4_curriculum/tests/test_token_publication_durability.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_test_results(raw):
    tree = ET.fromstring(raw)
    suites = [tree] if tree.tag == 'testsuite' else list(tree.iter('testsuite'))
    cases = list(tree.iter('testcase'))
    if (sum(int(s.get('tests', 0)) for s in suites) != 15 or len(cases) != 15
            or any(int(s.get(key, 0)) for s in suites for key in ('errors','failures','skipped'))
            or any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped'))):
        raise ValueError('native durability qualification requires fifteen actual unskipped passing cases')
    required = {'test_checked_native_merge_preserves_exact_document_order_and_ledger',
                'test_native_merge_grows_part_table_under_low_fd_limit'} | {
        f'test_writeback_or_close_error_prevents_success_receipt[{failure}]' for failure in
        ('fsync_tokens','fsync_lengths','close_tokens','close_lengths')}
    if not required <= {c.get('name') for c in cases}:
        raise ValueError('canonical native qualification lacks actual writeback/close fault or low-FD coverage')
    return sorted(c.get('name') for c in cases)


def require_native_qualification(root):
    root = Path(root)
    raw, sha = read_bound(root/REPORT)
    report = json.loads(raw)
    if (report.get('status') != 'PASS_CANONICAL_NATIVE_WRITEBACK'
            or report.get('override_source_used') is not False
            or report.get('qualification_tool_sha256') != digest(__file__)
            or set(report.get('source_inputs', {})) != {SOURCE, *TESTS}):
        raise ValueError('canonical native durability proof is absent or stale')
    inputs = dict(report['source_inputs'])
    for name, expected in inputs.items():
        read_bound(root/name, expected)
    xml, xml_sha = read_bound(root/XML, report.get('pytest_xml_sha256'))
    if require_test_results(xml) != report.get('passed_cases'):
        raise ValueError('native qualification differs from its actual pytest result')
    inputs.update({REPORT:sha, XML:xml_sha, str(Path(__file__).relative_to(root)):digest(__file__)})
    return inputs, {'report':REPORT, 'sha256':sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    tokenization = json.loads((root/'data/health/pretrain-1t-tokenization.json').read_bytes())
    if (tokenization.get('status') != 'PASS' or len(tokenization.get('jobs', {})) != 12
            or any(j.get('status') != 'PASS' for j in tokenization['jobs'].values())):
        raise ValueError('all twelve original stream jobs must finish before canonical native qualification')
    inputs = {name:digest(root/name) for name in (SOURCE,*TESTS)}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    environment.pop('YUFORMER_TEST_NATIVE_MERGE_SOURCE', None)
    command = ['/opt/yuformer-cpu-tests/bin/python','-m','pytest','-q',
               TESTS[0], TESTS[1]+'::test_native_merge_grows_part_table_under_low_fd_limit',
               TESTS[2], '--junitxml='+str(root/XML)]
    log = root/'report/logs/native-merge-writeback-qualification.log'
    report = dict(schema_version=1, status='RUNNING', source_inputs=inputs,
                  override_source_used=False, qualification_tool_sha256=digest(__file__), command=command)
    publish_checked(root/REPORT, (json.dumps(report, sort_keys=True, indent=2)+'\n').encode())
    try:
        with log.open('a') as output:
            completed = subprocess.run(command, cwd=root, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=300)
        if completed.returncode or any(digest(root/name) != sha for name,sha in inputs.items()):
            raise ValueError('actual canonical native fault/durability tests failed or their source changed')
        xml, xml_sha = read_bound(root/XML)
        publish_checked(root/XML, xml)
        report.update(status='PASS_CANONICAL_NATIVE_WRITEBACK', pytest_xml_sha256=xml_sha,
                      passed_cases=require_test_results(xml))
    except Exception as error:
        report.update(status='FAIL_CANONICAL_NATIVE_WRITEBACK', error=repr(error))
        raise
    finally:
        publish_checked(root/REPORT, (json.dumps(report, sort_keys=True, indent=2)+'\n').encode())
    require_native_qualification(root)
    print(json.dumps({'status':report['status'], 'actual_passing_cases':len(report['passed_cases'])}), flush=True)


if __name__ == '__main__':
    main()
