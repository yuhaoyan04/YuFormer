from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from qualify_native_merge import require_test_results


def passing_receipt():
    names = ['test_checked_native_merge_preserves_exact_document_order_and_ledger',
             'test_native_merge_grows_part_table_under_low_fd_limit'] + [
        f'test_writeback_or_close_error_prevents_success_receipt[{failure}]' for failure in
        ('fsync_tokens','fsync_lengths','close_tokens','close_lengths')]
    names += ['publication_check_'+str(i) for i in range(9)]
    suite = ET.Element('testsuite', tests='15', errors='0', failures='0', skipped='0')
    for name in names:
        ET.SubElement(suite, 'testcase', name=name)
    return suite


def test_complete_actual_fault_and_low_fd_coverage_is_required():
    suite = passing_receipt()
    assert len(require_test_results(ET.tostring(suite))) == 15


@pytest.mark.parametrize('fault', ['skip','failure','missing_fault','wrong_count'])
def test_false_green_or_incomplete_native_receipts_never_authorize_a_freeze(fault):
    suite = passing_receipt()
    if fault == 'skip': ET.SubElement(suite[0], 'skipped')
    elif fault == 'failure': suite.set('failures','1')
    elif fault == 'missing_fault': suite[2].set('name','unrelated_passing_case')
    else: suite.set('tests','14')
    with pytest.raises(ValueError):
        require_test_results(ET.tostring(suite))
