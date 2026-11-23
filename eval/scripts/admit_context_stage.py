#!/usr/bin/env python3
"""Score actual paired native measurements or recompute an existing admission."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from yuformer_eval.stage_admission import publish_admission, verify_admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.output:
            parser.error('--verify does not create or rewrite admission evidence')
        report = verify_admission(args.verify)
    else:
        if args.output is None:
            parser.error('--spec requires a new --output')
        report = publish_admission(args.spec, args.output)
    print(json.dumps({'status': report['status'], 'seq_len': report['seq_len'],
        'step': report['step'], 'suites': {name: value['status'] for name, value in report['suites'].items()}}))
    return 0 if report['status'] == 'PASS_STAGE_LONG_AND_SHORT_EVALUATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
