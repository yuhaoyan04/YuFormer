#!/usr/bin/env python3
"""Wait for formal 4K completion, then execute the frozen production ladder."""
import argparse
import json
from pathlib import Path
import sys
import signal
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'eval/src'), str(ROOT/'pretrain/src'), str(ROOT/'pretrain/scripts')]
from yuformer_eval.context_pipeline import ContextPipeline, formal_completion_state
from yuformer_pretrain.lease import RunLease
from process_watchdog import enable_subreaper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    pipeline = ContextPipeline(args.config)
    if args.validate_only:
        print(json.dumps({'status': 'PASS_FROZEN_CHAIN_CONTROLLER_PREFLIGHT_NO_GPU_LAUNCH',
            'formal_state': formal_completion_state(pipeline.config['formal_run'])}))
        return 0
    enable_subreaper()
    lease = RunLease(pipeline.output, 'authorized-full-pretrain-context-chain',
                     heartbeat_seconds=3600, stale_seconds=600).acquire()
    pipeline.lease = lease
    def interrupted(signum, frame):
        raise KeyboardInterrupt('context controller received signal '+str(signum))
    old_handlers = {signum: signal.signal(signum, interrupted)
                    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.publish('STOPPED_REQUIRES_DIAGNOSIS_NO_STAGE_GATE_BYPASS',
                         error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        pipeline.cleanup_owned()
        lease.release()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
