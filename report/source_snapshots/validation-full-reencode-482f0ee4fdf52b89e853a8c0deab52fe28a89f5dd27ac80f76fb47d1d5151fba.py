#!/usr/bin/env python3
"""Re-encode held-out streams separately and compare every existing token byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root/'pretrain/scripts'), str(root/'pretrain/src')]
    from source_document_binding import verified_source_document_binding
    from yuformer_pretrain.lease import RunLease
    domains = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')
    directory = root/'data/processed/audits/validation_reencode_20261114'
    original = root/'data/token_streams/m3_20261020_yf64k'
    health = root/'data/health/validation-full-reencode-audit.json'
    materializer = root/'pretrain/scripts/materialize_tokens.py'
    tokenizer = root/'tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json'
    gate = root/'data/TOKENIZER_READY.json'
    release = root/'data/processed/releases/mix_m3_e1_100b-20261019.release.json'
    paths = [Path(__file__).resolve(), materializer, tokenizer, gate, release,
             root/'pretrain/scripts/source_document_binding.py']
    paths += [root/f'data/processed/val_sets/{domain}.parquet' for domain in domains]
    paths += [original/f'val_{domain}.meta.json' for domain in domains]
    pinned = {str(path): digest(path) for path in paths}
    environment = subprocess.run([str(args.python), '-c',
        'import sys,tokenizers,pyarrow,json;print(json.dumps(dict(python=sys.version,tokenizers=tokenizers.__version__,pyarrow=pyarrow.__version__)))'],
        check=True, capture_output=True, text=True)
    state = {'schema_version': 1, 'pid': os.getpid(), 'status': 'RUNNING', 'results': {},
             'purpose': 'INDEPENDENT_FULL_VALIDATION_REENCODE_WITHOUT_REPLACING_PRODUCTION_ASSETS',
             'production_assets_modified': False, 'pinned_inputs': pinned}
    state['encoding_interpreter'] = str(args.python)
    state['encoding_runtime'] = json.loads(environment.stdout)

    def unchanged():
        for name, expected in pinned.items():
            if digest(Path(name)) != expected:
                raise ValueError('validation proof input changed: '+name)

    def save(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = health.with_name(health.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2)+'\n')
        os.replace(temporary, health)

    with RunLease(directory, 'independent-validation-full-reencode'):
        try:
            for domain in domains:
                unchanged()
                name = 'val_'+domain
                meta_path = original/(name+'.meta.json')
                old = json.loads(meta_path.read_bytes())
                if old['selection']['requested_max_reference_tokens'] != 20_000_000:
                    raise ValueError('unexpected original validation selection horizon: '+domain)
                log = directory/(domain+'.log')
                save(active_domain=domain, active_log=str(log))
                command = [str(args.python), str(materializer), '--order',
                    str(root/f'data/processed/val_sets/{domain}.parquet'),
                    '--s3-root', str(root/'data/processed/s3'), '--tokenizer', str(tokenizer),
                    '--out-dir', str(directory/'streams'), '--name', name,
                    '--max-tokens', '20000000', '--data-snapshot', str(release),
                    '--tokenizer-gate', str(gate), '--workers', '8', '--write-doc-lengths']
                with log.open('a') as output:
                    child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                        env=dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                                 OPENBLAS_NUM_THREADS='1'))
                    save(child_pid=child.pid)
                    code = child.wait()
                if code:
                    raise ValueError('validation re-encoding failed: '+domain+'; see '+str(log))
                new_path = directory/'streams'/(name+'.meta.json')
                new = json.loads(new_path.read_bytes())
                before = digest(original/(name+'.tokens.bin'))
                after = digest(directory/'streams'/(name+'.tokens.bin'))
                if (not verified_source_document_binding(new) or before != old['sha256']
                        or after != new['sha256'] or before != after
                        or any(new[key] != old[key] for key in ('bytes', 'n_docs', 'dtype', 'total_tokens',
                            'order_sha256', 'tokenizer_sha256', 'tokenizer_gate_sha256', 'data_snapshot_id'))):
                    raise ValueError('existing validation differs from independently source-verified token bytes: '+domain)
                state['results'][domain] = {'status': 'PASS_EXACT_BYTE_EQUALITY', 'tokens': old['total_tokens'],
                    'documents': old['n_docs'], 'original_and_reencoded_sha256': before,
                    'new_source_document_ids_verified': True,
                    'old_meta_sha256': pinned[str(meta_path)], 'new_meta': str(new_path.relative_to(root)),
                    'new_meta_sha256': digest(new_path), 'source_binding': new['source_document_binding']}
                save()
                print(domain, 'PASS_EXACT_BYTE_EQUALITY', old['total_tokens'], flush=True)
            unchanged()
            save(status='PASS_ALL_NINE_FULL_VALIDATION_STREAMS', active_domain=None, child_pid=None)
            return 0
        except Exception as error:
            save(status='FAIL_CLOSED', error=repr(error))
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
