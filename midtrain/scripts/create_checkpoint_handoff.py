"""Hash a complete native checkpoint once before handing it to the next stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

from yuformer_midtrain.window_release import stat_identity


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def create(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() or output.parent == checkpoint:
        raise ValueError('handoff must be a new immutable object outside the checkpoint directory')
    if not checkpoint.name.startswith('step-') or not checkpoint.name[5:].isdigit():
        raise ValueError('handoff requires an explicit native step checkpoint directory')
    identity = checkpoint.parent.parent/'run_identity.json'
    if not identity.is_file():
        raise ValueError('handoff checkpoint has no actual run identity')
    metadata_path = checkpoint/'.metadata'
    metadata_raw = metadata_path.read_bytes()
    metadata_sha = hashlib.sha256(metadata_raw).hexdigest()
    metadata = pickle.loads(metadata_raw)
    required = {'.metadata'}
    for info in metadata.storage_data.values():
        name = Path(info.relative_path)
        if (name.is_absolute() or name.name != str(name) or info.offset < 0 or info.length <= 0
                or not (checkpoint/name).is_file()
                or info.offset+info.length > (checkpoint/name).stat().st_size):
            raise ValueError('checkpoint storage metadata references an absent/truncated/unsupported shard')
        required.add(str(name))
    files = {p.name for p in checkpoint.iterdir() if p.is_file()}
    if files != required:
        raise ValueError('checkpoint has unreferenced artifacts or is incomplete')
    roots = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), roots/'src/yuformer_midtrain/window_release.py',
               roots/'src/yuformer_midtrain/checkpoint_contract.py']
    pins = {str(path): digest(path) for path in sources}
    rows = []
    for name in sorted(files):
        path = checkpoint/name
        before = stat_identity(path.stat())
        sha = digest(path)
        if before != stat_identity(path.stat()) or before['bytes'] <= 0:
            raise ValueError('checkpoint shard changed while hashing')
        rows.append({'file': name, 'sha256': sha, 'stat': before})
    if (digest(metadata_path) != metadata_sha
            or any(digest(path) != sha for path, sha in pins.items())
            or any(stat_identity((checkpoint/row['file']).stat()) != row['stat'] for row in rows)):
        raise ValueError('checkpoint metadata/shards/producer changed during complete byte snapshot')
    result = {'schema_version': 1, 'status': 'PASS_COMPLETE_CHECKPOINT_BYTE_SNAPSHOT',
        'production_qualified': False, 'checkpoint': str(checkpoint), 'artifacts': rows,
        'run_identity_sha256': digest(identity), 'metadata_sha256': metadata_sha,
        'producer_sha256': digest(__file__), 'source_inputs': pins,
        'note': 'Complete-byte integrity only; model/stage/step and numeric transition are checked independently.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.tmp-{os.getpid()}')
    with temporary.open('xb') as handle:
        handle.write((json.dumps(result, sort_keys=True, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = create(args.checkpoint, args.output)
    print(json.dumps({'status': result['status'], 'files': len(result['artifacts']),
                      'bytes': sum(row['stat']['bytes'] for row in result['artifacts'])}, sort_keys=True))
