"""Count explicit, block-aligned source windows without changing physical data."""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DOMAINS = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')
POLICY = 'prefix-if-valid-else-minimax-65-block-aligned-windows-v1'


def stat_identity(value):
    return {'bytes': value.st_size, 'device': value.st_dev,
            'inode': value.st_ino, 'mtime_ns': value.st_mtime_ns}


def count_source_windows(order, ledger, *, seq_len, stage_samples, block_samples=8,
                         candidate_count=65, ledger_alternate=None, progress=None):
    if (type(seq_len) is not int or seq_len < 1 or type(stage_samples) is not int
            or stage_samples < 1 or type(block_samples) is not int or block_samples < 1
            or stage_samples % block_samples or type(candidate_count) is not int
            or not 2 <= candidate_count <= 65):
        raise ValueError('source windows require exact positive whole-block geometry')
    order, ledger = Path(order), Path(ledger)
    with ledger.open('rb') as lengths_handle, order.open('rb') as order_handle:
        ledger_stat = stat_identity(os.fstat(lengths_handle.fileno()))
        order_stat = stat_identity(os.fstat(order_handle.fileno()))
        if ledger_stat['bytes'] == 0 or ledger_stat['bytes'] % 4:
            raise ValueError('invalid complete document-length ledger')
        lengths = np.memmap(lengths_handle, dtype='<u4', mode='r')
        total = int(lengths.sum(dtype=np.uint64))
        capacity = (total-1)//seq_len
        if stage_samples > capacity:
            raise ValueError('source window exceeds the actual token capacity')
        maximum_blocks = (capacity-stage_samples)//block_samples
        offsets = np.unique(np.linspace(0, maximum_blocks,
                            num=min(candidate_count, maximum_blocks+1), dtype=np.int64))*block_samples
        trained = stage_samples*seq_len
        edges = sorted({int(offset)*seq_len+delta for offset in offsets for delta in (1, trained+1)})
        prefixes, running = {}, np.zeros(len(DOMAINS), dtype=np.uint64)
        edge_index = rows = token_offset = 0
        parquet = pq.ParquetFile(order_handle)
        if parquet.metadata.num_rows != len(lengths):
            raise ValueError('source order and ledger populations differ')
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=['position', 'domain']):
            positions = batch.column(0).to_numpy(zero_copy_only=False)
            if not np.array_equal(positions, np.arange(rows, rows+batch.num_rows)):
                raise ValueError('source-window scan requires canonical position-ordered source rows')
            values = np.asarray(lengths[rows:rows+batch.num_rows], dtype=np.uint64)
            if np.any(values == 0):
                raise ValueError('source document length must include a positive EOD interval')
            ends = np.cumsum(values, dtype=np.uint64)+token_offset
            codes = pc.index_in(pc.cast(batch.column(1), pa.string()), value_set=pa.array(DOMAINS))
            if codes.null_count:
                raise ValueError('unknown source-window domain')
            codes = codes.to_numpy(zero_copy_only=False)
            while edge_index < len(edges) and edges[edge_index] <= int(ends[-1]):
                edge = edges[edge_index]
                index = int(np.searchsorted(ends, edge, side='left'))
                prefix = running+np.bincount(codes[:index], weights=values[:index],
                                             minlength=len(DOMAINS)).astype(np.uint64)
                before = token_offset if index == 0 else int(ends[index-1])
                prefix[int(codes[index])] += edge-before
                prefixes[edge] = prefix
                edge_index += 1
            running += np.bincount(codes, weights=values, minlength=len(DOMAINS)).astype(np.uint64)
            rows += batch.num_rows
            token_offset = int(ends[-1])
            if progress is not None:
                progress(rows, len(lengths))
        if rows != len(lengths) or token_offset != total or edge_index != len(edges):
            raise ValueError('source-window population/prefix accounting does not close')
        alternate = Path(ledger_alternate) if ledger_alternate is not None else ledger
        current_ledger = ledger if ledger.exists() else alternate
        if (stat_identity(os.fstat(lengths_handle.fileno())) != ledger_stat
                or stat_identity(current_ledger.stat()) != ledger_stat
                or stat_identity(os.fstat(order_handle.fileno())) != order_stat
                or stat_identity(order.stat()) != order_stat):
            raise ValueError('source window inputs changed while counting')
    candidates = []
    for offset in map(int, offsets):
        start, stop = 1+offset*seq_len, 1+(offset+stage_samples)*seq_len
        counts = prefixes[stop]-prefixes[start]
        if int(counts.sum()) != trained:
            raise ValueError('source window changed the supervised token budget')
        candidates.append({'source_sample_offset': offset, 'target_interval': [start, stop],
                           'domain_tokens': dict(zip(DOMAINS, map(int, counts)))})
    return {'policy': POLICY, 'seq_len': seq_len, 'stage_sample_limit': stage_samples,
            'block_samples': block_samples, 'source_capacity_samples': capacity,
            'total_source_tokens': total, 'source_documents': rows,
            'trained_target_tokens': trained, 'ledger_stat': ledger_stat, 'order_stat': order_stat,
            'whole_population_domain_tokens': dict(zip(DOMAINS, map(int, running))),
            'candidates': candidates}


def choose_source_window(counted, target_weights, *, absolute_tolerance=.002, relative_tolerance=.1):
    from audit_1t_overlap import audit_consumed_mixture
    evaluated = []
    for candidate in counted['candidates']:
        mixture = audit_consumed_mixture(candidate['domain_tokens'], target_weights,
            absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
        score = max(max(row['absolute_error']/absolute_tolerance,
                        row['relative_error']/relative_tolerance)
                    for row in mixture['domains'].values())
        evaluated.append({**candidate, 'mixture': mixture, 'normalized_maximum_error': score})
    admitted = [row for row in evaluated if row['mixture']['status'] == 'PASS']
    prefix = next(row for row in evaluated if row['source_sample_offset'] == 0)
    selected = (prefix if prefix['mixture']['status'] == 'PASS' else
                min(admitted, key=lambda row: (row['normalized_maximum_error'], row['source_sample_offset']))
                if admitted else None)
    return {**counted, 'status': 'PASS_WINDOW_COUNTS' if selected is not None else 'BLOCKED_NO_COMPLIANT_WINDOW',
            'selected': selected, 'candidates': evaluated,
            'physical_streams_changed': False,
            'supervised_prefix_unchanged': selected is not None and selected['source_sample_offset'] == 0}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_consumption_window(root, phase, order, ledger, prefix_output, *, context, seq_len,
                               stage_samples, targets, prefix, prefix_mixture,
                               absolute_tolerance, relative_tolerance, block_samples=8):
    from audit_1t_overlap import extract_consumed, audit_consumed_mixture
    root, order, ledger, prefix_output = map(Path, (root, order, ledger, prefix_output))
    if absolute_tolerance != .002 or relative_tolerance != .1:
        raise ValueError('source window admission must retain the original .002/.1 domain tolerances')
    trained = stage_samples*seq_len
    if prefix['target_start'] != 1 or prefix['target_stop'] != trained+1:
        raise ValueError('source window selection requires the verified original prefix')
    if prefix_mixture['status'] == 'PASS':
        counted = {'policy': POLICY, 'seq_len': seq_len, 'stage_sample_limit': stage_samples,
            'block_samples': block_samples, 'trained_target_tokens': trained,
            'total_source_tokens': context['total_source_tokens'],
            'source_capacity_samples': (context['total_source_tokens']-1)//seq_len,
            'candidates': [{'source_sample_offset': 0, 'target_interval': [1, trained+1],
                            'domain_tokens': prefix['domain_tokens']}]}
        searched = False
    elif prefix_mixture['status'] == 'FAIL':
        counted = count_source_windows(order, ledger, seq_len=seq_len, stage_samples=stage_samples,
                                        block_samples=block_samples, candidate_count=65)
        if (counted['total_source_tokens'] != context['total_source_tokens']
                or set(prefix['domain_tokens'])-set(DOMAINS)
                or counted['candidates'][0]['domain_tokens'] != {d:prefix['domain_tokens'].get(d, 0) for d in DOMAINS}):
            raise ValueError('source window scan disagrees with exact original consumed ID records')
        searched = True
    else:
        raise ValueError('source window selection cannot bypass a failed source or byte audit')
    selection = choose_source_window(counted, targets, absolute_tolerance=absolute_tolerance,
                                      relative_tolerance=relative_tolerance)
    if selection['status'] != 'PASS_WINDOW_COUNTS':
        raise ValueError('no block-aligned contiguous source window meets the original mixture gates')
    chosen = selection['selected']
    offset = chosen['source_sample_offset']
    start, stop = chosen['target_interval']
    output = prefix_output if offset == 0 else prefix_output.with_name(f'{phase}.window-{offset}.consumed.records')
    extracted = prefix if offset == 0 else extract_consumed(order, ledger, output,
        identity=f"{prefix['identity']}:source_sample_offset={offset}",
        trained_tokens=trained, target_start=start)
    mixture = audit_consumed_mixture(extracted['domain_tokens'], targets,
        absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
    if (mixture['status'] != 'PASS' or extracted['domain_tokens'] != chosen['domain_tokens']
            or extracted['target_start'] != start or extracted['target_stop'] != stop
            or offset % block_samples or offset+stage_samples > counted['source_capacity_samples']):
        raise ValueError('exact window ID records disagree with selected domain counts/geometry')
    meta_path = output.with_suffix('.json')
    payload = {**selection, 'schema_version': 1, 'scope': 'EXACT_CONSUMPTION_WINDOW_ADMISSION',
        'phase': phase, 'context': context, 'search_performed': searched,
        'exact_consumption': extracted, 'selector_tool_sha256': _sha(Path(__file__)),
        'consumption_records': {'file': str(output.relative_to(root)), 'sha256': extracted['sha256'],
            'stat': stat_identity(output.stat()), 'meta_file': str(meta_path.relative_to(root)),
            'meta_sha256': _sha(meta_path)}}
    raw = (json.dumps(payload, sort_keys=True, indent=2)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = root/'data/processed/training_windows'/context['release_manifest_sha256']
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/f'{phase}-{digest}.json'
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError('immutable source window evidence collision')
    else:
        temporary = destination.with_name(destination.name+f'.tmp-{os.getpid()}')
        with temporary.open('wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    binding = {'policy': POLICY, 'source_sample_offset': offset, 'block_samples': block_samples,
        'target_interval': [start, stop], 'physical_streams_changed': False,
        'supervised_prefix_unchanged': offset == 0,
        'selection_report': {'file': str(destination.relative_to(root)), 'sha256': digest}}
    return extracted, mixture, binding


def require_window_admission(root, report, *, phase, stage_samples, seq_len, release_sha256,
                             stream_meta_sha256, data_frozen_recipe_sha256):
    from training_data_origin import read_bound
    root = Path(root)
    binding = report.get('source_window') or {}
    reference = binding.get('selection_report') or {}
    name = reference.get('file')
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('source window admission requires an immutable repository-relative report')
    raw, digest = read_bound(root/name, reference.get('sha256'))
    document = json.loads(raw)
    context, selected = document.get('context') or {}, document.get('selected') or {}
    offset, block = binding.get('source_sample_offset'), binding.get('block_samples')
    interval = [1+offset*seq_len, 1+(offset+stage_samples)*seq_len] if type(offset) is int else None
    if (document.get('scope') != 'EXACT_CONSUMPTION_WINDOW_ADMISSION'
            or document.get('status') != 'PASS_WINDOW_COUNTS' or document.get('phase') != phase
            or document.get('policy') != POLICY or binding.get('policy') != POLICY
            or document.get('seq_len') != seq_len or document.get('stage_sample_limit') != stage_samples
            or type(offset) is not int or offset < 0 or type(block) is not int or block < 1
            or offset % block or stage_samples % block or document.get('block_samples') != block
            or context.get('release_manifest_sha256') != release_sha256
            or context.get('stream_meta_sha256') != stream_meta_sha256
            or context.get('data_frozen_recipe_sha256') != data_frozen_recipe_sha256
            or document.get('selector_tool_sha256') != _sha(Path(__file__))
            or document.get('source_capacity_samples') != (context.get('total_source_tokens', 0)-1)//seq_len
            or offset+stage_samples > document.get('source_capacity_samples', -1)
            or selected.get('source_sample_offset') != offset or selected.get('target_interval') != interval
            or binding.get('target_interval') != interval or binding.get('physical_streams_changed') is not False
            or report.get('status') != 'PASS' or report.get('stage') != phase
            or selected.get('mixture') != report.get('mixture')
            or report.get('mixture', {}).get('status') != 'PASS'
            or report.get('mixture', {}).get('absolute_tolerance') != .002
            or report.get('mixture', {}).get('relative_tolerance') != .1
            or report.get('mixture', {}).get('consumed_tokens') != stage_samples*seq_len
            or report.get('consumed', {}).get('consumed_tokens') != stage_samples*seq_len
            or selected.get('domain_tokens') != {d:r['tokens'] for d,r in report.get('mixture', {}).get('domains', {}).items()}
            or document.get('exact_consumption') != report.get('consumed')
            or report.get('consumed', {}).get('target_start') != interval[0]
            or report.get('consumed', {}).get('target_stop') != interval[1]):
        raise ValueError('source window evidence differs from actual consumed geometry/data/producer')
    records = document.get('consumption_records') or {}
    for key in ('file', 'meta_file'):
        value = records.get(key)
        if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('invalid exact source-window record evidence path')
    if (stat_identity((root/records['file']).stat()) != records.get('stat')
            or records.get('sha256') != report['consumed']['sha256']):
        raise ValueError('exact source-window records changed after admission')
    record_meta, _ = read_bound(root/records['meta_file'], records.get('meta_sha256'))
    if json.loads(record_meta) != report['consumed']:
        raise ValueError('source-window record metadata does not match its admitted counts')
    return {'source_sample_offset': offset, 'block_samples': block, 'report': document,
            'inputs': {name: digest, records['meta_file']: records['meta_sha256'],
                       str(Path(__file__).resolve().relative_to(root.resolve())): _sha(Path(__file__))}}
