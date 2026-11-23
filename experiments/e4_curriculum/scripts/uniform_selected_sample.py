"""Independent, reproducible document-rank sampling of a complete selected order."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import calibrate_training_tokens as calibration

SAMPLING = 'independent-blake2b-domain-rank-permutation-v1'


def draw_ranks(population: int, count: int, *, seed: str, phase: str, domain: str):
    if not 0 < count <= population < 2**63:
        raise ValueError('invalid rank population or sample count')
    key = hashlib.blake2b((SAMPLING + '\0' + seed + '\0' + phase + '\0' + domain).encode(),
                          digest_size=32).digest()
    limit = 2**64 - 2**64 % population
    chosen = {}
    counter = 0
    while len(chosen) < count:
        value = int.from_bytes(hashlib.blake2b(counter.to_bytes(8, 'little'), key=key,
                                               digest_size=8).digest(), 'little')
        counter += 1
        if value < limit:
            rank = value % population
            if rank not in chosen:
                chosen[rank] = len(chosen)
    ranks = np.array(sorted(chosen), dtype=np.int64)
    priorities = np.fromiter((chosen[int(r)] for r in ranks), dtype=np.int64, count=count)
    return ranks, priorities


def sample_order(parent: Path, target: int, destination: Path, *, population: dict,
                 expected_parent_sha256: str, seed: str, phase: str, batch_size=262144):
    domains = tuple(calibration.DOMAINS)
    if set(population) != set(domains) or target <= 0:
        raise ValueError('invalid population domain roster or target')
    ranks, priorities = {}, {}
    for domain in domains:
        count = int(population[domain]['selected_docs'])
        tokens = int(population[domain]['selected_tokens'])
        if count <= 0 or tokens < target:
            raise ValueError('insufficient population reference tokens: ' + domain)
        size = min(count, math.ceil(2 * target * count / tokens) + 128)
        ranks[domain], priorities[domain] = draw_ranks(count, size, seed=seed, phase=phase, domain=domain)
    seen = {d: 0 for d in domains}
    totals = {d: 0 for d in domains}
    pieces = []
    previous = None
    with parent.open('rb') as handle:
        identity = calibration.materializer._stat_identity(os.fstat(handle.fileno()))
        parent_sha = calibration.materializer._hash_open_file(handle)
        if parent_sha != expected_parent_sha256:
            raise ValueError('selected population order SHA-256 mismatch')
        handle.seek(0)
        for batch in pq.ParquetFile(handle).iter_batches(
                batch_size=batch_size, columns=['doc_id', 'domain', 'n_tokens']):
            ids = batch.column(0)
            if (batch.num_rows == 0 or ids.null_count or
                    (previous is not None and previous >= ids[0].as_py()) or
                    (batch.num_rows > 1 and pc.any(pc.greater_equal(ids.slice(0, batch.num_rows-1),
                                                                   ids.slice(1))).as_py())):
                raise ValueError('selected population doc IDs must be strictly sorted')
            previous = ids[-1].as_py()
            codes = pc.index_in(pc.cast(batch.column(1), pa.string()), value_set=pa.array(domains))
            if codes.null_count:
                raise ValueError('unknown population domain')
            codes = codes.to_numpy(zero_copy_only=False)
            refs = batch.column(2).to_numpy(zero_copy_only=False)
            if batch.column(2).null_count or np.any(refs <= 0):
                raise ValueError('invalid population reference lengths')
            for index, domain in enumerate(domains):
                locations = np.flatnonzero(codes == index)
                start, end = seen[domain], seen[domain] + len(locations)
                lo, hi = np.searchsorted(ranks[domain], [start, end])
                if hi > lo:
                    indices = locations[ranks[domain][lo:hi] - start]
                    part = pa.Table.from_batches([batch]).take(pa.array(indices))
                    pieces.append(part.append_column('_draw_priority', pa.array(priorities[domain][lo:hi])))
                seen[domain] = end
                totals[domain] += int(refs[locations].sum(dtype=np.int64))
        if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                calibration.materializer.source_file_identity(parent) != identity):
            raise ValueError('selected population order changed during independent sampling')
    for domain in domains:
        if (seen[domain] != population[domain]['selected_docs'] or
                totals[domain] != population[domain]['selected_tokens']):
            raise ValueError('selected population counts differ from manifest: ' + domain)
    candidates = pa.concat_tables(pieces)
    chosen = []
    counts = {}
    for domain in domains:
        part = candidates.filter(pc.equal(candidates['domain'], domain))
        part = part.sort_by([('_draw_priority', 'ascending')])
        refs = part['n_tokens'].combine_chunks().to_numpy(zero_copy_only=False)
        cumulative = np.cumsum(refs, dtype=np.int64)
        if not len(cumulative) or cumulative[-1] < target:
            raise ValueError('independent rank sample buffer insufficient; enlarge explicit buffer: ' + domain)
        keep = int(np.searchsorted(cumulative, target)) + 1
        chosen.append(part.slice(0, keep).drop(['_draw_priority']))
        counts[domain] = int(cumulative[keep-1])
    table = pa.concat_tables(chosen).sort_by([('doc_id', 'ascending')])
    table = table.append_column('position', pa.array(np.arange(table.num_rows), type=pa.int64()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}')
    pq.write_table(table, temporary, compression='zstd')
    os.replace(temporary, destination)
    return {'sampling': SAMPLING, 'seed': seed, 'phase': phase,
            'parent_order_sha256': parent_sha, 'parent_order_stat': identity,
            'population_docs_by_domain': seen, 'population_reference_tokens_by_domain': totals,
            'candidate_docs_by_domain': {d: len(ranks[d]) for d in domains},
            'reference_tokens_by_domain': counts, 'docs': table.num_rows,
            'sampler_tool_sha256': calibration.digest(__file__)}
