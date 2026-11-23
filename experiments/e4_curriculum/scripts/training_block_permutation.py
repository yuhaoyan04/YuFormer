"""Offline independent permutation of complete causal blocks within each WSD stage."""
from __future__ import annotations

import hashlib
import numpy as np

ALGORITHM = 'blake2b-128-independent-causal-block-sort-v1'


def build_permutation(samples: int, block_samples: int, *, seed: str, phase: str):
    if (type(samples) is not int or type(block_samples) is not int or samples < 1 or
            block_samples < 1 or samples % block_samples or samples // block_samples >= 2**32):
        raise ValueError('stage sample count must be positive and divisible by the explicit block size')
    blocks = samples // block_samples
    context = f'{ALGORITHM}\0{seed}\0{phase}\0{samples}\0{block_samples}'.encode()
    key = hashlib.blake2b(context, digest_size=32).digest()
    keys = np.fromiter((hashlib.blake2b(i.to_bytes(8, 'little'), key=key, digest_size=16).digest()
                        for i in range(blocks)), dtype='S16', count=blocks)
    permutation = np.argsort(keys, kind='stable').astype('<u4')
    validate_permutation(permutation, samples, block_samples)
    return permutation


def validate_permutation(permutation, samples: int, block_samples: int):
    if (type(samples) is not int or type(block_samples) is not int or samples < 1 or
            block_samples < 1 or samples % block_samples):
        raise ValueError('invalid permutation dimensions')
    blocks = samples // block_samples
    if (permutation.ndim != 1 or len(permutation) != blocks or permutation.dtype != np.dtype('<u4') or
            not np.array_equal(np.sort(permutation), np.arange(blocks, dtype='<u4'))):
        raise ValueError('permutation must cover every causal block exactly once')


def physical_sample(local_sample: int, permutation, block_samples: int) -> int:
    if local_sample < 0 or block_samples < 1:
        raise ValueError('invalid logical sample or block size')
    block, within = divmod(local_sample, block_samples)
    if block >= len(permutation):
        raise IndexError('logical stage sample is exhausted')
    return int(permutation[block]) * block_samples + within


def add_document_intervals(counts, difference, domain, starts, ends, *, block_tokens):
    """Accumulate exact per-domain token counts without expanding long documents."""
    if len(starts) != len(ends) or np.any(starts < 0) or np.any(starts >= ends):
        raise ValueError('invalid consumed document intervals')
    first, last = starts // block_tokens, (ends-1) // block_tokens
    if len(last) and np.max(last) >= len(counts):
        raise ValueError('document interval exceeds causal block population')
    same = first == last
    np.add.at(counts[:, domain], first[same], ends[same]-starts[same])
    split = ~same
    np.add.at(counts[:, domain], first[split], block_tokens-starts[split] % block_tokens)
    np.add.at(counts[:, domain], last[split], (ends[split]-1) % block_tokens+1)
    middle = last > first+1
    np.add.at(difference[:, domain], first[middle]+1, block_tokens)
    np.add.at(difference[:, domain], last[middle], -block_tokens)
