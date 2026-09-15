from __future__ import annotations

import hashlib
import math
from bisect import bisect, bisect_left, insort
from dataclasses import dataclass, field
from typing import Mapping, Sequence

IGNORE_INDEX = -100


def stable_hash(value: str, *, digest_size: int = 8) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=digest_size).digest()


def assign_context_bucket(token_count: int, context_lengths: Sequence[int]) -> str:
    boundaries = sorted(set(int(b) for b in context_lengths if int(b) > 0))
    if not boundaries:
        raise ValueError("context_lengths must contain positive values")
    value = int(token_count)
    for upper in boundaries:
        if value <= upper:
            return f"B{upper // 1024}K"
    return f"gt_{boundaries[-1] // 1024}K"


def allocate_mix_quotas(weights: Mapping[str, float], total_samples: int) -> dict[str, int]:
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    clean = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    if not clean:
        raise ValueError("no positive weights")
    total_weight = sum(clean.values())
    exact = {k: total_samples * v / total_weight for k, v in clean.items()}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    remaining = total_samples - sum(quotas.values())
    order = sorted(clean, key=lambda k: (-(exact[k] - quotas[k]), k))
    for k in order[:remaining]:
        quotas[k] += 1
    return quotas


def deterministic_shuffle_key(record_key: str, seed: str = "yuformer") -> bytes:
    return hashlib.blake2b(f"{seed}\0{record_key}".encode("utf-8"), digest_size=16).digest()


def select_mixed_sample_ids(
    source_to_ids: Mapping[str, Sequence[str]],
    quotas: Mapping[str, int],
    *,
    seed: str = "yuformer",
) -> list[str]:
    selected: list[tuple[bytes, str]] = []
    for source in sorted(quotas):
        quota = int(quotas[source])
        if quota < 0:
            raise ValueError(f"negative quota for {source}")
        ranked = sorted(
            (str(rid) for rid in source_to_ids.get(source, ())),
            key=lambda rid: deterministic_shuffle_key(rid, seed=f"{seed}:{source}"),
        )
        if len(ranked) < quota:
            raise ValueError(f"{source} needs {quota} but only {len(ranked)} available")
        for rid in ranked[:quota]:
            selected.append((deterministic_shuffle_key(rid, seed=f"{seed}:mix"), rid))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [rid for _, rid in selected]


@dataclass(frozen=True, slots=True)
class WorkUnit:
    key: str
    weight: int


def balance_work_units(units: Sequence[WorkUnit], workers: int) -> list[list[WorkUnit]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    groups: list[list[WorkUnit]] = [[] for _ in range(workers)]
    loads: list[tuple[int, int]] = [(0, w) for w in range(workers)]
    for unit in sorted(units, key=lambda u: (-u.weight, u.key)):
        load, worker = loads.pop(0)
        groups[worker].append(unit)
        insort(loads, (load + max(0, int(unit.weight)), worker))
    return groups


@dataclass(frozen=True, slots=True)
class TokenizedSample:
    tokens: tuple[int, ...]
    source_id: str
    targets: tuple[int, ...] | None = None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class PackedSequence:
    tokens: list[int] = field(default_factory=list)
    targets: list[int] | None = None
    cu_seqlens: list[int] = field(default_factory=lambda: [0])
    source_ids: list[str] = field(default_factory=list)


def build_loss_mask(targets: Sequence[int], *, ignore_index: int = IGNORE_INDEX) -> tuple[int, ...]:
    return tuple(0 if t == ignore_index else 1 for t in targets)


def validate_tokenized_sample(sample: TokenizedSample) -> None:
    if not sample.tokens:
        raise ValueError("tokenized sample is empty")
    if sample.targets is None:
        return
    if len(sample.tokens) != len(sample.targets):
        raise ValueError("tokens and targets length mismatch")
    if all(t == IGNORE_INDEX for t in sample.targets):
        raise ValueError("SFT sample has no trainable target token")


def pack_tokenized_samples(
    samples: Sequence[TokenizedSample],
    *,
    sequence_length: int,
    allow_overlength: bool = False,
) -> tuple[list[PackedSequence], list[TokenizedSample]]:
    accepted: list[TokenizedSample] = []
    overlength: list[TokenizedSample] = []
    for sample in samples:
        validate_tokenized_sample(sample)
        if sample.length > sequence_length:
            overlength.append(sample)
            if not allow_overlength:
                continue
        accepted.append(sample)
    accepted.sort(key=lambda s: (-s.length, s.source_id))
    free: list[tuple[int, int]] = []
    bins: list[PackedSequence] = []
    for sample in accepted:
        position = bisect_left(free, (sample.length, -1))
        if position == len(free):
            packed = PackedSequence(targets=[] if sample.targets is not None else None)
            bins.append(packed)
            bin_id = len(bins) - 1
        else:
            _, bin_id = free.pop(position)
            packed = bins[bin_id]
        if (packed.targets is None) != (sample.targets is None):
            raise ValueError("cannot mix pretraining and SFT samples in one packing call")
        packed.tokens.extend(sample.tokens)
        if packed.targets is not None and sample.targets is not None:
            packed.targets.extend(sample.targets)
        packed.source_ids.append(sample.source_id)
        packed.cu_seqlens.append(len(packed.tokens))
        remaining = sequence_length - len(packed.tokens)
        if remaining > 0:
            insort(free, (remaining, bin_id))
    return bins, overlength


def prefix_context_mix(
    base_buckets: Mapping[str, Sequence[str]],
    final_targets: Mapping[str, int],
    *,
    seed: str = "yuformer",
    eligible_by_final: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    context_lengths = sorted(
        int(b) for b in (eligible_by_final or {}).keys()
        if str(b).startswith("B") and str(b).endswith("K")
    ) or [4096, 16384, 65536]

    names = []
    for cl in context_lengths:
        names.append(f"B{cl // 1024}K")

    if not eligible_by_final:
        eligible: dict[str, tuple[str, ...]] = {}
        for i, name in enumerate(names):
            eligible[name] = tuple(names[: i + 1])
    else:
        eligible = dict(eligible_by_final)

    selected: set[str] = set()
    result: dict[str, list[str]] = {}

    order = [n for n in names if n in final_targets]
    order.extend(n for n in final_targets if n not in order)

    for final_name in order:
        target = int(final_targets[final_name])
        if target < 0:
            raise ValueError(f"negative target for {final_name}")
        candidates = {
            str(rid)
            for bucket in eligible.get(final_name, ())
            for rid in base_buckets.get(bucket, ())
            if str(rid) not in selected
        }
        if len(candidates) < target:
            raise ValueError(
                f"{final_name} needs {target} but only {len(candidates)} available"
            )
        chosen = sorted(
            candidates,
            key=lambda rid: (
                hashlib.blake2b(
                    f"{seed}:{final_name}\0{rid}".encode("utf-8"), digest_size=16
                ).digest(),
                rid,
            ),
        )[:target]
        result[final_name] = chosen
        selected.update(chosen)
    return result
