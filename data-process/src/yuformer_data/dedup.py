from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\w+", re.UNICODE)


def simhash64(text: str, ngram: int = 5) -> int:
    tokens = _TOKEN.findall(text.casefold())
    if not tokens:
        return 0
    width = max(1, int(ngram))
    weights = [0] * 64
    for i in range(max(1, len(tokens) - width + 1)):
        feature = " ".join(tokens[i : i + width])
        digest = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit in range(64):
        if weights[bit] >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(slots=True)
class NearDeduper:
    threshold: int = 3
    bands: int = 4
    fingerprints: list[int] = field(default_factory=list)
    indexes: list[dict[int, set[int]]] = field(init=False)

    def __post_init__(self) -> None:
        if 64 % self.bands != 0 or self.threshold >= self.bands:
            raise ValueError("bands must divide 64 and threshold must be smaller than bands")
        self.indexes = [{} for _ in range(self.bands)]

    def _band_values(self, fingerprint: int) -> list[int]:
        width = 64 // self.bands
        mask = (1 << width) - 1
        values = []
        for band in range(self.bands):
            shift = band * width
            values.append((fingerprint >> shift) & mask)
        return values

    def check_and_add(self, text: str, ngram: int = 5) -> bool:
        fingerprint = simhash64(text, ngram=ngram)
        values = self._band_values(fingerprint)
        candidates: set[int] = set()
        for band, value in enumerate(values):
            candidates.update(self.indexes[band].get(value, set()))
        duplicate = any(
            hamming_distance(fingerprint, self.fingerprints[idx]) <= self.threshold
            for idx in candidates
        )
        idx = len(self.fingerprints)
        self.fingerprints.append(fingerprint)
        for band, value in enumerate(values):
            self.indexes[band].setdefault(value, set()).add(idx)
        return duplicate
