from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import DataCategory, Sample, Stage

Hook = Callable[[Sample], Sample]


@dataclass(frozen=True, slots=True)
class DatasetAdapter:
    dataset: str
    stage: Stage
    category: DataCategory
    id_field: str = "id"
    text_fields: tuple[str, ...] = ("text",)
    field_map: tuple[tuple[str, str], ...] = ()
    hooks: tuple[Hook, ...] = ()

    def adapt(self, raw: dict[str, Any], index: int = 0) -> Sample:
        mapped: dict[str, Any] = dict(raw)
        for source_name, canonical_name in self.field_map:
            if source_name in mapped and canonical_name not in mapped:
                mapped[canonical_name] = mapped[source_name]
        source_id = str(mapped.get(self.id_field) or f"{self.dataset}:{index}")
        sample = Sample(
            dataset=self.dataset,
            source_id=source_id,
            stage=self.stage,
            category=self.category,
            data=mapped,
            text_fields=self.text_fields,
        )
        for hook in self.hooks:
            sample = hook(sample)
        return sample

    def adapt_many(self, rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
        for index, row in enumerate(rows):
            yield self.adapt(row, index)
