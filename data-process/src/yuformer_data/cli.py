from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .adapters import DatasetAdapter
from .io import iter_jsonl, write_jsonl
from .manifest import build_manifest, write_manifest
from .models import DataCategory, Decision, Sample, Stage, release_record
from .pipelines import build_pipeline


def load_config(config_path: str | None, cleaning_config_path: str | None = None) -> dict:
    cfg: dict = {}
    if cleaning_config_path and Path(cleaning_config_path).exists():
        with open(cleaning_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            raw = f.read().strip()
            if raw:
                data = json.loads(raw)
                if "quality_policies" in data:
                    cfg["quality_policies"] = data["quality_policies"]
                    for k in ("near_dedup", "near_dedup_threshold", "near_dedup_bands", "near_dedup_ngram"):
                        if k in data:
                            cfg[k] = data[k]
                else:
                    cfg["quality_policies"] = data
    if "quality_policies" not in cfg:
        cfg["quality_policies"] = None
    return cfg


def run_jsonl(
    input_path: str,
    output_path: str,
    *,
    stage: Stage,
    category: DataCategory,
    dataset: str,
    config: dict,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    field_map: tuple[tuple[str, str], ...] = (),
    manifest_path: str | None = None,
    summary_path: str | None = None,
    keep_dropped: bool = False,
) -> int:
    adapter = DatasetAdapter(
        dataset=dataset,
        stage=stage,
        category=category,
        id_field=id_field,
        text_fields=text_fields,
        field_map=field_map,
    )
    pipeline = build_pipeline(category, config=config)

    records: list[dict] = []
    for raw in iter_jsonl(input_path):
        sample = adapter.adapt(raw)
        sample = pipeline.run(sample)
        if sample.decision == Decision.DROP and not keep_dropped:
            continue
        records.append(release_record(sample))

    count = write_jsonl(output_path, records)

    summary = {
        "stage": stage.value,
        "category": category.value,
        "dataset": dataset,
        "written": count,
        "pipeline": pipeline.step_names(),
        "counters": dict(pipeline.context.counters),
    }
    if manifest_path:
        rows = list(iter_jsonl(output_path))
        manifest = build_manifest(rows, stage=stage.value, output=Path(output_path).name)
        write_manifest(manifest_path, manifest)
        summary["manifest"] = Path(manifest_path).name
    sp = Path(summary_path or (str(output_path) + ".summary.json"))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def main():
    parser = argparse.ArgumentParser(description="YuFormer data cleaning pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=[s.value for s in Stage])
    parser.add_argument("--category", required=True, choices=[c.value for c in DataCategory])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None, help="quality_policies JSON path")
    parser.add_argument("--cleaning-config", default=None, help="cleaning_config.yaml path")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-fields", default="text", help="comma-separated field names")
    parser.add_argument("--field-map", default="", help="src1:dst1,src2:dst2")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--keep-dropped", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    cleaning_config_path = args.cleaning_config or str(script_dir / "configs" / "cleaning_config.yaml")
    config = load_config(args.config, cleaning_config_path)

    text_fields = tuple(args.text_fields.split(","))
    field_map: tuple[tuple[str, str], ...] = ()
    if args.field_map:
        pairs = [p.split(":") for p in args.field_map.split(",")]
        field_map = tuple((p[0], p[1]) for p in pairs if len(p) == 2)

    count = run_jsonl(
        args.input,
        args.output,
        stage=Stage(args.stage),
        category=DataCategory(args.category),
        dataset=args.dataset,
        config=config,
        id_field=args.id_field,
        text_fields=text_fields,
        field_map=field_map,
        manifest_path=args.manifest,
        summary_path=args.summary,
        keep_dropped=args.keep_dropped,
    )
    print(f"Wrote {count} records to {args.output}")
