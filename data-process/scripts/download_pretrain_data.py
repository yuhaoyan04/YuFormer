#!/usr/bin/env python3
"""YuFormer 预训练数据下载器

按 configs/data_sources.yaml 逐源下载，支持:
  - 每源 max_size_gb 上限（累计文件大小达标即停，跳到下一个源）
  - include/exclude glob 过滤
  - 断点续传（已下载且大小匹配的文件自动跳过）
  --dry-run  仅列出文件与大小，不下载
  --source NAME  只下载指定源
  --test         每源只下载第一个文件（验证可访问性）
  --data-root PATH  数据根目录（默认 ./data）

用法:
  # 列出所有源的文件与大小（不下载）
  python download_pretrain_data.py --dry-run --all

  # 测试每源第一个文件
  python download_pretrain_data.py --test --all

  # 下载单个源
  python download_pretrain_data.py --source proof-pile-2

  # 下载全部（按配置的 max_size_gb）
  python download_pretrain_data.py --all

  # 覆盖某源的大小上限
  python download_pretrain_data.py --source fineweb-edu --max-size 50
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)


@dataclass
class SourceConfig:
    name: str
    domain: str
    repo_id: str
    max_size_gb: float = 0
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class FileResult:
    path: str
    size_bytes: int
    status: str = ""       # downloaded / skipped / failed / excluded
    local_path: str = ""
    error: str = ""


@dataclass
class SourceResult:
    name: str
    repo_id: str
    domain: str
    status: str = ""       # ok / partial / failed / empty
    total_files: int = 0
    data_files: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    cumulative_gb: float = 0.0
    max_size_gb: float = 0.0
    error: str = ""
    files: list[FileResult] = field(default_factory=list)
    notes: str = ""


def load_config(config_path: str) -> list[SourceConfig]:
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sources = []
    for s in data.get("sources", []):
        sources.append(SourceConfig(
            name=s["name"],
            domain=s.get("domain", ""),
            repo_id=s["repo_id"],
            max_size_gb=float(s.get("max_size_gb", 0)),
            include_patterns=list(s.get("include_patterns", [])),
            exclude_patterns=list(s.get("exclude_patterns", [])),
            notes=s.get("notes", ""),
        ))
    return sources


def matches_patterns(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def get_data_files(api: HfApi, repo_id: str, cfg: SourceConfig) -> list[tuple[str, int]]:
    """返回 (path, size_bytes) 列表，按路径排序。"""
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    all_sibs = info.siblings or []
    results = []
    for s in all_sibs:
        path = s.rfilename
        size = s.size or 0
        if not matches_patterns(path, cfg.include_patterns):
            continue
        if cfg.exclude_patterns and matches_patterns(path, cfg.exclude_patterns):
            continue
        if path.endswith(('.gitattributes', '.md', '.py', '.flake8', '.gitignore', '.png', '.svg', '.pdf', '.json', '.txt')) and not matches_patterns(path, cfg.include_patterns):
            continue
        results.append((path, size))
    results.sort(key=lambda x: x[0])
    return results


def format_size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def download_source(api: HfApi, cfg: SourceConfig, data_root: Path,
                    dry_run: bool = False, test_mode: bool = False,
                    max_size_override: float | None = None) -> SourceResult:
    result = SourceResult(
        name=cfg.name, repo_id=cfg.repo_id, domain=cfg.domain,
        max_size_gb=max_size_override if max_size_override is not None else cfg.max_size_gb,
        notes=cfg.notes,
    )

    try:
        files = get_data_files(api, cfg.repo_id, cfg)
    except GatedRepoError as e:
        result.status = "failed"
        result.error = f"GATED — 需在 HF 页面申请访问权限: https://huggingface.co/datasets/{cfg.repo_id}"
        return result
    except RepositoryNotFoundError as e:
        result.status = "failed"
        result.error = f"NOT FOUND — 仓库不存在或已被移除"
        return result
    except Exception as e:
        result.status = "failed"
        result.error = f"META ERR — {str(e)[:200]}"
        return result

    result.total_files = len(files)
    if not files:
        result.status = "empty"
        result.error = "NO FILES — 无匹配数据文件（检查 include_patterns）"
        return result

    if dry_run:
        result.status = "ok"
        result.data_files = len(files)
        total = sum(s for _, s in files)
        result.cumulative_gb = total / 1e9
        for path, size in files:
            result.files.append(FileResult(path=path, size_bytes=size, status="dry_run"))
        print(f"  [DRY] {cfg.name}: {len(files)} files, {format_size(total)}")
        for path, size in files[:5]:
            print(f"        {path}  ({format_size(size)})")
        if len(files) > 5:
            print(f"        ... ({len(files) - 5} more)")
        return result

    result.data_files = len(files)
    local_dir = data_root / cfg.name
    local_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = result.max_size_gb * 1e9 if result.max_size_gb > 0 else float("inf")
    cumulative = 0
    limit_reached = False

    for i, (path, expected_size) in enumerate(files):
        if test_mode and i >= 1:
            break

        if cumulative >= max_bytes:
            limit_reached = True
            result.files.append(FileResult(path=path, size_bytes=expected_size,
                                           status="excluded",
                                           error=f"max_size_gb={result.max_size_gb} reached"))
            continue

        local_path = local_dir / path
        fr = FileResult(path=path, size_bytes=expected_size)

        if local_path.exists() and local_path.stat().st_size == expected_size:
            fr.status = "skipped"
            fr.local_path = str(local_path)
            result.skipped_files += 1
            cumulative += expected_size
        else:
            try:
                downloaded = hf_hub_download(
                    repo_id=cfg.repo_id,
                    repo_type="dataset",
                    filename=path,
                    local_dir=str(local_dir),
                )
                actual = Path(downloaded).stat().st_size
                fr.status = "downloaded"
                fr.local_path = str(downloaded)
                result.downloaded_files += 1
                cumulative += actual
            except Exception as e:
                fr.status = "failed"
                fr.error = str(e)[:200]
                result.failed_files += 1

        result.files.append(fr)

        flag = {"downloaded": "OK", "skipped": "SKIP", "failed": "FAIL",
                "excluded": "SKIP"}[fr.status]
        print(f"  [{flag}] {path}  ({format_size(expected_size)})  "
              f"cum={cumulative / 1e9:.2f} GB / {result.max_size_gb} GB")

    result.cumulative_gb = cumulative / 1e9
    if result.failed_files > 0 and result.downloaded_files == 0:
        result.status = "failed"
    elif limit_reached:
        result.status = "partial"
    else:
        result.status = "ok"

    return result


def write_report(results: list[SourceResult], report_path: Path):
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hf_endpoint": HF_ENDPOINT,
        "sources": [
            {
                "name": r.name, "repo_id": r.repo_id, "domain": r.domain,
                "status": r.status, "max_size_gb": r.max_size_gb,
                "total_files": r.total_files, "data_files": r.data_files,
                "downloaded": r.downloaded_files, "skipped": r.skipped_files,
                "failed": r.failed_files, "cumulative_gb": round(r.cumulative_gb, 3),
                "error": r.error, "notes": r.notes,
            }
            for r in results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="YuFormer 预训练数据下载器")
    parser.add_argument("--config", default=None,
                        help="配置文件路径（默认同目录下 data_sources.yaml）")
    parser.add_argument("--data-root", default="./data",
                        help="数据根目录（默认 ./data）")
    parser.add_argument("--all", action="store_true",
                        help="下载全部数据源")
    parser.add_argument("--source", default=None,
                        help="只下载指定名称的数据源")
    parser.add_argument("--max-size", type=float, default=None,
                        help="覆盖 max_size_gb（0=无限制）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出文件与大小，不下载")
    parser.add_argument("--test", action="store_true",
                        help="每源只下载第一个文件（验证可访问性）")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = args.config or str(script_dir.parent / "configs" / "data_sources.yaml")

    if not Path(config_path).exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    sources = load_config(config_path)

    if args.source:
        sources = [s for s in sources if s.name == args.source]
        if not sources:
            print(f"ERROR: source '{args.source}' not in config. "
                  f"Available: {[s.name for s in load_config(config_path)]}")
            sys.exit(1)
    elif not args.all:
        parser.print_help()
        sys.exit(0)

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=HF_ENDPOINT)

    mode = "DRY-RUN" if args.dry_run else "TEST" if args.test else "DOWNLOAD"
    print(f"\n{'=' * 70}")
    print(f"  YuFormer Data Downloader [{mode}]")
    print(f"  HF endpoint : {HF_ENDPOINT}")
    print(f"  Data root   : {data_root}")
    print(f"  Sources     : {len(sources)}")
    print(f"  Max size    : {args.max_size or 'per-config'} GB")
    print(f"{'=' * 70}\n")

    results = []
    for cfg in sources:
        print(f"\n[{cfg.domain}] {cfg.name}  ({cfg.repo_id})")
        print(f"  max_size_gb={args.max_size if args.max_size is not None else cfg.max_size_gb}, "
              f"notes: {cfg.notes}")

        r = download_source(api, cfg, data_root,
                            dry_run=args.dry_run,
                            test_mode=args.test,
                            max_size_override=args.max_size)
        results.append(r)

        status_icon = {"ok": "✓", "partial": "◐", "failed": "✗",
                       "empty": "○"}.get(r.status, "?")
        print(f"  {status_icon} {r.status.upper()} — "
              f"files={r.data_files}, dl={r.downloaded_files}, "
              f"skip={r.skipped_files}, fail={r.failed_files}, "
              f"cum={r.cumulative_gb:.2f} GB")
        if r.error:
            print(f"  ! {r.error}")

        time.sleep(1)

    report_path = data_root / "download_report.json"
    write_report(results, report_path)

    print(f"\n{'=' * 70}")
    print(f"  Report saved: {report_path}")
    ok = sum(1 for r in results if r.status in ("ok", "partial"))
    fail = sum(1 for r in results if r.status == "failed")
    empty = sum(1 for r in results if r.status == "empty")
    total_gb = sum(r.cumulative_gb for r in results)
    print(f"  Summary: {ok} ok/partial, {fail} failed, {empty} empty, "
          f"total {total_gb:.2f} GB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
