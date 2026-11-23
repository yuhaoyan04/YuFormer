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
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
import re
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
    enabled: bool = True
    revision: str = ""
    max_size_gb: float = 0
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)  # 直连 URL（非 HF 托管源）
    url_sha256: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class FileResult:
    path: str
    size_bytes: int
    sha256: str = ""
    status: str = ""       # downloaded / skipped / failed / excluded
    local_path: str = ""
    error: str = ""


@dataclass
class SourceResult:
    name: str
    repo_id: str
    domain: str
    requested_revision: str = ""
    resolved_revision: str = ""
    selected_inventory_sha256: str = ""
    selection_limited: bool = False
    selected_files: int = 0
    selected_bytes: int = 0
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
            enabled=bool(s.get("enabled", True)),
            revision=str(s.get("revision") or ""),
            max_size_gb=float(s.get("max_size_gb", 0)),
            include_patterns=list(s.get("include_patterns", [])),
            exclude_patterns=list(s.get("exclude_patterns", [])),
            urls=list(s.get("urls", [])),
            url_sha256=dict(s.get("url_sha256", {})),
            notes=s.get("notes", ""),
        ))
    return sources


def select_sources(
    sources: list[SourceConfig],
    *,
    source_name: str | None,
    include_disabled: bool,
) -> list[SourceConfig]:
    """Apply explicit source selection without silently running disabled inputs."""
    if source_name:
        selected = [source for source in sources if source.name == source_name]
        if not selected:
            raise ValueError(
                f"source {source_name!r} not in config; available: "
                f"{[source.name for source in sources]}"
            )
        if not selected[0].enabled and not include_disabled:
            raise ValueError(
                f"source {source_name!r} is disabled; pass --include-disabled "
                "only for an intentional exploratory run"
            )
        return selected
    return [source for source in sources if source.enabled or include_disabled]


def matches_patterns(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get_data_files(
    api: HfApi, repo_id: str, cfg: SourceConfig
) -> tuple[list[tuple[str, int]], str]:
    """返回 (path, size_bytes) 列表与 HF 解析后 commit。"""
    info = api.repo_info(
        repo_id,
        repo_type="dataset",
        revision=cfg.revision,
        files_metadata=True,
    )
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
    return results, str(info.sha or "")


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
                    max_size_override: float | None = None,
                    allow_floating_revision: bool = False) -> SourceResult:
    result = SourceResult(
        name=cfg.name, repo_id=cfg.repo_id, domain=cfg.domain,
        requested_revision=cfg.revision,
        max_size_gb=max_size_override if max_size_override is not None else cfg.max_size_gb,
        notes=cfg.notes,
    )

    if cfg.urls:
        # 直连 URL 源（非 HF 托管）：curl 断点续传逐一下载
        expected_names = {url.rstrip('/').split('/')[-1] for url in cfg.urls}
        valid_digests = (
            set(cfg.url_sha256) == expected_names
            and all(re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in cfg.url_sha256.values())
        )
        if not valid_digests and not allow_floating_revision:
            result.status = "failed"
            result.error = (
                "UNPINNED DIRECT URLS — every URL filename requires a 64-hex "
                "url_sha256 entry"
            )
            return result
        if valid_digests:
            result.resolved_revision = hashlib.sha256(
                json.dumps(cfg.url_sha256, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            result.selected_inventory_sha256 = result.resolved_revision
        result.data_files = len(cfg.urls)
        result.total_files = len(cfg.urls)
        local_dir = data_root / cfg.name
        local_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        for url in cfg.urls:
            fname = url.rstrip('/').split('/')[-1]
            dest = local_dir / fname
            fr = FileResult(path=fname, size_bytes=0)
            if dest.exists() and dest.stat().st_size > 0:
                actual = file_sha256(dest)
                fr.size_bytes = dest.stat().st_size
                fr.sha256 = actual
                if valid_digests and actual != cfg.url_sha256[fname].lower():
                    fr.status = "failed"
                    fr.error = f"sha256 mismatch: expected {cfg.url_sha256[fname]}, got {actual}"
                    result.failed_files += 1
                else:
                    fr.status = "skipped"
                    result.skipped_files += 1
            elif dry_run:
                fr.status = "dry_run"
            else:
                try:
                    subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "5",
                                    "--retry-delay", "5", "-o", str(dest), url],
                                   check=True, timeout=7200)
                    actual = file_sha256(dest)
                    fr.size_bytes = dest.stat().st_size
                    fr.sha256 = actual
                    if valid_digests and actual != cfg.url_sha256[fname].lower():
                        raise ValueError(
                            f"sha256 mismatch: expected {cfg.url_sha256[fname]}, got {actual}"
                        )
                    fr.status = "downloaded"
                    result.downloaded_files += 1
                except Exception as e:
                    fr.status = "failed"
                    fr.error = str(e)[:200]
                    result.failed_files += 1
            fr.local_path = str(dest)
            result.files.append(fr)
            print(f"  [{fr.status[:4].upper()}] {fname}  {fr.error}", flush=True)
        result.cumulative_gb = sum(
            (local_dir / url.rstrip('/').split('/')[-1]).stat().st_size
            for url in cfg.urls
            if (local_dir / url.rstrip('/').split('/')[-1]).is_file()
        ) / 1e9
        result.selected_files = len(cfg.urls)
        result.selected_bytes = sum(file.size_bytes for file in result.files)
        result.status = "ok" if result.failed_files == 0 else "failed"
        return result

    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", cfg.revision):
        if not allow_floating_revision:
            result.status = "failed"
            result.error = (
                "UNPINNED REVISION — config must contain an immutable 40-64 hex commit; "
                "--allow-floating-revision is exploratory only"
            )
            return result

    try:
        files, result.resolved_revision = get_data_files(api, cfg.repo_id, cfg)
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

    result.data_files = len(files)
    local_dir = data_root / cfg.name
    local_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = result.max_size_gb * 1e9 if result.max_size_gb > 0 else float("inf")

    # 确定性选出本源的下载集合（与串行顺序完全一致）：
    # 按 blake2b 均匀排序后累计到 max_size_gb；test 模式只取第一个。
    if result.max_size_gb > 0:
        # 上限抽样：按 blake2b(源名, 路径) 均匀排序，使截断子集是
        # 全库的确定性均匀样本（跨 CC-MAIN 快照/年份/语言），
        # 而非按路径字典序截断造成的早期快照偏差。
        from hashlib import blake2b

        def order_key(item: tuple[str, int]) -> str:
            path, _ = item
            h = blake2b(digest_size=16,
                        person=cfg.name[:16].encode().ljust(16, b"\0"))
            h.update(path.encode())
            return h.hexdigest()

        files.sort(key=order_key)

    selected: list[tuple[str, int]] = []
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
        selected.append((path, expected_size))
        cumulative += expected_size
    result.selected_inventory_sha256 = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result.selection_limited = limit_reached
    result.selected_files = len(selected)
    result.selected_bytes = cumulative

    if dry_run:
        result.status = "ok"
        result.cumulative_gb = cumulative / 1e9
        for path, size in selected:
            result.files.append(FileResult(path=path, size_bytes=size, status="dry_run"))
        print(f"  [DRY] {cfg.name}: {len(selected)} selected files, {format_size(cumulative)}")
        for path, size in selected[:5]:
            print(f"        {path}  ({format_size(size)})")
        if len(selected) > 5:
            print(f"        ... ({len(selected) - 5} more selected)")
        return result

    # 并发下载已选集合；文件选择逻辑保持确定性，网络并发只影响完成顺序
    from concurrent.futures import ThreadPoolExecutor

    def fetch(item: tuple[str, int]) -> FileResult:
        path, expected_size = item
        local_path = local_dir / path
        fr = FileResult(path=path, size_bytes=expected_size)
        if local_path.exists() and local_path.stat().st_size == expected_size:
            fr.status = "skipped"
            fr.local_path = str(local_path)
            return fr
        try:
            downloaded = hf_hub_download(
                repo_id=cfg.repo_id,
                repo_type="dataset",
                filename=path,
                revision=result.resolved_revision,
                local_dir=str(local_dir),
            )
            fr.status = "downloaded"
            fr.local_path = str(downloaded)
            if expected_size and Path(downloaded).stat().st_size != expected_size:
                fr.status = "failed"
                fr.error = (
                    f"size mismatch: expected {expected_size}, "
                    f"got {Path(downloaded).stat().st_size}"
                )
            return fr
        except Exception as e:
            msg = str(e)
            if "too large" in msg or "hf_xet" in msg:
                # >10GB 单文件（如 arxiv-latex-corpus）走 resolve URL + curl 续传
                try:
                    import subprocess
                    url = (f"{HF_ENDPOINT}/datasets/{cfg.repo_id}"
                           f"/resolve/{result.resolved_revision}/{path}")
                    dest = local_dir / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        ["curl", "-sfL", "-C", "-", "--retry", "5",
                         "--retry-delay", "5", "-o", str(dest), url],
                        check=True, timeout=7200)
                    fr.status = "downloaded"
                    fr.local_path = str(dest)
                    if expected_size and dest.stat().st_size != expected_size:
                        fr.status = "failed"
                        fr.error = (
                            f"size mismatch: expected {expected_size}, got {dest.stat().st_size}"
                        )
                    return fr
                except Exception as e2:
                    fr.status = "failed"
                    fr.error = f"curl fallback: {str(e2)[:150]}"
                    return fr
            fr.status = "failed"
            fr.error = msg[:200]
            return fr

    workers = 1 if test_mode else int(os.environ.get("YUFORMER_DL_WORKERS", "8"))
    done_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fr in pool.map(fetch, selected):
            result.files.append(fr)
            if fr.status == "downloaded":
                result.downloaded_files += 1
            elif fr.status == "skipped":
                result.skipped_files += 1
            else:
                result.failed_files += 1
            done_bytes += fr.size_bytes
            flag = {"downloaded": "OK", "skipped": "SKIP", "failed": "FAIL",
                    "excluded": "SKIP"}[fr.status]
            print(f"  [{flag}] {fr.path}  ({format_size(fr.size_bytes)})  "
                  f"done={done_bytes / 1e9:.2f} GB / {cumulative / 1e9:.2f} GB selected", flush=True)

    result.cumulative_gb = cumulative / 1e9
    if result.failed_files > 0:
        result.status = (
            "partial"
            if result.downloaded_files + result.skipped_files > 0
            else "failed"
        )
    else:
        result.status = "ok"

    return result


def write_report(
    results: list[SourceResult], report_path: Path, *, config_sha256: str
):
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hf_endpoint": HF_ENDPOINT,
        "config_sha256": config_sha256,
        "sources": [
            {
                "name": r.name, "repo_id": r.repo_id, "domain": r.domain,
                "requested_revision": r.requested_revision,
                "resolved_revision": r.resolved_revision,
                "selected_inventory_sha256": r.selected_inventory_sha256,
                "selection_limited": r.selection_limited,
                "selected_files": r.selected_files,
                "selected_bytes": r.selected_bytes,
                "status": r.status, "max_size_gb": r.max_size_gb,
                "total_files": r.total_files, "data_files": r.data_files,
                "downloaded": r.downloaded_files, "skipped": r.skipped_files,
                "failed": r.failed_files, "cumulative_gb": round(r.cumulative_gb, 3),
                "error": r.error, "notes": r.notes,
                "files": [
                    {
                        "path": f.path,
                        "size_bytes": f.size_bytes,
                        "sha256": f.sha256,
                        "status": f.status,
                        "local_path": f.local_path,
                        "error": f.error,
                    }
                    for f in r.files
                ],
            }
            for r in results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)


def main() -> int:
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
    parser.add_argument(
        "--allow-floating-revision",
        action="store_true",
        help="仅探索用：允许未锁定 main/tag；该输出不得进入正式 release",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="仅探索用：允许 --all/--source 运行 enabled=false 的数据源",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = args.config or str(script_dir.parent / "configs" / "data_sources.yaml")

    if not Path(config_path).exists():
        print(f"ERROR: config not found: {config_path}")
        return 1

    all_sources = load_config(config_path)

    if args.source:
        try:
            sources = select_sources(
                all_sources,
                source_name=args.source,
                include_disabled=args.include_disabled,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
    elif not args.all:
        parser.print_help()
        return 0
    else:
        sources = select_sources(
            all_sources,
            source_name=None,
            include_disabled=args.include_disabled,
        )

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
                            max_size_override=args.max_size,
                            allow_floating_revision=args.allow_floating_revision)
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

    report_name = "download_report_dryrun.json" if args.dry_run else "download_report.json"
    report_path = data_root / report_name
    write_report(results, report_path, config_sha256=file_sha256(Path(config_path)))

    print(f"\n{'=' * 70}")
    print(f"  Report saved: {report_path}")
    ok = sum(1 for r in results if r.status == "ok")
    partial = sum(1 for r in results if r.status == "partial")
    fail = sum(1 for r in results if r.status == "failed")
    empty = sum(1 for r in results if r.status == "empty")
    total_gb = sum(r.cumulative_gb for r in results)
    print(f"  Summary: {ok} ok, {partial} partial, {fail} failed, {empty} empty, "
          f"total {total_gb:.2f} GB")
    print(f"{'=' * 70}")
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
