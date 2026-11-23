import importlib.util
from pathlib import Path
import subprocess
import sys
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_pretrain_data.py"
SPEC = importlib.util.spec_from_file_location("download_pretrain_data", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = download
SPEC.loader.exec_module(download)


class _Info:
    sha = "b" * 40
    siblings = []


class _Api:
    def __init__(self):
        self.revision = None

    def repo_info(self, *args, **kwargs):
        self.revision = kwargs["revision"]
        return _Info()


class _Sibling:
    def __init__(self, path, size):
        self.rfilename = path
        self.size = size


class _InventoryApi(_Api):
    def repo_info(self, *args, **kwargs):
        self.revision = kwargs["revision"]
        info = _Info()
        info.siblings = [_Sibling(f"data/{index}.parquet", 10) for index in range(3)]
        return info


def test_downloader_rejects_floating_revision_before_network(tmp_path):
    cfg = download.SourceConfig(name="src", domain="web", repo_id="org/data")
    result = download.download_source(_Api(), cfg, tmp_path, dry_run=True)
    assert result.status == "failed"
    assert "UNPINNED REVISION" in result.error


def test_downloader_resolves_pinned_commit(tmp_path):
    api = _Api()
    cfg = download.SourceConfig(
        name="src", domain="web", repo_id="org/data", revision="a" * 40
    )
    result = download.download_source(api, cfg, tmp_path, dry_run=True)
    assert api.revision == "a" * 40
    assert result.resolved_revision == "b" * 40


def test_configured_size_cap_is_a_complete_selection_not_partial(tmp_path):
    cfg = download.SourceConfig(
        name="src",
        domain="web",
        repo_id="org/data",
        revision="a" * 40,
        max_size_gb=15 / 1e9,
    )
    result = download.download_source(_InventoryApi(), cfg, tmp_path, dry_run=True)
    assert result.status == "ok"
    assert result.selection_limited is True
    assert result.selected_files == 2
    assert result.selected_bytes == 20


def test_downloader_rejects_unhashed_direct_urls(tmp_path):
    cfg = download.SourceConfig(
        name="direct",
        domain="books",
        repo_id="direct-urls",
        urls=["https://example.test/book.json.gz"],
    )
    result = download.download_source(_Api(), cfg, tmp_path, dry_run=True)
    assert result.status == "failed"
    assert "UNPINNED DIRECT URLS" in result.error


def test_direct_url_records_verified_size_and_digest(tmp_path):
    payload = tmp_path / "direct" / "book.json.gz"
    payload.parent.mkdir()
    payload.write_bytes(b"book-content")
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    cfg = download.SourceConfig(
        name="direct",
        domain="books",
        repo_id="direct-urls",
        urls=["https://example.test/book.json.gz"],
        url_sha256={"book.json.gz": digest},
    )
    result = download.download_source(_Api(), cfg, tmp_path, dry_run=True)
    assert result.status == "ok"
    assert result.files[0].size_bytes == len(b"book-content")
    assert result.files[0].sha256 == digest


def test_disabled_sources_are_excluded_unless_explicitly_overridden(tmp_path):
    config = tmp_path / "sources.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {"name": "on", "domain": "web", "repo_id": "org/on"},
                    {
                        "name": "off",
                        "domain": "web",
                        "repo_id": "org/off",
                        "enabled": False,
                    },
                ]
            }
        )
    )
    sources = download.load_config(config)
    assert [source.name for source in download.select_sources(
        sources, source_name=None, include_disabled=False
    )] == ["on"]
    try:
        download.select_sources(sources, source_name="off", include_disabled=False)
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("disabled source was selected without an override")
    assert [source.name for source in download.select_sources(
        sources, source_name="off", include_disabled=True
    )] == ["off"]


def test_cli_returns_nonzero_when_any_selected_source_fails(tmp_path):
    config = tmp_path / "sources.yaml"
    config.write_text(
        yaml.safe_dump(
            {"sources": [{"name": "unpinned", "domain": "web", "repo_id": "org/data"}]}
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--data-root",
            str(tmp_path / "raw"),
            "--all",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "UNPINNED REVISION" in completed.stdout
