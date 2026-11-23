"""Legacy single-host orchestration with release gates enforced fail-closed.

设计（宪法 4/6：条件驱动 + fail-fast + 幂等）：
  - 每阶段 = 启动条件（产物+summary 校验）+ 验收条件（shard_errors=0）
  - 所有底层 driver 幂等——编排器重复启动无害（E-7）
  - 模式：strict（默认，正式路径）/ fast（仅诊断，不会绕过 release gate）
  - 状态落盘 data/pipeline_state.json；终态写 DATA_READY.json（实验链的点火信号）
  - 阶段失败 → 记录并停在该阶段（不盲目越权重试；driver 内部分片级容错已自愈）
用法：nohup python scripts/pipeline_orchestrator.py [--mode fast] &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = (ROOT / ".." / "data" / "processed").resolve()
DATA = OUT.parent
PY = sys.executable
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": "/root", "HF_HUB_OFFLINE": "1"}
MIX = "mix_m3_e1_100b"
TOKENIZER = (ROOT.parent / "tokenizer" / "tokenizers" / "yf-64k-bpe").resolve()
PRETRAIN = ROOT.parent / "pretrain"
VAL_DIR = OUT / "val_sets"
STREAMS = DATA / "token_streams" / f"{MIX}-20261019"
TOKEN_STREAM_AUDIT = OUT / "audits" / f"formal_token_stream_release_{MIX}_20261019.json"
STATE_PATH = DATA / "pipeline_state.json"
RELEASE_SPEC = ROOT / "configs" / "releases" / f"{MIX}.yaml"
RELEASE_MANIFEST = OUT / "releases" / f"{MIX}-20261019.release.json"
TOKENIZATION_DEFERRED_GATES = {"E1", "E2", "E4", "G1", "G2"}


def log(msg: str):
    line = f"[{time.strftime('%F %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(DATA / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def state_load() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"phase": "init", "history": []}


def state_save(phase: str, detail: str = ""):
    s = state_load()
    s["history"].append({"ts": time.strftime("%F %H:%M:%S"), "phase": s["phase"], "event": detail})
    s["phase"] = phase
    STATE_PATH.write_text(json.dumps(s, ensure_ascii=False, sort_keys=True, indent=1))


def gs_errors(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        value = payload.get("shard_errors")
        if value is None:
            value = payload.get("task_errors")
        return int(value or 0)
    except Exception:
        return None


def run(cmd: list[str], logfile: str) -> bool:
    log(f"launch: {' '.join(cmd[-6:])} -> {logfile}")
    with open(DATA / logfile, "a") as lf:
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    ok = p.returncode == 0
    log(f"exit={p.returncode} ({'ok' if ok else 'FAIL'}) {logfile}")
    return ok


def token_stream_audit_valid(path: Path, dataset_id: str, snapshot_id: str) -> bool:
    """Accept a formal audit only while every audited stream remains unchanged."""
    try:
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "PASS"
            or payload.get("dataset_id") != dataset_id
            or payload.get("snapshot_id") != snapshot_id
            or payload.get("val_max_tokens_per_domain") != 20_000_000
            or payload.get("roster_exact") is not True
        ):
            return False
        rows = payload.get("streams") or []
        if not rows or any(row.get("status") != "PASS" for row in rows):
            return False
        audit_mtime = path.stat().st_mtime_ns
        for row in rows:
            name = row["name"]
            binary = STREAMS / f"{name}.tokens.bin"
            meta_path = STREAMS / f"{name}.meta.json"
            if not binary.is_file() or not meta_path.is_file():
                return False
            meta = json.loads(meta_path.read_text())
            if (
                binary.stat().st_size != int(row.get("actual_bytes", -1))
                or meta.get("sha256") != row.get("actual_sha256")
                or max(binary.stat().st_mtime_ns, meta_path.stat().st_mtime_ns) > audit_mtime
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def s4_done() -> bool:
    global_path = OUT / "stage_s4_global_summary.json"
    if gs_errors(global_path) != 0 or not global_path.is_file():
        return False
    try:
        summary = json.loads(global_path.read_text())
        score_shards = int(summary.get("score_shards") or 0)
        quantile_shards = int(summary.get("quantile_shards") or 0)
        total_rows = int(summary.get("total_rows") or 0)
        if (
            score_shards <= 0
            or quantile_shards != score_shards
            or int(summary.get("quantile_rows") or 0) != total_rows
        ):
            return False
        covered = (
            sum(1 for _ in (OUT / "s4_scored").rglob("*.summary.json"))
            if (OUT / "s4_scored").is_dir() else 0
        )
        if covered != score_shards:
            return False
        table_root = OUT / "s4_quantiles" / "tables"
        expected = {
            "": "composite",
            ".rule": "rule_composite",
            ".hybrid": "hybrid_composite",
            ".hybrid2": "hybrid_v2",
        }
        for suffix, score_field in expected.items():
            path = table_root / f"_build{suffix}.summary.json"
            if not path.is_file():
                return False
            build = json.loads(path.read_text())
            if (
                build.get("score_field") != score_field
                or int(build.get("pair_shards") or 0) != score_shards
                or int(build.get("total_rows") or 0) != total_rows
            ):
                return False
        return True
    except Exception:
        return False


def ndv2_domain_done(domain: str) -> bool:
    p = OUT / "s3_decisions" / f"{domain}.summary.json"
    if not p.is_file():
        return False
    try:
        summary = json.loads(p.read_text())
        decision_ok = (
            summary.get("policy") == "ndv2_minhash128_bbit4"
            and not summary.get("errors")
            and (summary.get("decision_contract") or {}).get("method")
            == "edge_dominance_external"
        )
        audit_path = OUT / "s3_v2" / f"{domain}_minhash_precision_audit.json"
        dropped = int(summary.get("dropped") or 0)
        lookup_root = OUT / "s3_lookup" / domain
        ids_path = lookup_root / "lookup.ids.bin"
        codes_path = lookup_root / "lookup.codes.bin"
        lookup_ok = (
            ids_path.is_file()
            and codes_path.is_file()
            and ids_path.stat().st_size == dropped * 16
            and codes_path.stat().st_size == dropped
        )
        if not decision_ok or not lookup_ok or not audit_path.is_file():
            return False
        audit = json.loads(audit_path.read_text())
        return (
            audit.get("passed") is True
            and not audit.get("errors")
            and (audit.get("audit_contract") or {}).get("edges_fingerprint")
            == summary.get("edges_fingerprint")
        )
    except Exception:
        return False

def s5_done() -> bool:
    ok = gs_errors(OUT / "stage_s5_global_summary.json")
    if ok != 0:
        return False
    try:
        summary = json.loads((OUT / "stage_s5_global_summary.json").read_text())
        coverage = summary.get("coverage") or {}
        expected = int(coverage.get("expected") or 0)
        if (
            expected <= 0
            or int(coverage.get("complete") or 0) != expected
            or int(coverage.get("missing_summary") or 0) != 0
            or int(coverage.get("missing_output") or 0) != 0
        ):
            return False
        contract = summary.get("input_contract") or {}
        if contract.get("schema_version") != 2:
            return False
        for relative, expected in (contract.get("control_sha256") or {}).items():
            path = OUT / relative
            if not path.is_file() or _file_sha256(path) != expected:
                return False
        for domain, expected in (contract.get("lookup_fingerprint") or {}).items():
            lookup = OUT / "s3_lookup" / domain
            actual = []
            for name in ("lookup.ids.bin", "lookup.codes.bin"):
                path = lookup / name
                if not path.is_file():
                    actual.extend([-1, -1])
                else:
                    stat = path.stat()
                    actual.extend([stat.st_size, stat.st_mtime_ns])
            if actual != expected:
                return False
        return (
            _file_sha256(TOKENIZER / "tokenizer.json") == contract.get("tokenizer_sha256")
            and _file_sha256(ROOT / "configs" / "cleaning_config.yaml")
            == contract.get("context_config_sha256")
        )
    except Exception:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def s6_done() -> bool:
    man = OUT / "mix_order" / f"{MIX}.manifest.json"
    mem = OUT / "mix_members" / f"{MIX}.parquet"
    if not (man.is_file() and mem.is_file()):
        return False
    try:
        payload = json.loads(man.read_text())
        return bool(payload.get("upstream_snapshot_sha256")) and isinstance(
            payload.get("upstream_snapshot"), dict
        )
    except Exception:
        return False


def refresh_release_manifest() -> dict | None:
    RELEASE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            PY,
            str(ROOT / "scripts" / "release_gate.py"),
            "--spec",
            str(RELEASE_SPEC),
            "--out",
            str(RELEASE_MANIFEST),
            "--hash-blocked-artifacts",
        ],
        cwd=ROOT.parent,
        env=ENV,
    )
    if completed.returncode not in (0, 2) or not RELEASE_MANIFEST.is_file():
        return None
    return json.loads(RELEASE_MANIFEST.read_text())


def candidate_ready_for_tokenization(release: dict) -> bool:
    """Allow EOD materialization once every machine data-integrity gate is green.

    E4 is produced by tokenization itself. E1/E2 and G1/G2 are review/sign-off
    gates that remain fail-closed for DATA_READY, but do not alter token bytes.
    """
    return bool(release.get("snapshot_id")) and set(
        release.get("failed_gates") or []
    ) <= TOKENIZATION_DEFERRED_GATES


def stream_meta_valid(meta_path: Path, snapshot_id: str) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        boundary = meta.get("document_boundary") or {}
        return (
            meta.get("data_snapshot_id") == snapshot_id
            and boundary.get("policy") == "append_eod"
            and boundary.get("count") == meta.get("n_docs")
        )
    except Exception:
        return False


def streams_done() -> bool:
    if not RELEASE_MANIFEST.is_file():
        return False
    release = json.loads(RELEASE_MANIFEST.read_text())
    snapshot_id = release.get("snapshot_id")
    if not snapshot_id:
        return False
    manifest_path = OUT / "mix_order" / f"{MIX}.manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    order_tables = manifest.get("order_tables") or []
    if not order_tables:
        return False
    stages = [Path(name).stem.removeprefix(f"{MIX}_") for name in order_tables]
    for st in stages:
        meta_path = STREAMS / f"{MIX}_{st}.meta.json"
        if not stream_meta_valid(meta_path, snapshot_id):
            return False
    val_pars = list((OUT / "val_sets").glob("*.parquet")) if (OUT / "val_sets").is_dir() else []
    for p in val_pars:
        meta_path = STREAMS / f"val_{p.stem}.meta.json"
        if not stream_meta_valid(meta_path, snapshot_id):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fast", "strict"], default="strict")
    ap.add_argument("--poll-sec", type=int, default=300)
    args = ap.parse_args()
    log(f"orchestrator start mode={args.mode}")
    STREAMS.mkdir(parents=True, exist_ok=True)

    while True:
        st = state_load()["phase"]
        # ---- 阶段 1：等待/补跑 S4（已在跑则纯等待）----
        # PAUSE_S4 串行调度（2026-10-01）：S4 与 ndv2 verify 双 IO 大户互抢是双输
        # （fetch_b 随机点查被顺序大 IO 挤压 5×）——ndv2 code 优先独占，完成后
        # 自动解除暂停恢复 S4（编排器自管理，无需人工删标志）。
        pause = (DATA / "PAUSE_S4").is_file()
        if pause:
            if ndv2_domain_done("code"):
                (DATA / "PAUSE_S4").unlink(missing_ok=True)
                log("ndv2 code 已完成——解除 S4 暂停，恢复正常流水")
            elif not any_stage_proc("ndv2"):
                log("PAUSE_S4 生效且 ndv2 未在跑——启动 ndv2（code 独占优先）")
                run([PY, "-m", "yuformer_data.scale.driver", "--stage", "ndv2",
                     "--out-root", str(OUT), "--workers", "48"], "ndv2_orch.log")
            time.sleep(args.poll_sec)
            continue
        if not s4_done():
            if st != "wait_s4":
                state_save("wait_s4", "S4 in flight (external) or pending")
                log("phase=wait_s4：等待 S4 全池打分 + 四套分位表")
            if not any_stage_proc("s4"):
                log("S4 driver 不在运行且未完成——启动它")
                run([PY, "-m", "yuformer_data.scale.driver", "--stage", "s4", "--workers", "160",
                     "--data-root", str(DATA / "pretrain-raw"), "--out-root", str(OUT)], "s4_orch.log")
            time.sleep(args.poll_sec)
            continue
        if st == "wait_s4":
            state_save("s4_done", "S4 全池完成")

        # ---- 阶段 2：ndv2（fast 模式全豁免 E-17/E-31——web FineWeb 先例、code 实测
        # 4-9 天不阻塞交付；strict 模式等两域 decide+lookup 完成）----
        if args.mode == "strict":
            need_ndv2_run = not ndv2_domain_done("code") or not ndv2_domain_done("web")
            if need_ndv2_run:
                if st != "wait_ndv2":
                    state_save("wait_ndv2", f"code_done={ndv2_domain_done('code')} web_done={ndv2_domain_done('web')}")
                    log("phase=wait_ndv2：strict 模式等待 ndv2 全量决定")
                if not any_stage_proc("ndv2"):
                    run([PY, "-m", "yuformer_data.scale.driver", "--stage", "ndv2",
                         "--out-root", str(OUT), "--workers", "48"], "ndv2_orch.log")
                time.sleep(args.poll_sec)
                continue
        if st in ("wait_ndv2", "s4_done"):
            state_save("ndv2_done", f"ndv2 按 mode={args.mode} 处理完毕（fast=豁免）")

        # ---- 阶段 3：S5 ----
        if not s5_done():
            if st != "run_s5":
                state_save("run_s5", "前置全绿，启动 S5 全量计长")
            allow = []
            if not ndv2_domain_done("web"):
                allow.append("--allow-e17-web")
            if not ndv2_domain_done("code"):
                allow.append("--allow-e17-code")
            if not any_stage_proc("s5"):
                run([PY, "-m", "yuformer_data.scale.driver", "--stage", "s5", "--workers", "160",
                     "--data-root", str(DATA / "pretrain-raw"), "--out-root", str(OUT), "--tokenizer", str(TOKENIZER)] + allow,
                    "s5_orch.log")
            time.sleep(args.poll_sec)
            continue
        if st == "run_s5":
            state_save("s5_done", "S5 完成 shard_errors=0")

        # ---- 阶段 4：S6 物化 ----
        if not s6_done():
            if st != "run_s6":
                state_save("run_s6", "S5 done，启动 S6 物化（hybrid_v2 选取 + goldrule 子集）")
            if not any_proc("stage s6"):
                run([PY, "-m", "yuformer_data.scale.driver", "--stage", "s6", "--mix-config", str(ROOT / "configs" / f"{MIX}.yaml"),
                     "--workers", "64", "--data-root", str(DATA / "pretrain-raw"), "--out-root", str(OUT)], "s6_orch.log")
            time.sleep(args.poll_sec)
            continue
        if st == "run_s6":
            state_save("s6_done", "S6 manifest+members 落盘")

        release = refresh_release_manifest()
        if release is None:
            log("release manifest 生成失败——停止")
            return 2
        if not candidate_ready_for_tokenization(release) and not streams_done():
            log(
                "tokenize 前置门禁未通过；仅允许剩余 E4 与人工质量/治理签收门时物化 token 流。"
                f" failed_gates={release.get('failed_gates')}"
            )
            return 2

        # ---- 阶段 5：tokenize（pretrain 侧；manifest 声明顺序 + per-domain val）----
        if not streams_done():
            if st != "run_tokenize":
                state_save("run_tokenize", "S6 done，按 manifest 物化 token 流 + val")
            s6_manifest = json.loads(
                (OUT / "mix_order" / f"{MIX}.manifest.json").read_text()
            )
            order_tables = s6_manifest.get("order_tables") or []
            for order_name in order_tables:
                stage = Path(order_name).stem.removeprefix(f"{MIX}_")
                if stream_meta_valid(
                    STREAMS / f"{MIX}_{stage}.meta.json", release["snapshot_id"]
                ):
                    continue
                ok = subprocess.run(
                    [PY, str(PRETRAIN / "scripts" / "materialize_tokens.py"),
                     "--order", str(OUT / "mix_order" / order_name),
                     "--s3-root", str(OUT / "s3"), "--tokenizer", str(TOKENIZER),
                     "--out-dir", str(STREAMS), "--name", f"{MIX}_{stage}",
                     "--data-snapshot", str(RELEASE_MANIFEST), "--workers", "64"],
                    cwd=PRETRAIN, env={**ENV, "PYTHONPATH": str(PRETRAIN / "src")},
                ).returncode == 0
                if not ok:
                    log(f"tokenize {stage} FAILED——停止（检查 materialize 日志）")
                    return 1
            val_done = all(
                stream_meta_valid(
                    STREAMS / f"val_{p.stem}.meta.json", release["snapshot_id"]
                )
                for p in VAL_DIR.glob("*.parquet")
            )
            if not val_done and VAL_DIR.is_dir():
                subprocess.run(
                    [PY, str(PRETRAIN / "scripts" / "materialize_tokens.py"),
                     "--order", str(OUT / "mix_order" / f"{MIX}_stage2.parquet"),
                     "--s3-root", str(OUT / "s3"), "--tokenizer", str(TOKENIZER),
                     "--out-dir", str(STREAMS), "--name", f"{MIX}_stage2",
                     "--val-dir", str(VAL_DIR), "--val-max-tokens", "20000000", "--val-only",
                     "--data-snapshot", str(RELEASE_MANIFEST), "--workers", "64"],  # E-60
                    cwd=PRETRAIN, env={**ENV, "PYTHONPATH": str(PRETRAIN / "src")},
                )
            time.sleep(60)
            continue
        if st == "run_tokenize":
            state_save("tokenize_done", "manifest 顺序表与 validation token 流就绪")

        # ---- 终态 ----
        release = refresh_release_manifest()
        if not release or release.get("status") != "PASS" or not release.get("dataset_id"):
            log(f"token 流完成但正式 release 未 PASS——不写 DATA_READY；failed={release and release.get('failed_gates')}")
            return 2

        if not token_stream_audit_valid(
            TOKEN_STREAM_AUDIT, release["dataset_id"], release["snapshot_id"]
        ):
            audit_ok = subprocess.run(
                [PY, str(PRETRAIN / "scripts" / "audit_token_stream_release.py"),
                 "--stream-dir", str(STREAMS), "--name", f"{MIX}_stage2",
                 "--order", str(OUT / "mix_order" / f"{MIX}_stage2.parquet"),
                 "--val-dir", str(VAL_DIR), "--val-max-tokens", "20000000",
                 "--snapshot", str(RELEASE_MANIFEST),
                 "--s6-summary", str(OUT / "stage_s6_global_summary.json"),
                 "--tokenizer", str(TOKENIZER), "--out", str(TOKEN_STREAM_AUDIT)],
                cwd=ROOT.parent,
                env={**ENV, "PYTHONPATH": str(PRETRAIN / "src")},
            ).returncode == 0
            if not audit_ok or not token_stream_audit_valid(
                TOKEN_STREAM_AUDIT, release["dataset_id"], release["snapshot_id"]
            ):
                log("正式 token-stream 全量审计失败——不写 DATA_READY")
                return 3

        ready = DATA / "DATA_READY.json"
        ready_tmp = ready.with_name(f".{ready.name}.tmp")
        ready_tmp.write_text(json.dumps({
            "ts": time.strftime("%F %H:%M:%S"), "mix": MIX, "streams_dir": str(STREAMS),
            "dataset_id": release["dataset_id"], "snapshot_id": release["snapshot_id"],
            "token_stream_audit": str(TOKEN_STREAM_AUDIT),
            "token_stream_audit_sha256": hashlib.sha256(TOKEN_STREAM_AUDIT.read_bytes()).hexdigest(),
            "note": "release gate 与正式 EOD token-stream 全量审计全部通过",
        }, ensure_ascii=False, sort_keys=True))
        ready_tmp.replace(ready)
        log("★ DATA_READY——管线全部完成，编排器退出")
        return 0


def any_stage_proc(stage: str) -> bool:
    try:
        pattern = rf"yuformer_data[.]scale[.]driver.*--stage[ =]{re.escape(stage)}(?: |$)"
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=20
        ).stdout
        return bool(out.strip())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
