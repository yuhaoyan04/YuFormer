"""quick_eval：pretrain checkpoint 的轻量下游评测（experiments/_common，2026-10-08）。

用途：训练中途 checkpoint 的能力曲线（GSM8K / MMLU / HumanEval 快速子集）。
不下载任何数据（HF_HUB_OFFLINE=1）；数据一律经 --data-<name> 传本地 JSONL。

契约
====
- checkpoint 格式（pretrain/src/yuformer_pretrain/checkpoint.py）：
  <out_dir>/checkpoints/ckpt-XXXXXXXX/{model.pt, cursor.json, ...} + latest.json。
  model.pt = FSDP full state_dict（干净 FQN 键，torch.save，weights_only 可读）；
  cursor.json 含 run_name；模型 config 不随 ckpt 存，经 manifest 找回：
  <out_dir>/results/<run_name>/manifest.json → config.model.preset /
  config.model.configs → ModelConfig.from_preset → create_model。
  --ckpt 支持三种指向：ckpt 目录、latest.json 所在 checkpoints 根目录、.pt 文件。
- 文本编码：--tokenizer 传 yuformer tokenizer 目录或 tokenizer.json（用
  yuformer_tokenizer.load_tokenizer）；缺省走字节级回退（UTF-8 byte = token id，
  clamp 到词表内），保证管线确定性可跑，但真实评测应显式给 tokenizer。
- 三个 benchmark（JSONL schema 见 experiments/_common/README.md）：
  gsm8k      生成式，temperature=0（贪心），max_new_tokens=256，
             从输出提取末位数字与 "#### N" 的 gold 比对。
  mmlu       loglikelihood：question 后接 " "+choice 的续写平均 log-prob，argmax。
  humaneval  生成代码 → 子进程执行 test（RLIMIT_CPU 10s / RLIMIT_AS 2GB），pass@1。
- 确定性：每题开始 torch.manual_seed(20261008)；输出 JSON sort_keys；
  per-item 结果写 JSONL（tmp+os.replace）。
- 已知环境适配：本机 torch 2.2.0a0 的版本字符串使 yuformer_model.attention
  的版本解析崩溃（int("0a0")）；导入 yuformer_model 期间临时清洗
  torch.__version__，导入后即恢复（不改 model/ 任何文件）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")

SEED = 20261008
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SRC = Path(os.environ.get("YUFORMER_MODEL_SRC", REPO_ROOT / "model" / "src"))
TOKENIZER_SRC = Path(os.environ.get("YUFORMER_TOKENIZER_SRC", REPO_ROOT / "tokenizer" / "src"))

RLIMIT_CPU_SECS = 10
RLIMIT_AS_BYTES = 2 << 30
SUBPROCESS_TIMEOUT = RLIMIT_CPU_SECS + 10


# ---------------------------------------------------------------------------
# 模型加载（checkpoint.py / manifest 契约）
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    model: Any  # torch.nn.Module（YuFormerModel）
    info: dict[str, Any]


def _import_yuformer_model() -> tuple[Any, Any]:
    """导入 yuformer_model；绕过 torch 版本字符串解析崩溃（见模块 docstring）。"""
    import torch

    orig_ver = torch.__version__
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", orig_ver.split("+")[0])
    if m and m.group(0) != orig_ver:
        torch.__version__ = m.group(0)  # 导入期间临时清洗，随后恢复
    try:
        if str(MODEL_SRC) not in sys.path:
            sys.path.insert(0, str(MODEL_SRC))
        from yuformer_model import ModelConfig, create_model
    finally:
        torch.__version__ = orig_ver
    return ModelConfig, create_model


def _resolve_ckpt(ckpt: Path) -> tuple[Path, Path | None]:
    """把 --ckpt 归一为 (model.pt 路径, cursor.json 路径或 None)。"""
    if ckpt.is_dir():
        if (ckpt / "latest.json").is_file():
            marker = json.loads((ckpt / "latest.json").read_text(encoding="utf-8"))
            ckpt = ckpt / marker["dir"]
        model_pt = ckpt / "model.pt"
        if not model_pt.is_file():
            raise FileNotFoundError(f"{ckpt} 下无 model.pt（不是 checkpoint 目录？）")
        return model_pt, ckpt / "cursor.json"
    if ckpt.is_file() and ckpt.suffix == ".pt":
        return ckpt, ckpt.with_name("cursor.json")
    raise FileNotFoundError(f"--ckpt 必须是 ckpt 目录、checkpoints 根目录或 .pt 文件：{ckpt}")


def _resolve_manifest(model_pt: Path, cursor: Path | None) -> dict[str, Any] | None:
    """从 cursor.json 的 run_name 找回 manifest（<out_dir>/results/<run>/manifest.json）。"""
    if cursor is None or not cursor.is_file():
        return None
    run_name = json.loads(cursor.read_text(encoding="utf-8")).get("run_name")
    if not run_name:
        return None
    # model_pt 位于 <out_dir>/checkpoints/ckpt-XXXXXXXX/ 或任意位置（.pt 直给时尽力而为）
    for cand in (model_pt.parent.parent.parent / "results" / run_name,
                 model_pt.parent.parent / "results" / run_name,
                 model_pt.parent / "results" / run_name):
        if (cand / "manifest.json").is_file():
            return json.loads((cand / "manifest.json").read_text(encoding="utf-8"))
    return None


def load_model(
    ckpt: str | Path,
    *,
    preset: str | None = None,
    model_configs: str | Path | None = None,
    device: str | None = None,
) -> LoadedModel:
    """按 checkpoint + manifest 重建模型并加载权重。

    preset / model_configs 缺省时从 manifest 读取（config.model.*）；
    仍缺省时 model_configs 落到仓库默认 model/configs/model_configs.yaml。
    """
    import torch

    model_pt, cursor = _resolve_ckpt(Path(ckpt))
    manifest = _resolve_manifest(model_pt, cursor)
    ModelConfig, create_model = _import_yuformer_model()

    if preset is None and manifest is not None:
        preset = manifest.get("config", {}).get("model", {}).get("preset")
    if model_configs is None and manifest is not None:
        model_configs = manifest.get("config", {}).get("model", {}).get("configs")
    if preset is None:
        raise SystemExit(
            "无法确定模型 preset：checkpoint 旁无 manifest 且未给 --preset"
            "（.pt 直给时需要 --preset + --model-configs）"
        )
    cfg = ModelConfig.from_preset(str(preset), config_path=model_configs)

    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    model = create_model(cfg)
    model.load_state_dict(state)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(torch.device(device))
    model.eval()

    it = None
    if cursor is not None and cursor.is_file():
        it = json.loads(cursor.read_text(encoding="utf-8")).get("iter")
    info = {
        "ckpt": str(model_pt),
        "run_name": (manifest or {}).get("arm_name"),
        "preset": preset,
        "iter": it,
        "params": model.num_parameters(),
        "params_non_embedding": model.num_parameters(non_embedding=True),
        "device": str(device),
    }
    return LoadedModel(model=model, info=info)


# ---------------------------------------------------------------------------
# 文本编码（tokenizer 可选；缺省字节级回退）
# ---------------------------------------------------------------------------


class ByteCodec:
    """字节级回退编码：id = UTF-8 byte（clamp 到词表内）。仅保证管线可跑。"""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(b, self.vocab_size - 1) for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="ignore")


class TokenizerCodec:
    """yuformer tokenizer（tokenizers 库）封装。"""

    def __init__(self, path: str | Path) -> None:
        if str(TOKENIZER_SRC) not in sys.path:
            sys.path.insert(0, str(TOKENIZER_SRC))
        from yuformer_tokenizer.trainer import load_tokenizer

        self._tok = load_tokenizer(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids])


def get_codec(tokenizer: str | Path | None, vocab_size: int) -> Any:
    return TokenizerCodec(tokenizer) if tokenizer else ByteCodec(vocab_size)


# ---------------------------------------------------------------------------
# 推理原语（贪心生成 + 续写平均 log-prob）
# ---------------------------------------------------------------------------


def generate_greedy(
    model: Any,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    context: int,
) -> list[int]:
    """贪心解码，返回续写 token ids（遇 EOS 或到达上限停）。"""
    import torch

    if not input_ids:
        input_ids = [0]
    ids = list(input_ids[-context:])
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x)[0, -1]
        nxt = int(torch.argmax(logits).item())
        if nxt == eos_token_id:
            break
        ids.append(nxt)
        if len(ids) >= context:
            break
    return ids[len(input_ids[-context:]):]


def continuation_logprob(
    model: Any,
    context_ids: list[int],
    cont_ids: list[int],
    *,
    max_len: int,
) -> float | None:
    """sum log p(cont | context) / len(cont)；cont 为空返回 None。"""
    import torch
    import torch.nn.functional as F

    if not cont_ids:
        return None
    ctx = context_ids[-(max_len - 1):] if max_len > 1 else []
    ids = (ctx + cont_ids)[-max_len:]
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=next(model.parameters()).device)
    y = torch.tensor(ids[1:], dtype=torch.long, device=next(model.parameters()).device)
    logits = model(x)[0]  # (T, V)
    logp = F.log_softmax(logits, dim=-1)
    n_cont = min(len(cont_ids), len(ids) - 1)
    tok_lp = logp[-n_cont:].gather(1, y[-n_cont:].unsqueeze(1)).squeeze(1)
    return float(tok_lp.sum().item()) / n_cont


# ---------------------------------------------------------------------------
# GSM8K（生成式，末位数字比对）
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_last_number(text: str) -> float | None:
    """提取输出中的末位数字（容忍千分位逗号、小数、负号）。"""
    nums = NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_gold(answer: str) -> float | None:
    if "####" not in answer:
        return None
    tail = answer.split("####")[-1].strip().replace(",", "").replace("$", "")
    try:
        return float(tail)
    except ValueError:
        return None


def run_gsm8k(
    model: Any,
    codec: Any,
    items: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    cfg = model.config
    out: list[dict[str, Any]] = []
    import torch

    for it in items:
        torch.manual_seed(seed)
        q = str(it["question"]).strip()
        gold = gsm8k_gold(str(it["answer"]))
        prompt_ids = codec.encode(f"Question: {q}\nAnswer:")
        gen_ids = generate_greedy(
            model, prompt_ids, max_new_tokens=max_new_tokens,
            eos_token_id=cfg.eos_token_id, context=cfg.max_seq_len,
        )
        text = codec.decode(gen_ids)
        extracted = extract_last_number(text)
        correct = (
            gold is not None and extracted is not None
            and abs(extracted - gold) <= max(1e-4, abs(gold) * 1e-6)
        )
        out.append({
            "id": it.get("id"), "question": q, "gold": gold,
            "generated": text, "extracted": extracted, "correct": bool(correct),
        })
    return out


# ---------------------------------------------------------------------------
# MMLU（续写 loglikelihood，argmax；无生成）
# ---------------------------------------------------------------------------


def run_mmlu(
    model: Any,
    codec: Any,
    items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        q = str(it["question"]).strip()
        choices = [str(c) for c in it["choices"]]
        gold_idx = int(it["answer"])
        ctx = codec.encode(q)
        scores = [
            continuation_logprob(model, ctx, codec.encode(" " + c), max_len=model.config.max_seq_len)
            for c in choices
        ]
        valid = [s is not None for s in scores]
        if not any(valid):
            pred_idx = 0
        else:
            pred_idx = max(
                (i for i, ok in enumerate(valid) if ok),
                key=lambda i: scores[i],  # type: ignore[index]
            )
        out.append({
            "id": it.get("id"), "question": q,
            "scores": [round(s, 6) if s is not None else None for s in scores],
            "pred": pred_idx, "gold": gold_idx,
            "correct": pred_idx == gold_idx,
        })
    return out


# ---------------------------------------------------------------------------
# HumanEval（生成 + 子进程沙箱执行 test，pass@1）
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """剥离生成输出里的 markdown 代码围栏，取第一个代码块（去首尾空行）。"""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip("\n") if m else text


def run_generated_test(code: str, test: str, entry_point: str) -> tuple[bool, str]:
    """子进程内执行 code + test 并调用 check(entry_point)；返回 (passed, error)。

    资源限额（CPU 10s / 内存 2GB）直接写进 runner 脚本在子进程 exec 后生效，
    避免 preexec_fn 在 fork 与 exec 之间跑 Python 代码（torch 线程活跃时可死锁）。
    """
    limits = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({RLIMIT_CPU_SECS}, {RLIMIT_CPU_SECS}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({RLIMIT_AS_BYTES}, {RLIMIT_AS_BYTES}))\n"
    )
    runner = limits + code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "runner.py"
        script.write_text(runner, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td, capture_output=True, text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:  # spawn 失败等
            return False, f"spawn_error: {exc}"
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or "").strip().splitlines()
        return False, err[-1] if err else f"exit={proc.returncode}"


def run_humaneval(
    model: Any,
    codec: Any,
    items: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    cfg = model.config
    out: list[dict[str, Any]] = []
    import torch

    for it in items:
        torch.manual_seed(seed)
        prompt = str(it["prompt"])
        gen_ids = generate_greedy(
            model, codec.encode(prompt), max_new_tokens=max_new_tokens,
            eos_token_id=cfg.eos_token_id, context=cfg.max_seq_len,
        )
        completion = strip_code_fences(codec.decode(gen_ids))
        code = prompt + completion
        passed, error = run_generated_test(code, str(it["test"]), str(it["entry_point"]))
        out.append({
            "task_id": it.get("task_id"), "prompt": prompt,
            "completion": completion, "correct": passed, "error": error or None,
        })
    return out


# ---------------------------------------------------------------------------
# 聚合输出
# ---------------------------------------------------------------------------


def atomic_write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def atomic_write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".tmp-", suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(bench: str, rows: Sequence[dict[str, Any]], items_path: Path) -> dict[str, Any]:
    n = len(rows)
    acc = float(sum(1 for r in rows if r["correct"])) / n if n else 0.0
    return {"n": n, "accuracy": round(acc, 6), "per_item_path": str(items_path)}


def run_evals(
    loaded: LoadedModel,
    codec: Any,
    data: dict[str, str],
    *,
    out_json: str | Path,
    max_items: int = 0,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """跑给定的 benchmark 数据并写结果 JSON + per-item JSONL。"""
    model = loaded.model
    out_path = Path(out_json)
    result: dict[str, Any] = {"model": loaded.info, "seed": SEED}
    runners: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "gsm8k": lambda: run_gsm8k(model, codec, read_jsonl(data["gsm8k"])[: max_items or None],
                                   max_new_tokens=max_new_tokens),
        "mmlu": lambda: run_mmlu(model, codec, read_jsonl(data["mmlu"])[: max_items or None]),
        "humaneval": lambda: run_humaneval(model, codec, read_jsonl(data["humaneval"])[: max_items or None],
                                           max_new_tokens=max_new_tokens),
    }
    for bench in sorted(data):
        rows = runners[bench]()
        items_path = out_path.with_name(out_path.stem + f".{bench}.items.jsonl")
        atomic_write_jsonl(items_path, rows)
        result[bench] = summarize(bench, rows, items_path)
    atomic_write_json(out_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="quick_eval：checkpoint 轻量下游评测（experiments/_common）")
    p.add_argument("--ckpt", help="ckpt 目录 / checkpoints 根目录（含 latest.json）/ model.pt")
    p.add_argument("--tokenizer", help="tokenizer 目录或 tokenizer.json（缺省字节级回退）")
    p.add_argument("--data-gsm8k", help="GSM8K 子集 JSONL（{question, answer}）")
    p.add_argument("--data-mmlu", help="MMLU 子集 JSONL（{question, choices, answer}）")
    p.add_argument("--data-humaneval", help="HumanEval 子集 JSONL（{task_id, prompt, test, entry_point}）")
    p.add_argument("--out", default="quick_eval_result.json", help="结果 JSON 路径")
    p.add_argument("--device", default=None, help="cpu / cuda（缺省自动）")
    p.add_argument("--preset", default=None, help="覆盖 manifest 的模型 preset")
    p.add_argument("--model-configs", default=None, help="覆盖 model_configs.yaml 路径")
    p.add_argument("--max-items", type=int, default=0, help="每 benchmark 最多评测条数（0=全部）")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--selftest", action="store_true", help="合成数据自测（随机 tiny 模型，无需真实数据）")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    data = {
        k: v
        for k, v in (("gsm8k", args.data_gsm8k), ("mmlu", args.data_mmlu),
                     ("humaneval", args.data_humaneval))
        if v
    }
    if not args.ckpt or not data:
        raise SystemExit("需要 --ckpt 且至少一个 --data-<name>（或 --selftest）")
    loaded = load_model(args.ckpt, preset=args.preset, model_configs=args.model_configs,
                        device=args.device)
    codec = get_codec(args.tokenizer, loaded.model.config.vocab_size)
    result = run_evals(loaded, codec, data, out_json=args.out,
                       max_items=args.max_items, max_new_tokens=args.max_new_tokens)
    print(json.dumps({k: v for k, v in result.items() if k != "model"},
                     indent=2, sort_keys=True, ensure_ascii=False))
    print(f"模型：{loaded.info['preset']} iter={loaded.info['iter']} params={loaded.info['params']:,} "
          f"→ {args.out}")
    return 0


# ---------------------------------------------------------------------------
# 自测：随机 tiny 模型 + 合成数据，验证管线确定性与契约
# ---------------------------------------------------------------------------

SELFTEST_PRESET = "selftest-tiny"
SELFTEST_PRESET_YAML = f"""presets:
  {SELFTEST_PRESET}:
    vocab_size: 256
    hidden_size: 64
    intermediate_size: 128
    num_layers: 2
    num_heads: 4
    num_kv_heads: 2
    max_seq_len: 256
    window_size: 8
    attention_every: 1
    rope_every: 1
    rope_percent: 1.0
    qk_norm: true
    attention_output_gate: false
    tie_word_embeddings: true
    notes: quick_eval selftest tiny 模型
"""


def _synthetic_mmlu(path: Path, n: int = 5) -> None:
    rows = [
        {
            "id": i,
            "question": f"下列哪个数是 {i} 的平方？",
            "choices": [str(i * i), str(i * i + 1), str(i * i + 2), str(i * i + 3)],
            "answer": 0,
        }
        for i in range(1, n + 1)
    ]
    atomic_write_jsonl(path, rows)


def _synthetic_gsm8k(path: Path, n: int = 5) -> None:
    rows = [
        {
            "id": i,
            "question": f"小明有 {i} 个苹果，又买了 {i} 个，一共有多少个？",
            "answer": f"小明共有 {i} + {i} = {2 * i} 个苹果。\n#### {2 * i}",
        }
        for i in range(1, n + 1)
    ]
    atomic_write_jsonl(path, rows)


def _synthetic_humaneval(path: Path, n: int = 2) -> None:
    rows = [
        {
            "task_id": f"Synth/{i}",
            "prompt": f"def add_{i}(a, b):\n    \"\"\"返回 a + b + {i}\"\"\"\n",
            "test": (
                "def check(candidate):\n"
                f"    assert candidate(1, 2) is not None\n"
            ),
            "entry_point": f"add_{i}",
        }
        for i in range(n)
    ]
    atomic_write_jsonl(path, rows)


def _build_selftest_ckpt(root: Path) -> Path:
    """按 checkpoint.py 磁盘格式造一个 tiny 随机 ckpt + manifest。"""
    import torch

    ModelConfig, create_model = _import_yuformer_model()
    preset_yaml = root / "model_configs_selftest.yaml"
    preset_yaml.write_text(SELFTEST_PRESET_YAML, encoding="utf-8")
    model = create_model(ModelConfig.from_preset(SELFTEST_PRESET, config_path=preset_yaml))
    torch.manual_seed(SEED)
    for p in model.parameters():
        p.data = torch.randn_like(p) * 0.02

    out_dir = root / "out"
    ckpt_dir = out_dir / "checkpoints" / "ckpt-00000002"
    ckpt_dir.mkdir(parents=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    (ckpt_dir / "cursor.json").write_text(
        json.dumps({"iter": 2, "run_name": "selftest-run", "config_sha256": "x"}, sort_keys=True),
        encoding="utf-8")
    atomic_write_json(out_dir / "checkpoints" / "latest.json",
                      {"iter": 2, "config_sha256": "x", "run_name": "selftest-run",
                       "dir": "ckpt-00000002"})
    manifest = {
        "arm_name": "selftest-run",
        "seed": SEED,
        "params": model.num_parameters(),
        "config": {
            "model": {"preset": SELFTEST_PRESET, "configs": str(preset_yaml)},
            "train": {"budget_tokens": 1000},
        },
    }
    atomic_write_json(out_dir / "results" / "selftest-run" / "manifest.json", manifest)
    return out_dir / "checkpoints"


def selftest() -> int:
    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            raise SystemExit(f"selftest 失败: {name} {detail}")

    # 纯函数单测：数字提取
    check("extract_last_number",
          extract_last_number("共 3,600 元，答案是 42.") == 42.0
          and extract_last_number("没有数字") is None
          and extract_last_number("结果是 -3.5") == -3.5)
    check("gsm8k_gold", gsm8k_gold("算式。\n#### 18") == 18.0
          and gsm8k_gold("无标记") is None)
    check("strip_code_fences",
          strip_code_fences("前言\n```python\nx=1\n```\n后语") == "x=1")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt_root = _build_selftest_ckpt(root)
        data_dir = root / "data"
        data_dir.mkdir()
        _synthetic_mmlu(data_dir / "mmlu.jsonl")
        _synthetic_gsm8k(data_dir / "gsm8k.jsonl")
        _synthetic_humaneval(data_dir / "humaneval.jsonl")

        # manifest 路径解析 + 模型重建
        loaded = load_model(ckpt_root, device="cpu")  # checkpoints 根目录（latest.json）
        manifest = json.loads(
            (ckpt_root.parent / "results" / "selftest-run" / "manifest.json").read_text()
        )
        check("load_model via latest.json",
              loaded.info["iter"] == 2 and loaded.info["preset"] == SELFTEST_PRESET
              and loaded.info["params"] == manifest["params"],
              json.dumps(loaded.info, sort_keys=True))
        codec = get_codec(None, loaded.model.config.vocab_size)

        data = {k: str(data_dir / f"{k}.jsonl")
                for k in ("gsm8k", "mmlu", "humaneval")}
        # 随机 tiny 模型几乎不触发 EOS：自测用小生成长度验证管线（速度确定性不受影响的
        # 前提下，真实评测用默认 256）
        small_gen = 16
        r1 = run_evals(loaded, codec, data, out_json=root / "r1.json", max_new_tokens=small_gen)
        r2 = run_evals(loaded, codec, data, out_json=root / "r2.json", max_new_tokens=small_gen)
        strip = lambda r: {k: {kk: vv for kk, vv in v.items() if kk != "per_item_path"}
                           for k, v in r.items() if isinstance(v, dict)}  # noqa: E731
        check("确定性：两遍结果一致", strip(r1) == strip(r2))
        check("确定性：per-item 逐条一致",
              read_jsonl(root / "r1.mmlu.items.jsonl") == read_jsonl(root / "r2.mmlu.items.jsonl")
              and read_jsonl(root / "r1.gsm8k.items.jsonl") == read_jsonl(root / "r2.gsm8k.items.jsonl")
              and read_jsonl(root / "r1.humaneval.items.jsonl") == read_jsonl(root / "r2.humaneval.items.jsonl"))
        check("n 正确", all(r1[b]["n"] == 5 for b in ("gsm8k", "mmlu")) and r1["humaneval"]["n"] == 2)
        check("accuracy 合法",
              all(0.0 <= r1[b]["accuracy"] <= 1.0 for b in ("gsm8k", "mmlu", "humaneval")))

        # per-item 结构
        mmlu_items = read_jsonl(root / "r1.mmlu.items.jsonl")
        check("mmlu per-item",
              len(mmlu_items) == 5 and all("scores" in r and "pred" in r for r in mmlu_items)
              and all(r["pred"] in range(4) for r in mmlu_items))
        gsm_items = read_jsonl(root / "r1.gsm8k.items.jsonl")
        check("gsm8k per-item", all("generated" in r and "extracted" in r for r in gsm_items))
        he_items = read_jsonl(root / "r1.humaneval.items.jsonl")
        check("humaneval per-item 沙箱", all("error" in r for r in he_items))

        # --ckpt 指向 ckpt 子目录与 .pt 文件（配 --preset/--model-configs）也通
        loaded2 = load_model(ckpt_root / "ckpt-00000002", device="cpu")
        loaded3 = load_model(ckpt_root / "ckpt-00000002" / "model.pt", device="cpu",
                             preset=SELFTEST_PRESET,
                             model_configs=root / "model_configs_selftest.yaml")
        check("ckpt 子目录/.pt 直给加载",
              loaded2.info["iter"] == 2 and loaded3.info["preset"] == SELFTEST_PRESET)

        # CLI 端到端
        rc = main(["--ckpt", str(ckpt_root), "--device", "cpu",
                   "--data-gsm8k", str(data_dir / "gsm8k.jsonl"),
                   "--data-mmlu", str(data_dir / "mmlu.jsonl"),
                   "--out", str(root / "cli.json"), "--max-items", "3",
                   "--max-new-tokens", str(small_gen)])
        check("CLI main", rc == 0 and json.loads((root / "cli.json").read_text())["gsm8k"]["n"] == 3)

    print("selftest 完成：全部 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
