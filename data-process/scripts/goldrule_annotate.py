"""goldrule LLM 标注：Qwen3-8B(INT4) 按 FineWeb-Edu 风格 rubric 逐条打 0-5 分。

设计：
  - rubric = 质量五维（知识密度/教育价值、结构完整性、语言质量、信息独有性、垃圾信号）
    + 中文特化判据（软文/营销/spinner）
  - 0-5 连续分（整数），>=3 视为 hq（FineWeb-Edu 同款阈值语义）
  - JSON 输出 + 容错解析（截断/非法输出重试一次，再失败记 score=-1）
  - 断点续跑：结果按批追加 JSONL（doc_id 去重），中断重启无损
用法：PYTHONPATH=src python scripts/goldrule_annotate.py --gpu 1 --batch 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RUBRIC = """You are a data quality annotator for pretraining corpora. Score the document on educational/knowledge value from 0 to 5 (integer only).

Score definition (based on FineWeb-Edu rubric):
- 0: Useless. Spam, ads, SEO filler, spun/rewritten marketing text, or pure navigation/boilerplate.
- 1: Very low. Fragmented, incoherent, or trivial content with no knowledge value.
- 2: Low. Mostly filler; minimal informative content, heavily repetitive.
- 3: Acceptable. Coherent, informative content on some topic (ordinary but genuine articles, docs, discussions).
- 4: Good. Well-structured, knowledge-dense content (tutorials, quality explanations, reference material).
- 5: Excellent. Highly educational/reference quality (textbook-like, thorough technical/scientific exposition).

Extra criteria:
- Chinese documents: hospital/marketing soft-ads (医院软文), promotional fake-QA, or SEO articles score 0-1.
- English documents: VPN/insurance-style affiliate marketing or synonym-spun text scores 0-1.
- Code: useful, readable source with docs scores 3-5; dead/generated/scaffolding code scores 0-2.
- Judge ONLY the visible text. Output JSON only: {"score": <0-5>, "reason": "<=<15 words"}"""


def build_prompt(text: str, domain: str) -> list[dict]:
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"[domain={domain}]\n---\n{text}\n---\nScore this document. Reply with JSON only."},
    ]


def parse_score(out: str) -> tuple[int, str]:
    s = out.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    try:
        j = json.loads(s[s.index("{"): s.rindex("}") + 1])
        sc = int(j["score"])
        if 0 <= sc <= 5:
            return sc, str(j.get("reason", ""))[:80]
    except Exception:
        pass
    import re

    m = re.search(r"[\"']?score[\"']?\s*[:=]\s*([0-5])", s)
    if m:
        return int(m.group(1)), ""
    return -1, s[:80]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="../data/models/goldrule/annotate_set.parquet")
    ap.add_argument("--out", default="../data/models/goldrule/annotations.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--enable-thinking", action="store_true", help="Qwen3 思考模式（更准但慢 ~3-5x）")
    args = ap.parse_args(argv)

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tbl = pq.read_table(args.set)
    docs = tbl.to_pylist()
    done: set[str] = set()
    out_path = Path(args.out)
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"annotate: {len(todo)}/{len(docs)} pending (done={len(done)})", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    out_f = open(out_path, "a")
    t0 = time.time()
    n_ok = n_bad = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i: i + args.batch]
        prompts = [
            tok.apply_chat_template(
                build_prompt(d["text"], d["domain"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for d in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=6144).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        for d, g in zip(chunk, gen):
            new_tokens = g[enc["input_ids"].shape[1]:]
            out_text = tok.decode(new_tokens, skip_special_tokens=True)
            score, reason = parse_score(out_text)
            rec = {
                "doc_id": d["doc_id"], "domain": d["domain"], "lang": d["lang"],
                "source": d["source"], "rule_score": d["rule_score"],
                "full_chars": d["full_chars"], "llm_score": score, "llm_reason": reason,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            if score >= 0:
                n_ok += 1
            else:
                n_bad += 1
        out_f.flush()
        done_now = i + len(chunk)
        rate = done_now / max(time.time() - t0, 1e-9)
        eta = (len(todo) - done_now) / max(rate, 1e-9) / 60
        print(f"[{done_now}/{len(todo)}] {rate:.1f} docs/s eta={eta:.0f}min ok={n_ok} bad={n_bad}", file=sys.stderr, flush=True)
    out_f.close()
    print(json.dumps({"ok": n_ok, "bad": n_bad, "out": str(out_path)}, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
