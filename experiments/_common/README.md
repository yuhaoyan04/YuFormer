# _common — 实验共享基础设施

- `tracker.py` — 实验追踪：每臂 manifest（config hash/数据快照 hash/种子/loss 曲线落盘）
- `plotting.py` — E1 scaling law 拟合与图表（L(N,D) 最小二乘、isoflop 前沿、文献对照、预算外推）
- `quick_eval.py` — 轻量下游评测（GSM8K/HumanEval/MMLU 快速子集，checkpoint 能力曲线）

各实验目录只读引用，不改这里。

## plotting.py — E1 拟合与图表

拟合 Chinchilla 形式 `L(N,D) = E + A/N^α + B/D^β`：

- `(α,β)` 网格搜索（粗 0.05 → 细 0.01，最多 3 轮细化），每个候选 `(α,β,E)` 下对
  `(A,B)` 做**线性域**最小二乘（模型对 A/B 线性，2×2 正规方程闭式解，非负约束）；
  E 网格 [0, 0.5] 粗 0.02 → 细 0.004。
- 置信带：parametric bootstrap（seed=20261008，默认 200 次），输出 95% 预测带。
- 文献对照（表 E1-2）：Kaplan 2020（α=0.076, β=0.095，单变量指数）vs
  Chinchilla 2022（α=0.34, β=0.28）。

```bash
python3 plotting.py --arms-dir <results_dir> --out <out_dir> [--c-budget <FLOPs> ...]
python3 plotting.py --selftest
```

- 输入契约（tracker.py/trainer.py）：扫描 `<arms_dir>/<arm>/{metrics.jsonl, manifest.json}`。
  最终 loss 取值顺序：`eval_val_loss` → `val_loss_*` 均值（trainer 实际写入的键）→
  `final_loss` → `loss`（均取 step 最大的记录）；`N = manifest["params"]`；
  `D = manifest["config"]["train"]["budget_tokens"]`。
- 产出：`<out_dir>/figures/E1-3.png`（isoflop 曲线族，双对数，散点+拟合曲线族+95% 带）、
  `figures/E1-4.png`（D*/N* vs 算力 + Chinchilla≈20 对照线）、
  `fit_result.json`（拟合参数、bootstrap 分位、isoflop 前沿、表 E1-2、外推 (N*,D*)；
  自动含 10× 最大观测算力点，`--c-budget` 可指定 45 天预算等任意 FLOPs）。
- 库 API：`fit_lnbd(points)` / `bootstrap_fits()` / `curve_band()` /
  `isoflop_frontier(points, C_values)` / `extrapolate(fit, C_budget, N_grid)` /
  `lit_comparison_table(fit)`。
- 风格与 `experiments/e0_tokenizer/scripts/plot_e0.py` 一致（Agg backend、同字体回退、
  PNG 200dpi）。matplotlib 不可用时 CLI 报 SKIP（selftest 跳过绘图断言并注明）。

## quick_eval.py — checkpoint 轻量下游评测

```bash
python3 quick_eval.py --ckpt <ckpt> [--tokenizer <dir|tokenizer.json>] \
    [--data-gsm8k f.jsonl] [--data-mmlu f.jsonl] [--data-humaneval f.jsonl] \
    --out result.json [--device cpu] [--max-items N] [--max-new-tokens 256] \
    [--preset P --model-configs path.yaml]
python3 quick_eval.py --selftest
```

- `--ckpt` 三种指向：ckpt 目录（`ckpt-XXXXXXXX/`）、checkpoints 根目录（含
  `latest.json`）、`model.pt` 文件。模型 config 不随 ckpt 存：默认经
  `cursor.json` 的 `run_name` 找回 `<out_dir>/results/<run_name>/manifest.json`
  的 `config.model.preset/configs` 重建（`.pt` 直给时用 `--preset/--model-configs`）。
- 文本编码：`--tokenizer` 给 yuformer tokenizer（`tokenizer/` 产物）时按 BPE 编码；
  缺省走字节级回退（UTF-8 byte = token id）——只保证管线确定可跑，**真实评测必须显式
  传 tokenizer**。
- 确定性：每题 `torch.manual_seed(20261008)`，贪心解码（temperature=0）；
  输出 JSON sort_keys + per-item JSONL（均 tmp+os.replace）。
- 不下载任何数据（`HF_HUB_OFFLINE=1`）。

数据 schema（均为 JSONL，每行一条）：

| benchmark | 字段 | 评测方式 |
|---|---|---|
| gsm8k | `{id?, question, answer}`（answer 含 `"#### N"`） | 生成式，`max_new_tokens=256`，提取输出**末位数字**与 gold 比对（容忍千分位逗号/小数/负号） |
| mmlu | `{id?, question, choices: [...], answer: int}` | loglikelihood：`question` 后接 `" "+choice` 的续写平均 log-prob，argmax，无生成 |
| humaneval | `{task_id, prompt, test, entry_point}`（`test` 须定义 `check(candidate)`） | 生成代码（剥离 markdown 围栏）后子进程执行 `code + test + check(entry_point)`，pass@1 |

沙箱限额：子进程 `RLIMIT_CPU=10s` / `RLIMIT_AS=2GB`（限额写在 runner 脚本内、
exec 后生效，避免 preexec_fn 与 torch 线程的死锁风险），父进程 `timeout=20s`，
`-I` 隔离模式 + 独立临时 cwd。

输出 JSON：

```json
{"gsm8k": {"n": 100, "accuracy": 0.12, "per_item_path": "result.gsm8k.items.jsonl"}, ...}
```

`--selftest`：随机初始化 tiny 模型 + 5 条合成 mmlu + 5 条合成 gsm8k + 2 条合成
humaneval，按 checkpoint.py 磁盘格式造临时 ckpt，跑两遍验证逐条结果一致
（含子进程执行路径）。

## 环境备注

- 本机 torch 为 `2.2.0a0`，其版本字符串会使 `yuformer_model.attention` 的
  `int()` 解析崩溃；quick_eval 在导入 yuformer_model 期间临时清洗
  `torch.__version__` 后恢复（不改 `model/` 任何文件）。
- 真实训练/评测环境用 `YUFORMER_PYTHON`（env 文件），本目录工具仅在
  CPU/本机自测通过；GPU 路径（cuda device）由 `--device` 选择。
