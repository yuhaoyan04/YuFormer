# YuFormer Pretrain 原子化验证套件报告（V1–V5）

> **历史证据已失效（2026-10-18）**：现有 `verify_report.json` 混入不在 V1–V5 scope
> 内的 V6 OOM 失败，因而不能作为机器 PASS 证据。runner 已修复为正式运行前清空旧分片并
> 写入显式 scope；需要在新的锁定环境和独占 GPU 窗口完整重跑后，才能恢复本页的 PASS 声明。

> 运行：`pretrain/tests/verify/run_all.sh`（单命令全量；每项可单跑 `python -m pytest pretrain/tests/verify/test_vN_*.py -q`）
> 环境：CUDA_VISIBLE_DEVICES=3（与 E0 臂共享），torch 2.2.0+cu121；每项峰值显存自报（VRAM_REPORT）<3GB 硬断言。
> 机器报告：`verify_report.json`（每项 checks 含全部数值；`report.d/` 为逐项 JSON 片段）。
> 说明：`verify_report.json` 中 `test_anneal_*` 两项属并行开展的中段训练（midtrain）验证，不在本套件范围。

## 一、总览：24/24 PASS

| 验证 | 状态 | 关键证据 |
|---|---|---|
| V1 断点续训原子性 | PASS | SIGKILL 强杀后 resume，21 个 log 点 loss **逐位一致**（max_abs_diff=0.0，容差 2e-3）；resume 后数据序与全量 replay 800 样本 checksum 全同；RESUME_ITER 错值 exit 4 拒绝 |
| V2 LR schedule/warmup | PASS | warmup 线性逐步对拍（rel 1e-12）；constant 后严格=peak（10⁶ 步不衰减）；cosine 端点 peak/min_ratio×peak 精确；子进程 60 步逐步 lr 与独立重算 0 处不符；muon 双优化器 param_groups 实写校验 |
| V3 课程两段衔接 | PASS | 常量流序检测：恰 2 个相异指纹、sample 40 处单次切换；token 总和守恒（6600=40×33×1+80×33×2）；manifest 记录 `data_stage_samples=[40,160]`/`switch_at=[40]`；stage2 中途断点续训 loss 逐位一致 |
| V4 长度外推 | PASS | SWA 16K 输出 vs 4K 滑窗参考 **max_abs_diff=0.0**（127:4096 窗口）；mask 缓存键 (kind, seq_len, window, device) 往返/变窗不串；4K→16K 载权重 forward loss 有限、rope buffer 按 1e7 重建；64K 前向 smoke（CPU）loss 有限；跨 seq resume 被 config_sha 守卫显式拒绝（exit 1）→ 决策项 D1 |
| V5 Muon vs AdamW | PASS | 双优化器 1500 iters 无 NaN/Inf、loss 均下降；Muon 末段 1.5132 ≤ AdamW 1.5139×1.01（**边际通过**）；稳态 tokens/s、双口径 MFU、峰值显存落盘 |

## 二、MFU 表（H800 BF16 峰值 989 TFLOPS）

公式：FLOPs/token = 6·N_nonemb + 12·h·(n_swa·w + n_full·S)（SWA 修正，ctx 上界口径）｜未修正 = 6·N_nonemb + 12·L·S·h（trainer 现行）。MFU = FLOPs/token × tokens/s ÷ 989e12。

**本套件最终口径（`small` 预设，8L/h512/54.0M 总/20.5M non-emb，6 SWA+2 full，w=128，S=4096，mbs1，iters 501–1500 稳态，共享卡实测）**

| 优化器 | tokens/s | 步均 s | TFLOPS（修正） | MFU（修正） | TFLOPS（未修正） | MFU（未修正） | 峰值显存 |
|---|---|---|---|---|---|---|---|
| AdamW | 45,646 | 0.0897 | 8.12 | **0.82%** | 14.79 | 1.50% | 2411MB |
| Muon | 45,103 | 0.0908 | 8.02 | **0.81%** | 14.62 | 1.48% | 2332MB |

参考（窗口期跑通的 ~100M 档 `e1-100m` 实测，12L/h640/59.4M non-emb，10 SWA+2 full；该档峰值 3063MB 贴 3GB 预算，且 OOM 会造成永久显存泄漏，故套件固定为 small）：AdamW 22,472 tok/s、MFU 修正 0.98%；Muon 19,067 tok/s、修正 0.83%；Muon 末段 loss 反例（1.552 vs 1.521，+2.0% > 1% 阈值，三次独立复现一致）——**即 Muon 优势在 ~50M 档不显著、在 ~100M/1500-iters 档为负，供优化器选型决策参考**。小档下 Muon 反超（本表），与"Muon 随规模收益增大"的文献口径一致。

**口径提醒**：绝对 MFU 受共享卡邻居争用影响波动（同配置三次实测 tokens/s 19K–53K）， Efficiency 优化对照应以独占窗口复测；E-50 的 gpu_neighbor_snapshot 已并排出盘可用于拟合争用系数。

## 三、修复清单（全部已修并重跑到绿）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| F1 | `pretrain/src/yuformer_pretrain/trainer.py:88` | 互斥守卫 `raise ConfigError` 未导入该符号，触发时是 NameError 而非预期 ConfigError | 补 `ConfigError` 导入 |
| F2 | `trainer.py`（两处 write_manifest） | manifest 不记录课程段切换点（V3 断言要求；亦为中段审计所需） | 增 `data_stage_samples`、`data_stage_switch_at_samples` |
| F3 | `trainer.py`（run 末） | 无每进程显存自报；vGPU 下 nvidia-smi 按 PID 归因对新进程失效、按 GPU 总量差值被邻居噪声污染（均实测不可信） | 增 `VRAM_REPORT peak_allocated/reserved_mb` 日志行（E-50 同族可观测性） |
| F4 | 验证套件 conftest | run_trainer 用 PIPE 收日志：PrefetchLoader fork worker 继承写端，trainer 退出后 communicate() 等 EOF 死锁（实测挂死 20min，E-1/E-2 事故模式复现） | 日志落文件；SIGKILL 场景改 `start_new_session`+`killpg`（连 worker 一起清，防孤儿） |
| F5 | 验证套件 conftest | `MASTER_PORT=29517` 硬编码与 E0 臂撞车（errno 98 实测） | 按 PID 取端口 |
| F6 | 验证套件 conftest | 共享 vGPU 瞬时显存空窗以 OOM/CUBLAS_ALLOC/launch failure 随机击毁训练进程，死进程泄漏不可回收（E-23 同族，实测泄漏 3.5GB） | run_trainer 内置点火前置 `wait_gpu_free` + 仅对三类显存错误自动重试；V5 候顺手清理了自己遗留的孤儿训练进程（释放 3.5GB） |

注：测试自身缺陷（metrics 落盘路径/layout 理解错误、常量流查重判据、cosine 端点步选择、rope_base 探针物理前提、e1-100m 分析预设笔误等）在迭代中修复，不计入产品修复清单。

## 四、点火前必须决策清单（设计级，未改实现）

- **D1 跨 seq_len/rope_base 续训无协议（阻塞 midtrain，不阻塞 pretrain 点火）**：trainer 的 config_sha256 守卫会拒绝一切配置变更的 `--load` 续训（V4 实测 exit 1，信息明确）。ZGCM 语义下 midtrain 是"载权重+重置优化器"（`--pretrained-checkpoint`），与 pretrain 的 `--load`（全量恢复）是两种协议。需决策：a) midtrain 独立目录实现 weight-only 加载协议（推荐，符宪法 1 运行闭包）；或 b) trainer 增加 `--init-from` 模式（权重加载 + 优化器重置 + RESUME_ITER 不适用，需另设 weight-only 身份守卫）。V4 已证明权重级加载本身可行（load_state_dict strict=True、非持久 rope buffer 按新 base 重建、16K 前向有限）。
- **D2 flash-attn 环境不可用**：包已装但 ABI 损坏（`undefined symbol`，import 即失败），`attn_impl=flash` 当前恒走 fail-fast。生产配置若计划 flash 后端需先修环境（重装匹配 torch 2.2 的 flash-attn≥2.3）；否则 64K+ 场景 sdpa bool mask 显存不可行（64K² mask=4.2GB）。64K smoke 已在 CPU 验证模型语义正确。
- **D3 MFU 核算口径回填**：trainer 的 `flops_per_token` 把 SWA 层按全注意力计（27/32 层注意力 FLOPs 高估 ~3.9×）。V5 双口径公式已验证并落盘；决策：是否将 SWA 修正口径回填 trainer manifest 的 mfu 字段（影响 E 臂间 MFU 可比性与 E-50 门禁基线）。
- **D4 Muon 选型证据链**：~100M 档 1500 iters 下 Muon 末段 loss 高于 AdamW +2%（三次复现），~50M 档边际占优。若最终 pretrain（0.5B）拟用 Muon（pretrain_base.yaml 现状），建议点火前在 200-400M 档补一组对比或引用文献缩放趋势拍板；当前数据不足以支撑"Muon 全面优于 AdamW"。
- **D5 共享卡训练窗口治理**：E-23 泄漏 + 多租户导致 GPU 3 有效空窗 1–5GB 波动。最终 pretrain 点火前应有独占窗口或泄漏清理（gpu-reset）手段；验证套件的空窗等待+重试模式可直接复用为 launcher 的前置检查。

## 五、资源与邻护

- 全程 CUDA_VISIBLE_DEVICES=3 串行；套件自身峰值显存 295MB（V2 tiny）–2411MB（V5 small），均 <3GB 硬预算（自报口径）。
- E0 臂（yf-64k-bpe）全程未受影响、持续出 metrics；`data/experiments/e0q2/` 只读未碰；未做任何 git 提交。
