# YuFormer Model Architecture

从零实现的 Transformer 解码器（ZGCM-1 式混合注意力：门控滑窗 + 全注意力），PyTorch 原生 + SDPA，不依赖 Megatron。

## 设计原则

1. **从零实现**：不依赖 Megatron-LM，全部用 PyTorch 原生实现，代码可读、可学、可改
2. **配置驱动**：所有超参经 `ModelConfig` dataclass 声明，一配置一模型
3. **ZGCM-1 规格准绳**：门控/QK-norm/层分布/RoPE/初始化逐条对齐 ZGCM-1 官方（Megatron fork）实现

## 架构规格

```
Input Tokens
  → Embedding（默认 untied；tie_word_embeddings=true 时与 lm_head 共享）
  → [Transformer Block] × N
    → RMSNorm → GQA Attention (Full causal 或 门控滑窗) → Residual
    → RMSNorm → SwiGLU MLP → Residual
  → RMSNorm (final)
  → lm_head → logits (float32)
```

### 层分布（1-indexed）

层编号从 **1** 开始：`layer_no % attention_every == 0` 的层为**全注意力层**（恒因果，无滑窗），其余为**滑窗注意力层**。

- `attention_every=1`：全部 Full Attention（标准 Dense）
- `attention_every=6`：第 6,12,18,… 层为 Full（ZGCM-1 规格：32 层 → 5 Full + 27 SWA；0.5B 的 24 层 → 4 Full + 20 SWA）

### 滑窗注意力（SWA）与输出门控

- 窗口 `(left=window_size-1, right=0)`：每个 token 关注 `[i-window_size+1, i]`，含自身共 `window_size` 个 token
- mask 按 `(seq_len, window_size, device)` 缓存为 bool tensor（True=保留），向量化构造（`arange` 差值 + 比较），无 Python 循环
- **门控（仅滑窗层）**：QKV 融合投影的额外 gate 通道，布局 `[q|g|k|v]`，gate 宽 = `num_heads × head_dim`（与 q 同宽），初始化同 QKV（N(0, 0.02)）；gate 在所有头 concat 之后、O-proj 之前施加：`out = out * sigmoid(gate.float())`（sigmoid 在 fp32 算）。全注意力层无 gate 通道
- 开关：`attention_output_gate`（全局层永远无 gate）

### QK-RMSNorm

- 开关：`qk_norm`。QKV 投影后、RoPE 前，对 q 和 k 分别做 per-head RMSNorm（对 head_dim 维），eps=1e-6，可学习 weight 初值 1，无 bias

### 分组查询注意力（GQA）

- `num_heads` 个 Q 头，`num_kv_heads` 个 KV 头；SDPA `enable_gqa=True`（torch>=2.5；旧版本回退 `repeat_interleave`）
- ZGCM-1 规格：32 Q / 8 KV，head_dim=128；0.5B：16 Q / 8 KV，head_dim=64

### RoPE

- rotate-half（GPT-NeoX）风格
- `rot_dim = int(head_dim × rotary_percent)`（截断，向下取偶），只旋转前 rot_dim 维、其余直通；ZGCM-1：`int(128×0.334)=42`
- `inv_freq = base^(-2i/rot_dim)`, i=0..rot_dim/2-1
- `rope_base` 可配置（ZGCM pretrain=5e6，256K 阶段=1e7）
- 按 `position_ids` 动态 gather（缺省 0..S-1 走切片快路径），为 packing/变长预留接口

### 位置编码混合（YuFormer 研究开关）

层 `i`（1-indexed）用 RoPE 当且仅当 `layer_no % rope_every == 0`；`rope_every=1` 即全 RoPE（ZGCM-1 规格，所有预设默认值）。`rope_every>1` 是 YuFormer 保留的 NoPE 消融开关，ZGCM-1 官方没有。

### 初始化

- N(0, 0.02)：embedding、QKV（含 gate）、fc1（gate_proj/up_proj）
- N(0, 0.02/√(2·num_layers))：O-proj、fc2（down_proj）
- RMSNorm weight = 1
- 当前 `pretrain-0.5b` 使用 tied embedding；untied 版本保留为独立消融 preset。新候选 `pretrain-0.5b-cs336` 也使用 tied embedding，并单独存放在 `configs/pretrain_0.5b_cs336.yaml`，避免改动历史 scaling 结果绑定的主 preset 文件。

### Mask 契约

- 统一 bool mask，True=保留：`attention_mask` 形状 `[batch, seq]`（key padding）或 `[batch, 1, seq, seq]`
- 全注意力层恒因果：无 padding mask 时走 SDPA `is_causal=True` 快路径；有 padding mask 时与 causal 按位合并（bool AND），不会丢失因果性
- 滑窗层：窗口 mask 与 padding mask 按位合并

## 配置

见 `configs/model_configs.yaml`：

| 配置 | 层/隐藏/FFN | 头/KV头 | 参数量 | 用途 |
|---|---|---|---|---|
| `tiny` | 4 / 256 / 512 | 4 / 2 | ~10.8M | 管线验证 |
| `small` | 8 / 512 / 1024 | 8 / 4 | ~54.0M | 数据配比实验 |
| `base` | 12 / 768 / 2048 | 12 / 4 | ~282.2M | 缩放律初步 |
| `medium` | 24 / 1024 / 2816 | 16 / 8 | ~572.6M | 扩展验证 |
| `pretrain-0.5b` | 32 / 1024 / 2816 | 16 / 8 | ~473M (tied) | **0.5B 预训练主配置**（32L 对齐 ZGCM 层分布：5 Full + 27 SWA） |
| `pretrain-0.5b-cs336` | 16 / 1536 / 4096 | 24 / 6 | ~530.1M (tied) | 独立候选配置 `configs/pretrain_0.5b_cs336.yaml`：H/L=96，SwiGLU FFN=8/3·H，head_dim=64；尚未替换生产锁定配置 |
| `pretrain-0.5b-untied` | 32 / 1024 / 2816 | 16 / 8 | ~540M | tied-vs-untied 消融臂（E3） |
| `large` | 32 / 4096 / 11008 | 32 / 8 | ~7.20B | ZGCM-1 规格 |

词表对齐 tokenizer 消融档位（32768 / 65536 / 131072）。参数计数公式：
`2·V·d(untied) + L_sw×(qgkv+o+mlp) + L_full×(qkv+o+mlp) + norms`（`tests/test_model.py` 对每个预设做解析式断言，容许 2% 误差）。

## 使用

```python
from yuformer_model import ModelConfig, create_model

config = ModelConfig.from_preset("pretrain-0.5b")
model = create_model(config)
tokens = torch.randint(0, config.vocab_size, (2, 4096))
logits = model(tokens)                              # [2, 4096, vocab], float32
logits = model(tokens, attention_mask=bool_mask)    # True = keep
logits = model(tokens, position_ids=packed_positions)  # packing 接口
```

## 运行时环境与测试

系统 Python（/usr/bin/python3，3.10）+ torch 2.2.0a0（CUDA 12.3）；`pyproject.toml` 声明 `torch>=2.2`。
flash 后端（`attn_impl="flash"`）要求 flash-attn>=2.3.0（`window_size` kwarg，上游 2.3.0 引入），
init 期 fail-fast 并提示改用 sdpa/升级；flash-attn 2.6.3 已由源码构建安装（2026-10-14，nvcc
12.3/sm90），不满足版本的旧包（如 2.0.4）会被 guard 拦截。

```bash
PYTHONPATH=src python3 -m pytest tests/ -q   # 32 项全过（含 flash parity，CUDA 上真实执行）
```

## 与 ZGCM-1 官方实现的对应关系与已知差异

**对应（逐条核实自 ZGCM-1 Megatron fork 源码）**：

| 特性 | ZGCM-1 | YuFormer |
|---|---|---|
| 层分布 | `layer_no % 6 == 0` 全局（1-indexed） | 同，attention_every 可配 |
| 门控 | QKV 融合 gate 通道，仅滑窗层，`x*sigmoid(gate)` fp32 | 同，布局 `[q|g|k|v]` |
| QK-norm | QKV 后 RoPE 前 per-head RMSNorm eps=1e-6 | 同 |
| 滑窗 | 含自身 128 token | 同，window_size=128 |
| RoPE | rotate-half，rot_dim=trunc(0.334·hd)=42，base 5M→10M | 同，percent/base 可配 |
| 初始化 | 0.02；o/fc2 用 0.02/√(2L) | 同 |

**已知差异**：

1. YuFormer 保留 `rope_every`/NoPE 研究开关（ZGCM-1 全 RoPE）；预设默认 `rope_every=1` 与官方一致
2. 词表对齐自训 BPE 消融档（32768/65536/131072），非 GLM-5.1 的 154880；large 预设已注明
3. mask 语义为 bool True=保留（Megatron 内部用 additive），语义等价但契约不同
4. 实现为 PyTorch 原生 SDPA（非 TE/FA3 kernel），4K 上下文 + 4×H800 可训（训练侧接入时配合 FSDP/重计算）

## 可扩展性（V2+ 预留）

- **MoE**：将 SwiGLU MLP 替换为 MoE Layer（`MLP` 基类 → `MoELayer`），路由器共享 attention 的层选择策略
- **MHC（Multi-Head Convolution）**：在 SWA 层的注意力旁路加 depthwise conv 替代显式注意力
- **门控变体**：`use_gate` / `attention_output_gate` 配置化，可做门控施加范围、激活函数的消融
