# YuFormer Model Architecture

从零实现的 Transformer 解码器，支持混合注意力（SWA/Full 交替）与混合位置编码（RoPE/NoPE 交替）。

## 设计原则

1. **从零实现**：不依赖 Megatron-LM，全部用 PyTorch 原生实现，代码可读、可学、可改
2. **配置驱动**：所有超参经 `ModelConfig` dataclass 声明，一配置一模型
3. **可扩展**：第一版 Dense 模型，架构预留 MoE / MHC 扩展接口

## 架构规格

### V1: Dense Transformer + 混合注意力

```
Input Tokens
  → Embedding (tied with output)
  → [Transformer Block] × N
    → RMSNorm → GQA Attention (SWA or Full, RoPE or NoPE) → Residual
    → RMSNorm → SwiGLU MLP → Residual
  → RMSNorm
  → Output Projection (tied embedding)
```

**混合注意力**：层 `i` 的注意力类型由 `(i % attention_every) == 0` 决定：
- `attention_every=1`：全部 Full Attention（标准 Dense）
- `attention_every=6`：每 6 层一个 Full，其余为 SWA（ZGCM-1 规格：27 SWA + 5 Full）

**混合位置编码**：层 `i` 的位置编码由 `(i % rope_every) == 0` 决定：
- `rope_every=1`：全部用 RoPE（标准做法）
- `rope_every=3`：每 3 层一个 RoPE，其余为 NoPE（NoPE = 无位置编码，靠注意力偏置或全局注意力学习位置信息）

> NoPE（No Positional Encoding）研究表明，Full Attention 层不需要显式位置编码也能感知位置——因为全局注意力可以通过内容-内容交互隐式学习顺序。YuFormer 创新点之一：**RoPE/NoPE 交替的消融实验**。

### 分组查询注意力（GQA）

- `num_heads` 个 Q 头，`num_kv_heads` 个 KV 头（`num_kv_heads < num_heads`）
- ZGCM-1 规格：32 Q / 8 KV，head_dim=128
- 小规模实验：可配置为 MHA（`num_kv_heads == num_heads`）或 MQA（`num_kv_heads == 1`）

### 滑窗注意力（SWA）

- 窗口大小 `window_size`：每个 token 只关注 `[i - window_size, i]` 范围（左闭右闭）
- ZGCM-1 规格：`window_size=127`（128 token 窗口）
- 小规模实验：可配置为 64 / 128 / 256

### RoPE

- `rotary_percent`：应用 RoPE 的维度比例（ZGCM-1 用 0.334，即约 1/3）
- `rope_base`：频率基数（ZGCM-1 pretrain 用 5M，256K 阶段用 10M）

## 配置

见 `configs/model_configs.yaml`，预置多组规模：

| 配置 | 层/隐藏/FFN | 头/KV头 | 参数量 | 用途 |
|---|---|---|---|---|
| `tiny` | 4 / 256 / 512 | 4 / 2 | ~5M | 管线验证 |
| `small` | 8 / 512 / 1024 | 8 / 4 | ~30M | 数据配比实验 |
| `base` | 12 / 768 / 2048 | 12 / 4 | ~100M | 缩放律初步 |
| `medium` | 24 / 1024 / 2816 | 16 / 8 | ~350M | 扩展验证 |
| `large` | 32 / 4096 / 11008 | 32 / 8 | ~7B | ZGCM-1 规格 |

## 使用

```python
from yuformer_model import ModelConfig, create_model

config = ModelConfig.from_preset("small")
model = create_model(config)
tokens = torch.randint(0, config.vocab_size, (2, 512))
logits = model(tokens)
```

## 可扩展性（V2+ 预留）

- **MoE**：将 SwiGLU MLP 替换为 MoE Layer（`MLP` 基类 → `MoELayer`），路由器共享 attention 的层选择策略
- **MHC（Multi-Head Convolution）**：在 SWA 层的注意力旁路加 depthwise conv 替代显式注意力
- **门控注意力**：参考 ZGCM-1 的 `attention_output_gate`（`x * sigmoid(gate)`），仅作用于 SWA 层
- 这些扩展通过 `BlockType` 枚举和 `MLPBase` / `AttentionBase` 基类预留接口，V1 仅实现 Dense 版本
