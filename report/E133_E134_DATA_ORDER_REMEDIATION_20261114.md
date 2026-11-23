# E133/E134 独立校准与训练打乱索引修复验证

2026-11-14。正式训练未启动；旧启动协调器已退役，stable 编码继续。

## 已确认问题

质量排序的同分 tie-break、配额截断与后续样本前缀/训练顺序复用同一 doc_id，造成筛选与顺序相关。E133：warmup 完整 web 中 fineweb-edu 参考 token 比例19.59%，校准前缀却33.60%；代码域 github 为33.32% 对48.89%。E134：同一 warmup 实际训练 token 连续十段中 web 从72.70%降到49.90%，目标60.00%。全阶段总配比通过，不能排除此类时间漂移。

## 对齐的数据管线原则

NVIDIA Megatron Core 文档将 document index、sample index 和独立 shuffle index 分开保存，最终训练样本由 shuffle index 映射到样本边界。这里借鉴其显式、可缓存索引的原则，使用独立 BLAKE2b 种子生成因果样本块的双射；块内保留连续读取，不声称与 Megatron 的逐样本分布完全相同。来源：[NVIDIA Data Loading at Scale](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html)、[GPTDataset API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.datasets.gpt_dataset.html)。

E133 用独立文档 rank 抽样扫描完整已选集合，逐域核对文档数和 reference tokens，避免再次选取筛选哈希前缀。哈希、解析与文件身份检查绑定同一打开 FD。每阶段每域至少100M reference tokens；pilot 误差区间为有限总体比率线性近似，仅用于诊断，不能替代全量真实消费配比门禁。

E134 保留现有 token/ledger 字节及原定因果目标集合，只增加阶段内部、独立的块置换。实际 warmup 原连续十段最大偏差 .1269244；64样本块 .0044483；8样本块 .00269676。全部域的全程 token 计数完全不变；headroom 排除、WSD阶段容量和LR边界不变。新块置换原型读取器验证 dataset/snapshot/release/meta SHA、几何尺寸及完整双射；checkpoint 绑定 manifest SHA。真实 loader 原型已通过三个阶段切换、2→1/4拓扑恢复的逐样本一致性、换索引拒绝恢复、游标越界拒绝等测试。

## 身份迁移与剩余门禁

原 frozen113配方所有36inputs/6outputs连同冻结文件、release、3个资格报告，共47文件已逐个复制并重新核对SHA。不可变归档 manifest SHA=e84ea11c27a527a1d8086188e27da92e0d7b203c2b10864634b06c24a481cf57。大型token/ledger/order保留原字节与原身份，不通过改元数据伪造新来源。

新训练配方计划显式绑定原数据配方归档、原release SHA和独立索引。原型校验器拒绝改动阶段、配额、混合目标、token预算、消费样本集合、并行拓扑和原始release；只允许经过独立资格验证的模型/LR及独立块读取策略。原型代码目前保留在 experiments/e4_curriculum，正式runtime和validator尚未接入。

必须等当前模型持续测试结束后再改训练代码，并重新完成CPU、数值、LR、真实生产模型的中断/保存/加载恢复、八卡持续吞吐和逐域质量复核。stable/anneal编码继续使用原冻结配方；只有安全边界后才迁移 canonical 校准及训练配方。三个阶段完整流/长度账本/实际消费配比、交叠和最终生产门禁仍全部必需。

## 模型增宽与学习率

1.060B后备200步吞吐29.98天，但验证NLL6.5694高于0.53B基线6.4177约2.36%，应保留质量复核。更大模型仅测到更高GPU活动利用率或参数量增加，不能据此认定最终质量提高。标准参数化下的最优学习率也不能假定随宽度不变；μP研究通过特定参数化与分组缩放研究超参迁移，不能把它的结论直接当作当前模型已实现μP。来源：[Tensor Programs V](https://arxiv.org/abs/2203.03466)、[作者维护的μP实现](https://github.com/microsoft/mup)。当前仍为原有AdamW/初始化，没有静默引入μP或新优化器。
