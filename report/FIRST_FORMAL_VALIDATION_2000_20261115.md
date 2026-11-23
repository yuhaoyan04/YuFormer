正式八卡 pretrain 第 2000 步首次九域验证已通过独立验收（2026-11-15 16:24 UTC）。

训练 loss 4.91371；验证总体 token NLL **5.238113880**，九域等权均值 5.238114039，差约1.59e-7，符合预声明1e-6相对浮点容差。每域64个4096-target样本（262,144标签），九域共2,359,296个实际留出标签。没有空域或非有限loss。

| 域 | Token NLL |
| --- | ---: |
| 学术 | 4.739239 |
| 书籍 | 4.657475 |
| 中文 | 6.599841 |
| 代码 | 4.774806 |
| 百科 | 5.383183 |
| 数学 | 4.499459 |
| 多语言 | 6.382286 |
| 问答 | 4.776426 |
| 网页 | 5.330310 |

同一步完整native DCP于16:24:22 UTC保存，计数独立PASS：step/scheduler2000，global consumed_samples2,048,000，global_labels8,388,608,000；164组Adam计数/LR及8rank RNG完整。实际训练LR约1.67771159e-5，匹配冻结warmup计划。八次定期保存250..2000全部原生结构与控制状态PASS。

原始TensorBoard在训练中继续追加；本次证据将其106,867字节原始前缀独立封存，再从不可变副本重读全部指标，校验前缀SHA61f1e8744874c25231b8fe406cdef7012debffbfac098db690c7606c8d8b30f3。验收JSON所有source_inputs哈希已再次独立匹配，不以活跃事件文件的未来完整哈希作为历史证据。

- [实际验收JSON](../data/health/first-formal-nine-domain-validation-2000-admission-20261115.json)，SHA 6741ea104c37277ec3308ca0dd6e1462b135ae8398dc57f316425e463482b4be。
- [原生2000控制状态](../data/health/formal-native-checkpoint-state-admissions-20261115/step-2000.json)。
- [实际不可变事件前缀](../data/health/first-formal-validation-2000-evidence-20261115-a00c4c66fed0/events.out.tfevents.snapshot.0)。

这是早期训练进展和验证完整性检查，不是全量训练完成、最终能力准入或与不同训练预算模型的受控质量比较。1.5166B模型/AF54源码、数据/配比/顺序、c5bc正式attempt及966254395392总标签预算保持不变。继续监控验证后的训练衔接、后续保存及完整WSD/16K/64K阶段。
