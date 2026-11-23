# TorchTitan backend

这是 YuFormer 正式训练栈的版本边界。`backend.lock.json` 固定 TorchTitan、PyTorch、
Python 与 CUDA；`preflight.py` 在任何正式实验启动前检查全部硬条件。

```bash
export YUFORMER_TORCHTITAN_ROOT=/opt/torchtitan-v0.3.0
python pretrain/backends/torchtitan/preflight.py \
  --out data/health/torchtitan-preflight.json
```

只有输出 `status=ready` 才允许执行 TorchTitan smoke、恢复/reshard canary 或正式训练。
Python/PyTorch/CUDA build 按 lock 的 major/minor（PyTorch 含 patch）精确匹配，driver 只设最低值；
TorchTitan checkout 还必须位于锁定 commit 且 tracked tree 干净，防止“同 commit、不同本地代码”。
当前仓库中的 `yuformer_pretrain.trainer` 标记为 `pytorch-fsdp1-compat`，用于读取旧
checkpoint、数据契约测试与迁移对拍，不再承载新的 scaling/pretrain 结论。

v0.3.0 的官方 release validation 固定 PyTorch 2.14.0、torchvision 0.29.0、
torchao 0.18.0、Triton 3.8.0 和 `cu126` wheel；YuFormer 还显式锁定上游
`pyproject.toml` 声明但 `requirements.txt` 漏列的 `torch-checkpointing==0.1.0`。CUDA 12.x
按 NVIDIA minor-version compatibility 要求 Linux driver >=525.60.13；本节点 R535 在门禁内。
Python 3.12.15
独立安装在 `/opt/yuformer-python`，不修改系统 Python。userspace 环境可以和 CPU 数据物化
并行准备，不需要等待驱动升级：

```bash
bash pretrain/backends/torchtitan/prepare_env.sh
```

脚本在临时目录中安装固定 core wheel 与 TorchTitan checkout，`pip check` 通过后才原子发布
`/opt/yuformer-torchtitan`；状态和完整依赖快照分别写入
`data/health/torchtitan-env-prepare.json` 与 `data/health/torchtitan-env-freeze.txt`。它不执行 CUDA
初始化，因此不会占用 GPU。发布后 `validate_userspace.py` 还会独立检查完整版本组合、FSDP2、
DCP、源码 commit、`pip check`、依赖快照和迁移后的 console-script shebang，结果写入
`data/health/torchtitan-userspace.json`。本机无需升级驱动，之后执行：

已发布环境若仍与源码 commit、依赖快照和版本锁一致，准备脚本会先走本地 validator 快速路径，
不再访问包索引；只有验证失败时才进入修复安装。

构建脚本会清除宿主镜像遗留的 `SETUPTOOLS_USE_DISTUTILS=stdlib`；Python 3.12 已移除 stdlib
distutils，保留该变量会令 editable metadata 构建失败。TorchTitan editable 安装复用环境中锁定
的 setuptools，并禁用临时 build isolation，避免产生未审计的第二套构建依赖。

```bash
bash pretrain/backends/torchtitan/bootstrap_env.sh
```

脚本会先检查驱动，再幂等调用 userspace 准备脚本并运行同一个 preflight。GPU 训练仍由
`status=ready` 严格门禁。

仓库已提供外部模块 `yuformer_ext`，通过 TorchTitan 的 fully-qualified module 协议接入，
不修改上游 checkout。适配层包含 YuFormer `BaseModel`、FSDP2 分层包装、sequence-chunked CE、
预 tokenized 的 DCP stateful dataloader、WSD/AdamW typed config 翻译和正式启动器：

```bash
bash pretrain/backends/torchtitan/launch.sh pretrain/configs/pretrain_base.env
```

启动器先运行版本/驱动/源码 preflight，任何一项不合格都会在创建训练进程前退出。当前代码接入
的运行状态以 `data/health/torchtitan-qualification.json` 及其当前代码身份为准。
2026-10-22 审计修复了 meta 初始化、FSDP 梯度归一化和恢复路径中的实际缺陷；旧运行证据已失效。
数值资格 PASS 后仍须完成实验、冻结数据与生产吞吐门禁，详见
[`PRETRAIN_DEEP_AUDIT_20261022`](../../../report/PRETRAIN_DEEP_AUDIT_20261022.md)。
`pretrain/scripts/pretrain_readiness_watch.py` 自适应检查门禁：阶段切换和小时级任务每 15 分钟，
稳定的日级物化或训练任务每 2 小时；环境满足且至少四卡空闲后自动运行
`run_torchtitan_qualification.py`，末端 4→8 测试需要八卡。验收顺序是：

1. 用锁定版 TorchTitan 的真实 scheduler 验证 50B/800B/150B WSD step 边界和末端 LR；
2. 在 Python 3.12 / PyTorch 2.14 环境对拍 meta 初始化、单卡 logits/loss/gradient，以及两卡 masked chunked loss 的梯度累积；
3. 4×H800 FSDP2 + DCP async smoke；
4. 分别在训练、异步保存和加载时执行 SIGKILL，再验证恢复后的 loss 与数据游标；
5. 从 4 卡 checkpoint 以 2 卡和 8 卡重分片加载，验证完整 rank roster 与全局样本游标；
6. 资格报告全部 PASS 后启动 15 臂 bracketed LR pilot（3 个规模 × 5 点），再运行 9 臂
   batch/LR 联合 pilot；两者均 PASS 后才允许 E1/E2 scaling。

E1 的四卡工期先按单卡吞吐和 0.70 保守分布式效率筛选候选。E1/E2/E3 冻结后，自动生成与
正式模型、global batch、LR 和四卡 FSDP2 拓扑完全一致的 200-step sustained canary；最终门禁
使用稳态每卡 p10 tokens/s 推算 1T 工期，必须不超过 45 天。canary 报告与冻结配方、配置和 env
哈希绑定，任一输入变化都会强制重跑。

八份运行合同位于 `pretrain/configs/qualification/`。恢复对拍使用与中断臂完全相同的
24-step cosine 调度 reference；每次 dataloader 恢复会在 run 目录写入
按 consumed-sample/world/rank 命名的游标证据；缺 rank、重复数据位置或 loss 对拍超差均会令
`data/health/torchtitan-qualification.json` 保持 FAIL。

九域验证按完整 DP global batch 轮转，每个 global batch 只含一个域。验证步数使用
`domains × samples_per_domain / (DP world × local batch)`，因此生产四卡和单卡实验都严格消费
声明的每域 64 个样本；合同未按 global batch 对齐时在启动前失败。

资格报告还记录 `qualification_identity`：它覆盖 TorchTitan 适配代码、模型实现、八份合同、
模型配置、数据 release/audit/meta 以及实际加载的 tokenizer 文件。任一输入变化后，旧 PASS
会被判为 STALE；runner 在 generation lease 保护下归档旧证据和旧 checkpoint，再从第一项门禁
重新验收。正式启动器也会重新计算 tokenizer 文件 SHA，声明哈希与实体文件不一致时拒绝启动。

权威版本与原因见 [`训练栈选型与迁移.md`](../../../docs/research/训练栈选型与迁移.md)。

共享 loss 实验与性能证据分别验收。LR/batch 选定后，watcher 自动生成五档 32-update 的
`e1_performance` 合同，与 E1 并行，在空闲卡上补测。队列每 5 秒检查额外 compute PID，
出现争用会中断并归档自有短测，再从头重试；不终止外部进程。E1 的性能拟合读取独立补测，
E2/E3 的 loss 决策需当前资格和对应组身份审计；全局冻结身份门禁覆盖 91 组证据。
最终选型仍需持续四卡生产 canary，短测不代替长期吞吐验收。
