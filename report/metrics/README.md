# 轻量实验数据

- [当前运行快照](latest_pretrain_status.json)：最近完整半日复核的观测，保留逐域回升和未完成项。
- [逐域验证 CSV](pretrain_validation.csv)：每个实际验证 step 的九域 loss、宏平均、训练 labels 和原验收记录 SHA。
- [最初规模筛查](model_size_initial_screen.csv)：旧后端24步筛查，不能当作当前后端生产资格。
- [生产阶段吞吐](qualified_stage_throughput.json)：实际 canary 与长阶段系统资格的预算预测，不代表最终能力。
- [来源清单](provenance.json)：每份本地机器证据和导出文件的 SHA256、字节数。

在有完整本地证据的工作区运行 `python tools/export_pretrain_report.py` 更新；只读取证据并写 README、技术报告及本目录，不加载权重、不启动训练、不更改冻结输入。曲线由 `tools/plot_pretrain_metrics.py` 输出独立 SVG。

`data/` 路径指外部资产，Git 不包含语料、权重、DCP shard 或 TensorBoard 事件。当前快照并非远端服务，不随后台训练实时变化。冻结 JSON 的准备时点标记保留原样，当前运行状态看本目录的 `reviewed_at`。所有质量结论的范围遵循技术报告。
