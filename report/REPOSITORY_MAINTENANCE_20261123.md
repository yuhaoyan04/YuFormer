# 2026-11-23 仓库整理与发布记录

用户明确授权维护 `git@github.com:yuhaoyan04/YuFormer.git`，提交/推送最新代码、轻量实验数据和报告，清理确认过期的本地产物。本次授权取代早期交接文档中的不提交/不推送限制。

## 已完成

- README 与 TECHNICAL_REPORT 更新为真实八卡1.5166B运行状态，统一总预算966254395392 labels，包含4K WSD/16K/64K，明确后训练和最终能力未完成。
- 导出31个固定九域实际验证点、13个初始规模筛查记录、正式系统吞吐与来源SHA，并生成独立SVG。旧后端24步筛查与当前生产资格分开，4B分配器抖动不作为可选生产结果。
- 异常库包含44条历史记录和17条近期案例；保留books +0.1208%观察、E1 FAIL、失败系统配置及未知根因，没有把FAIL改成PASS。
- 保存11份正在使用的外部观察器源码副本与原SHA；运行租约token脱敏，实际运行源码和进程路径未改。
- 清理52个过期缓存/旧产物目录、961个文件，回收13,661,559,135字节（13.662GB）。主要为旧pre-calibration canary DCP shard与结构日志，原生metadata哈希和文件尺寸留在清理清单。
- 391份冻结依赖核对未变；小型文件重新SHA，大文件检查精确stat身份，不重复扫描TB级数据。实际训练、断点归档、token流、release和活跃租约未动。
- 新增维护工具8项测试通过：拒绝错误attempt/验证缺域/非有限值/错误宏平均、保留质量观察、阻止清理保护依赖和symlink边界。已存在训练代码沿用当前源码绑定的资格证据，没有因发布重复启动GPU测试。

清理后训练仍在推进，观测第61240步，loss 2.33931，每卡TPS 48503；守护无新告警。完整半日检查继续按原12小时节奏，不因仓库维护提高采样频率。

## 初始阻塞与实际发布

本次只读检查发现远端main为`0e340e0af7f8033314ec9880c5a55c9613f86bcb`。SSH认证身份`dhh1995`收到对目标仓库的Permission denied；GitHub连接器身份`Edenwu617`的仓库权限`push=false`；HTTPS helper指向缺失的gh程序且无可用token。这次失败已保留为事故证据。

先完成本地提交 `b41084f4d14565b1449f59f482d63c507ea12b8b` 与已验证离线bundle（5,816,784字节，SHA256 `dd9fb2682daf90c32a6cdc8afb733b788567afb55ac4d305be2b4d89abfeaae2`）。用户随后完成本机 GitHub CLI 设备授权；实际账号核验为 `yuhaoyan04`、仓库 `push=true`。重新读取远端并确认祖先关系后，执行非强制 HTTPS 推送；UTC 2026-11-23 04:46:22 核验远端 main 等于该完整快照提交，正式发布成功。轻量机器证据见 [发布核验](metrics/repository_publication_20261123.json)。

凭据保存在仓库外的隔离目录，目录700、凭据文件600。本仓库配置独立 HTTPS pushurl 和局部 credential helper，后续可直接 `git push origin main`；原SSH fetch URL、全局Git配置及聊天GitHub连接器未改。每次后续提交仍须独立检查实际远端SHA；设备授权不等于聊天连接器账号切换。

## 继续维护

半日复核后运行`tools/export_pretrain_report.py`和`tools/plot_pretrain_metrics.py`；提交前运行`tools/audit_repository.py`检查Git blob大小、凭据形态、运行产物边界和当前文档链接。新故障追加案例及实际证据。仍被训练/测试/审计引用的历史代码和配置保留至引用解除；不靠日期批量删库。大型语料、权重、事件和运行归档不进入Git。
