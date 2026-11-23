# 外围守护恢复记录（2026-11-18）

已恢复同一次正式八卡训练的外围守护，八卡训练未重启。核验时第 23060 步，累计 96,720,650,240 labels，stable，train loss 2.46738，每卡 TPS 48450。总预算仍为 966254395392 labels，包含 4K WSD、16K 和 64K。

外围协调、运行锁续租和六个辅助任务的原进程均为 zombie，退出记录为 SIGKILL（原始 exit_code=9），发送来源未知。cgroup oom_kill=0、历史 memory.failcnt=4。原 bash launcher 430884、torchrun 430905 和八个训练 worker 430909..430916 均保留原启动时间及父子关系，冻结执行源码校验全部通过。

恢复采用显式运维接管：归档原 owner、状态文件和进程证据，保留正在训练的 generation 6/token，确认原本地 owner 已死后将 owner 交给独立后台 guardian 539905 并续租。没有建立会 fence 八卡 worker 的新训练 generation，没有调用训练 launcher，没有修改训练配置、模型、数据顺序、源码、attempt、优化器、游标或训练完成标记。主状态指向恢复后的 guardian，实际子进程指向原 bash launcher。终态仍仅接受原 launcher 产生的同 attempt 成功标记。

ETA/LR、原生检查点、九域验证、只读预取、长阶段 LR 和冻结长阶段控制器全部恢复，新的 pid 记录在实时 recovery health。guardian 每 30 秒续租并检查子守护退出，不进行 LLM 轮询；失败保留告警，不自动覆盖失败证据。长阶段控制器仍等待真实 4K 完成，能力门禁未绕过。

独立恢复验收：全部 92 份 250..23000 检查点历史验收记录重新验证，六计数、164 Adam 计数/LR、8 rank RNG 及 source pins 一致，最新 metadata SHA 匹配；本项不是重复全浮点扫描。原生观察器重启后的 checked_steps 仅列出当前留存检查点，历史记录保持完整；半日复核 v3 从 immutable admissions 重建全量历史并检查周期连续性。九域最新 22000 步聚合 loss 2.491586685，不可变事件前缀 SHA 和独立重读通过。实际 LR 标量契约全部匹配。

恢复脚本拒绝活 owner、跨主机 owner、错误 attempt/identity/终态的保护检查通过。半日复核 v3 的实际进程、完整历史及验证检查通过，并按预期在未到期处拒绝提前发布。

保守全链路预计 32.569 天，剩余 29.683 天。用户每 12 小时复核的节奏保持不变，下次 2026-11-18T12:46:12.619001+00:00；等待期间不采样、不持续发送进度。

机器验收：`data/health/live-pretrain-supervisor-recovery-admission-20261118.json`，SHA256 `a8b55620d7a30b5cc3b5936e1e2b3b771f4dbb0730db96d4e0f2d025e5869fa4`。原始归档与执行脚本总证据：`data/health/live-pretrain-supervisor-recovery-evidence-20261118.json`，SHA256 `437c4705e855fcb59c60ef4bae0745a896c6982c83617f07be76c96087e1c3ee`。半日检查入口：`/opt/yuformer-build/review_pretrain_every12h_20261118_v3.py`。
