#!/usr/bin/env python3
"""Export small, provenance-bound training reports without touching training.

Reads local receipts only. Does not load weights, initialize CUDA, launch jobs,
claim final capability, or modify frozen recipe inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def loss_values(receipt: dict) -> dict[str, float]:
    metrics = receipt['metrics']
    names = ['academic', 'books', 'chinese', 'code', 'encyclopedia',
             'math', 'multilingual', 'qa', 'web']
    expected = {f'validation_metrics/domain/{name}/loss' for name in names}
    if set(metrics) != expected | {'validation_metrics/loss'}:
        raise ValueError('validation domain roster changed')
    values = {name: metrics[f'validation_metrics/domain/{name}/loss']['value'] for name in names}
    values['aggregate'] = metrics['validation_metrics/loss']['value']
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('validation has invalid loss')
    if abs(sum(values[n] for n in names)/len(names)-values['aggregate']) > 1e-6:
        raise ValueError('validation aggregate differs from equal-domain macro')
    if any(m['step'] != receipt['step'] for m in metrics.values()):
        raise ValueError('validation step mismatch')
    return values


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def export(root: Path) -> dict:
    root = root.resolve()
    health = root/'data/health'
    dest = root/'report/metrics'
    sources: dict[str, dict] = {}

    def load(relative: str | Path) -> dict:
        path = root/relative
        raw = path.read_bytes()
        if len(raw) > 16*1024**2:
            raise ValueError('report input unexpectedly large: '+str(path))
        sources[str(path.relative_to(root))] = {'sha256': digest(raw), 'bytes': len(raw)}
        return json.loads(raw)

    reviews = sorted(health.glob('assistant-half-day-pretrain-review-*.json'))
    if not reviews:
        raise ValueError('no completed half-day review exists')
    review_path = reviews[-1]
    review = load(review_path)
    budget = load('experiments/e4_curriculum/results/effective_total_training_budget.json')
    identity = load('data/runs/pretrain-final-1t-8gpu/run_identity.json')
    attempt = load('data/runs/pretrain-final-1t-8gpu/current_attempt.json')
    if review['attempt_id'] != attempt['attempt_id'] or attempt['identity_sha256'] != identity['identity_sha256']:
        raise ValueError('report review is from another training attempt')
    rows = []
    history = health/'all-production-nine-domain-validation-receipts-20261115'
    for path in history.glob(f'4096-{attempt["attempt_id"]}-step-*.json'):
        receipt = load(path)
        if receipt['step'] > max(map(int, review['validation_losses'])):
            continue
        if receipt['status'] != 'PASS_ACTUAL_NINE_DOMAIN_VALIDATION_AND_IMMUTABLE_EVENT_PREFIX':
            raise ValueError('unadmitted validation receipt')
        rows.append({'step': receipt['step'], 'training_labels': receipt['step']*4194304,
                     **loss_values(receipt), 'receipt_sha256': sources[str(path.relative_to(root))]['sha256']})
    rows.sort(key=lambda row: row['step'])
    if not rows or rows[-1]['step'] != max(map(int, review['validation_losses'])):
        raise ValueError('latest validation receipt missing')
    sweep = load('data/health/model-size-sweep.json')
    scaling = []
    for name, case in sweep['cases'].items():
        measurement = case.get('evaluation', {})
        scaling.append({'case': name, 'parameters': case.get('parameters'),
            'p10_tps_per_device': measurement.get('p10_tps_per_device'),
            'projected_1t_days_duty_09': measurement.get('projected_1t_days_duty_09'),
            'status': measurement.get('status', case.get('status', 'NOT_MEASURED')),
            'scope': 'HISTORICAL_INITIAL_BACKEND_24_STEP_SCREEN_NOT_PRODUCTION_QUALIFICATION'})
    canary = load('data/health/production-throughput-canary.json')
    qualifications = {}
    for name in ['production-recovery-qualification.json', 'midtrain-actual-eight-gpu-stage-qualification.json',
                 'long-production-window-release-v2.json', 'final-all-cpu-suites.json',
                 'final-changed-cpu-suites-20261115.json', 'live-pretrain-supervisor-recovery-admission-20261118.json']:
        path = health/name
        if path.exists():
            proof = load(path)
            qualifications[name] = {'status': proof['status'], 'source': str(path.relative_to(root)),
                                    'source_sha256': sources[str(path.relative_to(root))]['sha256']}
    snapshot = {key: review[key] for key in [
        'status', 'scope', 'reviewed_at', 'next_review_at', 'attempt_id', 'training_metrics',
        'labels_at_last_logged_step', 'phase', 'data_cursor_phase', 'regressed_domains',
        'projected_total_days', 'whole_chain_remaining_days', 'projected_completion_utc']}
    snapshot.update(schema_version=1, latest_validation_step=rows[-1]['step'],
        latest_validation_losses={key: value for key, value in rows[-1].items() if key not in ('step', 'training_labels', 'receipt_sha256')},
        checkpoint_receipt_count=len(review['checkpoint_receipts_rechecked']),
        actual_lr_observations=sum(row['observations_checked'] for row in review['lr_contracts']),
        original_review=str(review_path.relative_to(root)),
        original_review_sha256=sources[str(review_path.relative_to(root))]['sha256'],
        final_capability_completed=False, sft_rl_training_completed=False,
        total_training_labels=budget['actual_total_tokens'], qualifications=qualifications)
    payloads = {
        'latest_pretrain_status.json': json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)+'\n',
        'pretrain_validation.csv': csv_text(rows, list(rows[0])),
        'model_size_initial_screen.csv': csv_text(scaling, list(scaling[0])),
        'qualified_stage_throughput.json': json.dumps({
            'scope': 'PRELAUNCH_SYSTEM_QUALIFICATION_NOT_FINAL_CAPABILITY',
            'source': 'data/health/production-throughput-canary.json',
            'evaluation': canary['evaluation']}, sort_keys=True, indent=2)+'\n',
    }
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        (dest/name).write_text(value, encoding='utf-8')
    provenance = {'schema_version': 1, 'reviewed_at': review['reviewed_at'],
        'scope': 'LIGHTWEIGHT_EXPORT_OF_LOCAL_MACHINE_RECEIPTS_NOT_RAW_TRAINING_DATA',
        'sources': sources, 'exports': {name: {'sha256': digest(text.encode()), 'bytes': len(text.encode())} for name, text in payloads.items()},
        'external_assets': 'Paths under data/ refer to local external assets; not bundled or publicly downloadable by this export.',
        'receipt_verification': 'Domain roster, finite values, step and macro checked here. Original observer independently verified immutable TensorBoard prefixes; full native/finite proof remains external.'}
    (dest/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    write_docs(root, review, snapshot, rows, budget, scaling, canary)
    return {'status': 'EXPORTED_LOCAL_RECEIPTS_NO_TRAINING_MUTATION',
            'reviewed_at': review['reviewed_at'], 'step': review['training_metrics']['step'],
            'validation_rows': len(rows), 'screen_rows': len(scaling), 'output': str(dest)}


def write_docs(root, review, snapshot, rows, budget, scaling, canary):
    m = review['training_metrics']
    date = review['reviewed_at'][:10]
    regression = json.dumps(review['regressed_domains'], ensure_ascii=False)
    banner = (f'**{date}：1.5166B 正式预训练正在八张 H800 上运行。** 最近半日复核为第 {m["step"]} 步，'
        f'已消费 {review["labels_at_last_logged_step"]/1e9:.3f}B training labels，处于 {review["phase"]} 阶段；'
        f'全链路预计 {review["projected_total_days"]:.2f} 天，剩余约 {review["whole_chain_remaining_days"]:.2f} 天。'
        f'这是运行时预测，最终能力尚未完成。最新验证聚合 loss {rows[-1]["aggregate"]:.6f}，逐域回升观察为 `{regression}`。')
    readme = f'''# YuFormer

从数据治理、自训 tokenizer、模型实现，到八卡预训练、长上下文和后训练的可审计 LLM 工程项目。

{banner}

| 入口 | 内容 |
|---|---|
| [技术报告](report/TECHNICAL_REPORT.md) | 当前架构、数据、选型、预算、结果与限制 |
| [实验数据](report/metrics/README.md) | 验证曲线、规模实测、机器快照与来源哈希 |
| [工程经验手册](docs/engineering/PLAYBOOK.md) | 数据、内存、训练、恢复、评测与后训练排障 |
| [异常案例库](docs/engineering/INCIDENTS.md) | 症状、根因、修复、验证及可迁移规则 |
| [维护规范](docs/engineering/REPOSITORY_MAINTENANCE.md) | 发布、清理、证据保留与训练冻结边界 |

## 当前训练

- 模型：1,516,600,064 参数，18L/H2560/FFN6912，40 Q / 10 KV heads，混合滑窗与全局注意力，自训 65536 词表 BBPE。
- 后端：锁定 TorchTitan v0.3.0、FSDP2、AdamW，确定性 Flash 全局 attention 与编译 Flex 滑窗。每次更新 4,194,304 labels。
- 总预算：**966,254,395,392 labels（约 0.966T），包含 4K WSD、16K 和 64K**；启动工期上限 33 天。
- 4K：218453 updates，warmup/stable/decay 为 11921/170770/35762，peak LR 1e-4、末端比例 0.1。
- 16K/64K：7152/4768 updates，各自 warmup 与 cosine decay，25% 短样本 replay。阶段间只加载权重，同阶段恢复完整状态。
- 长阶段系统与数据资格通过；正式长阶段和长短能力验收等待真实父阶段完成。

预算真源：[effective_total_training_budget.json](experiments/e4_curriculum/results/effective_total_training_budget.json)。其中启动标记属于冻结前快照，不能随运行状态修改；当前状态见带时间的 [metrics](report/metrics/README.md)。

## 目录与完成边界

| 目录 | 作用与状态 |
|---|---|
| `data-process/`、`tokenizer/` | 正式三阶段数据和 64K tokenizer 已准备并验收 |
| `model/`、`pretrain/` | 生产 1.5166B 已过八卡资格，正式 4K 运行中 |
| `midtrain/`、`eval/` | 长阶段数据和八卡系统/恢复已通过；最终能力待测 |
| `sft/` | 对话与 assistant-only packing 契约已实现，完整训练闭环待接入 |
| `rl/` | verl/SGLang、rollout 契约已实现，模型 adapter 和训练闭环待完成 |

E1 scaling NRMSE 3.663% 超过预注册 3% 门槛，保留 FAIL 和用户豁免。短时筛查、系统资格、训练期验证及最终能力分别报告。

## 环境与复现

生产依赖见 [backend.lock.json](pretrain/backends/torchtitan/backend.lock.json)，各目录 README 和 `docs/` 说明入口。RL 使用独立环境。源码、配置、测试、正式 tokenizer 和轻量指标进入 Git；大型语料、权重与运行归档通过外部资产和哈希管理。克隆源码不自动获得全部训练资产，也不构成启动重复正式训练的指令。

参考工程：[ZGCM-1](https://github.com/zgcagi/ZGCM-1)。历史材料的日期表示当时事实，当前状态以上述快照为准。
'''
    (root/'README.md').write_text(readme, encoding='utf-8')
    stage_rows = '\n'.join(f'| {p["name"]} | {p["tokens"]:,} | {p["updates"]:,} |' for p in budget['stages'])
    screen_rows = '\n'.join(f'| {r["case"]} | {r["parameters"] or "—"} | {r["p10_tps_per_device"] or "—"} | {r["projected_1t_days_duty_09"] or "—"} | {r["status"]} |' for r in scaling)
    technical = f'''# YuFormer 阶段性技术报告

更新依据：{review['reviewed_at']} 的完整半日复核。原始复核状态 **{review['status']}**。

{banner}

## 数据与架构

正式三阶段数据在完整流、实际消费前缀配比、重叠、顺序和 tokenizer 绑定审计后冻结。统计使用最终 tokenizer 的 training labels；参考 token 用于选取和容量估算，不能替代训练预算。当前独特源集合约 821.290B reference tokens，不能称为 966B 独特训练 tokens。

模型为 1,516,600,064 参数、18L/H2560/FFN6912、40Q/10KV、head dim 64，SwiGLU、RMSNorm、QK normalization、tied embedding，每六层设置全局 attention，其余滑窗。最终 64K BBPE tokenizer 固定 SHA `5dd6bb3f9f587c301e2bb6bb8c987b725fa04d5c49230849038c7d7ec8c5cf9e`，EOS=0。配置见 [生产模型](../model/configs/pretrain_final_model_configs.yaml)。

## 规模实验与选型

八卡初筛比较 1B–5B，并保留实际 OOM、分配器抖动和短时测量限制。下表来自初始后端、24 updates、16 个稳态点，预算为单独 1T；它不能与后来确定性 Flash/Flex 生产结果混成同一后端的加速比。

| case / batch | 参数量 | P10 tokens/s/卡 | 1T 天数，duty .9 | 观测状态 |
|---|---:|---:|---:|---|
{screen_rows}

4.010B microbatch4 虽完成，但预热后出现16301条分配器重试警告，不能作为可选生产结果；后续microbatch2对照约13386.6 tokens/s/卡、120.08天。详细失败与复测见 [规模实验流水账](GPU_MODEL_SIZE_SWEEP_20261114.md)。表内PASS_SCREENING只表示短测完成，不代表显存稳定、质量或生产准入。

宽深、LR、batch、attention backend 与显存策略继续独立优化。最终生产 200 步 canary P10 为 {canary['evaluation']['p10_tps_per_device']:.2f}/卡，4K/16K/64K 整套初始预测 {canary['evaluation']['projected_total_training_days']:.3f} 天。最终同后端、同实际 warmup 样本的 0.530B/1.5166B 对照及扩大开发验证通过逐域门禁；短学习曲线仍不能预测最终能力。证据见 [生产选型](../experiments/e2_architecture/results/model-scale-production-selection.json) 和 [选型分析](../experiments/e2_architecture/results/model-scale-scientific-assessment.json)，其中 preparation 标记属于发布时点。

1.5166B 是在用户优先较大模型、总工期不超过33天的约束下选取的生产候选；它不是最高tokens/s配置，也没有证明扩大模型不会延长训练。不能跨后端或数据口径对比来宣称算力收益。

E1 24 点 × 3 seeds 拟合 NRMSE 3.663% > 3%，保留 FAIL。用户显式豁免 E1 后采用独立系统、数据、数值和恢复门禁启动生产；不能说 E1 已证明最终模型最优。

## 冻结预算与课程

| 阶段 | labels | updates |
|---|---:|---:|
{stage_rows}

总计 **{budget['actual_total_tokens']:,} labels**，相对名义 1T 减少约 3.3746%，减少部分来自 stable，已明确预算口径。4K warmup/stable/decay 为 11921/170770/35762 updates，分别消费 50,000,297,984 / 716,261,294,080 / 149,996,699,648 labels。4K world=8、microbatch=8、grad_accum=16；global batch=4,194,304 labels。AdamW、clip=1，peak LR=1e-4，WSD 末端比例 0.1。

实际 warmup→stable 边界已检查；训练数据使用独立 B8 block permutation，按 global sample cursor 恢复。全阶段总配比通过并不能排除时间漂移，因此另行验证实际消费切片、独立校准和恢复后的下一样本。

16K 为 microbatch2/accum16，64K 为 microbatch1/accum8，均 HSDP 2×4、同一 global batch。peak LR=2e-5、warmup72/48、cosine 到 2e-6；25% 短样本 replay。64K 使用 MLP 重计算、32 路 loss chunk 和已资格验证的分配器/保存策略。跨阶段 weight-only，同阶段 full-state resume。

## 正式训练进展与可靠性

正式训练于 2026-11-15 启动。最近复核：第 {m['step']} 步，{review['labels_at_last_logged_step']:,} labels，phase={review['phase']}，训练 loss={m['loss']}，每卡 TPS={m['tps_per_device']}，CUDA 峰值 {m['memory_gib']} GiB。共 {snapshot['checkpoint_receipt_count']} 份周期检查点历史验收记录、{snapshot['actual_lr_observations']} 个实际 LR 标量重新核对。结构/控制状态检查不等于每次都重扫全部浮点；首250步完整有限性扫描及独立保存另有证据。

实际故障资格包含八卡 SIGKILL 后完整优化器、调度器、数据游标与全 rank RNG 恢复，已测路径逐步 loss 差为0。2026-11-18 外围进程遭 SIGKILL，发送来源未知，cgroup OOM kill=0；原八卡训练存活，通过显式原 generation/token 接管恢复守护，训练未重启。不能仅凭旧 RUNNING、PID 存在或显存高判断健康。

![训练期验证曲线](figures/pretrain_validation.svg)

最新固定九域验证第 {rows[-1]['step']} 步，聚合 loss={rows[-1]['aggregate']:.9f}；逐域短窗回升 `{regression}`。每域固定64样本、262144目标 tokens，聚合为等域宏平均。这是早期点估计，无此处置信区间，不代表公开 benchmark 或最终能力。2026-11-23 books +0.1208% 观察的跟进记录保留，未修改原复核状态或配方。

## 长阶段与后训练边界

4K 完成后控制器必须核对真实 terminal、完整计数和 native checkpoint，再运行 16K。实际 baseline/candidate 的 RULER、长文 BPB、固定短能力与 RepoQA 四组规则重新计算通过后，才准入64K。系统 smoke 和 gold 自测不能充当最终质量；RepoQA 公共代码有污染风险，报告 no-context control，不能称为无污染公开成绩。

长数据与八卡 4K→16K→64K 系统切换、两阶段完整恢复已过资格；真实生产长阶段和最终能力尚未完成。SFT 已有对话状态机、tool call 与 assistant-only packing；RL 已有独立 verl/SGLang 锁版与 rollout 契约，模型 adapter、完整训练和效果验证尚待完成。

## 复用经验与证据

[工程经验手册](../docs/engineering/PLAYBOOK.md) 和 [异常案例库](../docs/engineering/INCIDENTS.md) 覆盖预处理内存、缓存失效、顺序/配比、LR 边界、CUDA context、保存、恢复、评测和运维。新项目复用规则和反例，重新绑定自身数据、模型、依赖和硬件，不继承旧 PASS。

公开轻量数据见 [metrics](metrics/README.md)：CSV、当前机器快照和来源 SHA。原始语料、权重、事件前缀和完整运行证据属于外部资产，未随 Git 打包。报告以实际测量为依据，未把预测工期、未执行的后训练或准备标记写成已完成结果。
'''
    (root/'report/TECHNICAL_REPORT.md').write_text(technical, encoding='utf-8')
    (root/'report/metrics/README.md').write_text('''# 轻量实验数据

- [当前运行快照](latest_pretrain_status.json)：最近完整半日复核的观测，保留逐域回升和未完成项。
- [逐域验证 CSV](pretrain_validation.csv)：每个实际验证 step 的九域 loss、宏平均、训练 labels 和原验收记录 SHA。
- [最初规模筛查](model_size_initial_screen.csv)：旧后端24步筛查，不能当作当前后端生产资格。
- [生产阶段吞吐](qualified_stage_throughput.json)：实际 canary 与长阶段系统资格的预算预测，不代表最终能力。
- [来源清单](provenance.json)：每份本地机器证据和导出文件的 SHA256、字节数。

在有完整本地证据的工作区运行 `python tools/export_pretrain_report.py` 更新；只读取证据并写 README、技术报告及本目录，不加载权重、不启动训练、不更改冻结输入。曲线由 `tools/plot_pretrain_metrics.py` 输出独立 SVG。

`data/` 路径指外部资产，Git 不包含语料、权重、DCP shard 或 TensorBoard 事件。当前快照并非远端服务，不随后台训练实时变化。冻结 JSON 的准备时点标记保留原样，当前运行状态看本目录的 `reviewed_at`。所有质量结论的范围遵循技术报告。
''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(export(args.root), ensure_ascii=False))
