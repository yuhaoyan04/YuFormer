# Active pretrain operations

Updated 2026-11-23T02:16:09.537593+00:00. This file is the current handoff. Historical diaries were consolidated into `report/history/operations-consolidation-20261123/`; dated technical evidence remains in the report and incident registry.

## User instructions

- Keep the session open and monitor every12hours using interruptible clock.sleep, without continuous LLM polling. Background guards handle automatic heartbeats. Do not send a final answer merely because a review passes.
- Full 8GPU1.5166B pretrain is authorized and running. Budget966254395392labels includes4KWSD/16K/64K; whole-chain ceiling33days. Quantified small single-domain tradeoffs are accepted; data/numeric/recovery and future actual capability gates remain strict.
- On2026-11-23 the user explicitly authorized committing/pushing latest code, experimental metrics and reports to `git@github.com:yuhaoyan04/YuFormer.git` and deleting confirmed obsolete local artifacts. This supersedes old no-commit/no-push handoff notes. Do not force-push unknown remote changes.
- Do not spawn agents. Workspace is this inner `YuFormer` directory. Do not change live training dependencies or the locked environment.

## Current training and state

Latest complete review: 2026-11-23T00:46:28.253923+00:00, step60780, 254,929,797,120labels, phasestable, loss2.3253, TPS47746/device, CUDA peak67.46GiB. StatusHALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION. Nine-domain validation step60000, aggregate2.308151960. Books56000→60000+0.120791% is an open observation; config/cursor/LR checks pass. See `data/health/books-domain-regression-followup-20261123T0046.json`. Do not rewrite that review as PASS or assert noise/recovery without evidence.

Forecast whole-chain32.010days, remaining24.411days. Fixed64samples/domain are point estimates, not final capability. Frozen-budget JSON preparation flags are historical, not current run status.

Actual run: `data/runs/pretrain-final-1t-8gpu`, original attempt `c5bc01ad619700958f8fdac0967605f2`, identity85e119790f052669f2950847d93a749649eb21353e676fc01637320d7512521a, training sourceaf54b5e072aafbdae6033e04dc7434a4167ea1654d19b3882f08b8a2711dfc3a. Physical run_identity SHA8eeae4ac539490c01ed9987b027434fafc82d8f24df473efb3350a19cff05b5d.

Actual original bash430884, torchrun430905, workers430909..430916 remain live. Original outer430805/430858 and old auxiliary PIDs died bySIGKILL onNov18; sender unknown, cgroup oom_kill0. Guardian539905 explicitly adopted the same generation6/token, maintaining the still-live workers. Never launch a replacement training or mint a new formal generation while these workers run. Lease values stay in external live state, not public report code.

## Background helpers

Guardian source `/opt/yuformer-build/recover_live_pretrain_supervisors_20261118.py`; health `data/health/live-pretrain-supervisor-recovery-20261118.json`. It renews the existing formal generation and observes helper exits. Original monitor193642 remains alive. Helper PIDs at handoff:

- context_chain: 539979, source `/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer/eval/scripts/run_context_extension_pipeline.py`.
- eta_lr: 539911, source `/opt/yuformer-build/observe_total_runtime_and_actual_lr_20261115_v3.py`.
- long_lr: 539914, source `/opt/yuformer-build/observe_frozen_long_stage_lr_and_native_counters_20261115.py`.
- native: 539912, source `/opt/yuformer-build/watch_formal_native_checkpoint_states_20261115.py`.
- nine_domain: 539913, source `/opt/yuformer-build/watch_all_production_nine_domain_validation_20261115_v2.py`.
- prefetch: 539915, source `/opt/yuformer-build/prefetch_frozen_future_blocks_20261115.py`.

Preserve these exact original sources; `ops/active_sources/` contains audit copies, including a redacted recovery token, and is not a deployment replacement. Native observer restart checked_steps covers retained checkpoints; immutable historical admissions remain in `data/health/formal-native-checkpoint-state-admissions-20261115/` and the reviewv3 independently verifies the complete contiguous history.

The context controller `eval/scripts/run_context_extension_pipeline.py` uses the frozen114-file closure and `eval/configs/authorized_context_chain_20261115_v1.json`. It waits for actual full4K terminal, invokes finalcollector only after actual completion, then audits native full budget before16K. Actual paired4suite capability admission after16K is required for64K. Do not run finalcollector on live training or handwrite PASS.

## Frozen recipe and system qualifications

4K218453updates: warmup11921/50,000,297,984labels, stable170770/716,261,294,080, decay35762/149,996,699,648. Global4194304labels/update, eightGPU micro8/accum16; peakLR1e-4, endratio.1. Warmup→stable actual boundary verified. Data stage_sample_limits12207104/174868480/36620288 with B8 independent block permutation.

16K7152updates/29,997,662,208labels micro2/accum16;64K4768/19,998,441,472 micro1/accum8. Both HSDP2×4,25%short replay, peakLR2e-5→2e-6 warm72/48. Cross-stage weight-only, same-stage full-state resume. Full production windows and actual eightGPU transitions/bothSIGKILL resumes passed; future actual final capability remains unmeasured.

First250 native fullfinite9file18.88GB scan and independent copy passed; archive under `data/checkpoint_archives/pretrain-final-1t-8gpu-first250-20261115/verified_native_copy`. Periodic checks cover structure, six counters,164Adam/LR and8rankRNG; do not describe them as repeated full-floating scans. Frozen recipe inputs314/outputs7 and controller114 remain unchanged. Formal dataset/tokenizer/release/order inputs must not be retokenized or rebound for cleanup.

Use `/opt/yuformer-torchtitan/bin/python`3.12 for CPU audits with CUDA_VISIBLE_DEVICES=''. DefaultPython3.10 lacks hashlib.file_digest. Do not install dependencies in the live environment. Existing source-bound fullCPU suites and GPU qualifications remain valid; rerun only for relevant changes/failures.

## Next review and repository maintenance

Next review 2026-11-23T12:46:12.619001+00:00. CPU-only source `/opt/yuformer-build/review_pretrain_every12h_20261118_v3.py`. Run once when due, inspect real exceptions/domain regression, report briefly, then sleep returned wait_ms. Do not restart a full12h countdown after user input; calculate remaining time from `data/health/assistant-monitoring-cadence.json`. V3 requires actual original process startup times and full history, intentionally requests manual review at a phase/attempt change.

After a review, run `python tools/export_pretrain_report.py` and `MPLCONFIGDIR=/tmp/yuformer-report-matplotlib python tools/plot_pretrain_metrics.py`, inspect meaningful changes and publish if credentials permit. Use `tools/audit_repository.py` before commit. Current GitHub publication is blocked: SSH dhh1995 lacks write, connector Edenwu617 push=false, HTTPS helpergh missing. The local full snapshot commit/bundle status is recorded in data/health/repository-publication-20261123.json; check actual Git state before claiming publication. Keep pending credential request distinct from user authorization, which is already explicit.

Cleanup20261123 removed961files/13,661,559,135bytes from52obsolete caches/retired pre-calibration canary directories.391frozen dependencies unchanged; machine proof `report/metrics/cleanup_dependency_verification.json`, manifest `report/metrics/cleanup_20261123.json`. Training/weights/token streams/active releases untouched. Avoid broad deletion of data/ or all historical configs. Historical failure cases and referenced fixtures have a retention purpose.

Current docs: `report/TECHNICAL_REPORT.md`, `report/metrics/README.md`, `docs/engineering/PLAYBOOK.md`, `docs/engineering/INCIDENTS.md`, `docs/engineering/REPOSITORY_MAINTENANCE.md`. Registry has44historical+17recent cases; append new facts with evidence and unresolved causes, do not invent completed SFT/RL results. User wants transferable full-pipeline engineering experience across future projects.
