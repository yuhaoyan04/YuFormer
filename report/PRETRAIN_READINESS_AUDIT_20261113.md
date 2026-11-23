# Pretrain readiness audit — 2026-11-13

> Subsequent user steering supersedes the four-GPU plan: production now uses eight GPUs with a 30-day target and 33-day launch ceiling. Four-GPU frozen evidence has been archived and marked STALE. See [the eight-GPU plan](EIGHT_GPU_PRETRAIN_PLAN_20261113.md) for the current configuration and gates. The PASS observations below describe the earlier audit iteration.

## Decision

**BLOCKED: do not start the formal 1T pretrain from this workspace state.** The
TorchTitan adapter and its frozen training arithmetic have no detected static
defects, but the exact 1T input streams and release audits are absent. The
configured TorchTitan Python and source checkout are also unavailable on the
current host. The production launcher will reject these missing inputs before
creating training workers.

The user authorization recorded in
`data/health/pretrain-user-full-pretrain-authorization.json` waives the failed
E1 scaling fit for this run. It does not waive the final data release, stream,
overlap, runtime, or production-contract gates.

## Verified contracts

- Production is pinned to TorchTitan v0.3.0 / FSDP2 and uses the stateful,
  pretokenized DCP dataloader. Training stream identity binds the tokenizer,
  data snapshot, release, and token-stream audit.
- The 1T plan is 50B warmup, 800B stable, and 150B anneal. At 4,194,304 tokens
  per optimizer update, the frozen config has 238,418 updates and stage limits
  `[12,207,104, 195,312,640, 36,620,288]` samples. Each limit aligns to the
  1,024-sample global update. The corresponding WSD boundaries are 11,921
  warmup, 190,735 stable, and 35,762 anneal updates. The TorchTitan scheduler
  qualification passed on 2026-11-12.
- The current mixture is a declared fixed nine-domain baseline, not a mixture
  selected by comparative E3 results. Its stable-stage weights are web 74%,
  code 12%, math 4%, Chinese 3.5%, academic 3.5%, books 0.4%, encyclopedia
  0.5%, multilingual 2%, and QA 0.1%. The E3 matrix/order/tokenization audits
  record eight arms and a 17-stream audit as PASS; the E3 selection itself is
  deferred. The frozen recipe records this evidence limitation explicitly.
- TorchTitan DCP/FSDP4, LR schedule, numerical parity, large-batch
  accumulation, training/save/load interruption recovery, and 4-to-2 resume
  qualification are recorded PASS on 2026-11-12. The dataloader restores a
  global consumed-sample cursor and checks it against trainer and scheduler
  state before the first resumed update.
  These are historical PASS records. The validator/model fixes below change
  the qualification identity; the current code requires a new GPU qualification.
- The pretrain model is configured for 4K context (`max_seq_len: 4096`). The
  4K→16K→64K length-extension ladder is a separate midtrain stage and remains
  blocked until its base checkpoint and long-context release are available.
- The frozen recipe and regenerated post-freeze static plan audit are PASS. The historical
  production canary passed the configured 75-day ceiling; its 4-GPU projection
  is about 60.3 days, so the stated 30-day objective is not met by that
  measurement.
  That canary is now stale and cannot authorize the changed training code.

## Blocking evidence from the current workspace

The live `audit_pretrain_stack.py` run on 2026-11-13 reports `BLOCKED` with
`defects=[]`. It found:

1. No `warmup`, `stable`, or `anneal` final token stream, metadata, or document
   length ledger under `data/token_streams/pretrain-1t/`.
2. No derived 1T release, full token-stream audit, or cross-stage overlap
   audit. The production contract is therefore absent. A direct contract
   validation attempt failed at the missing
   `data/processed/audits/pretrain_1t_overlap.json`.
3. The configured `/opt/yuformer-torchtitan/bin/python` and
   `/opt/torchtitan-v0.3.0` do not exist on the current host. The checked-in
   preflight report says `ready`, but its recorded paths are not usable here.
   The system Python is 3.10.12; it is not the locked training runtime.
4. The checked-in readiness snapshot is from 2026-11-06 and lists GPU 4,
   whereas the current host reports GPUs 0–7. It is stale and is not a valid
   current resource or readiness decision.

The processed-data gate in `data/PRETRAIN_DATA_READY.json` identifies the older
100B release. It does not establish that the final 1T stage orders and streams
are materialized.

## Audit fix

`pretrain/scripts/audit_pretrain_stack.py` now rejects a preflight report older
than 24 hours, a report whose runtime paths disagree with
`pretrain_base.env`, or runtime paths that are missing. This closes a stale
`status=ready` false positive while preserving the TorchTitan qualification
identity in the first iteration. Subsequent substantive fixes invalidate the old
qualification and canary evidence intentionally. Stale generated evidence is a
BLOCKED condition; both BLOCKED and FAIL prohibit a formal launch.

## Continued audit: defects fixed

1. **Ignored YAML options.** The formal validator now rejects unknown fields,
   including unsupported `resume`, `init_weights_from`, runtime sampling-policy
   changes, and curriculum flags. These previously looked configured but were
   not consumed by the TorchTitan adapter. Supported legacy optimizer metadata
   remains allowed explicitly.
2. **Ambiguous YAML values.** Duplicate keys in training or model YAML, fractional
   warmup/stage limits, string booleans and negative retention intervals now
   fail before a run identity is published. Three-stage WSD must align its first
   and last data limits with the LR warmup and decay boundaries.
3. **Model switches silently falling back.** Invalid selective-recompute modes,
   string-valued model booleans, non-positive dimensions, and non-finite RoPE or
   normalization parameters now fail in ModelConfig.
4. **Actual consumed mixture was unaudited.** The consumed-token overlap audit
   now counts domain tokens on the exact causal-target prefix, including partial
   first/last documents and excluding unconsumed headroom. It compares every
   stage with the frozen final-tokenizer domain weights. The plan requires both
   at most 0.002 absolute weight error and at most 10% relative error. Missing
   rare domains fail. Existing extraction caches without domain ledgers are
   rebuilt. This gate cannot pass until the final orders and lengths exist.
5. **Canary PASS survived source changes.** Canary evidence now binds the complete
   qualification identity, checks that identity before and after execution, and
   archives runs whose code/config/env inputs changed. Production validation and
   stack audit reject a canary without the current identity.

The real dataloader also has a CPU regression covering three-stage boundaries,
gradient accumulation, headroom exclusion, every optimizer-boundary resume,
2-to-1 data topology changes, terminal resume, and capacity exhaustion. Only the
TorchTitan abstract base is substituted; the actual adapter sampling code runs.

## Required gates before launch

1. Restore or provision the locked Python 3.12 / TorchTitan checkout at the
   paths declared in `pretrain/configs/pretrain_base.env`, then run a fresh
   environment preflight.
   Re-run numerical/recovery/reshard qualification and the production canary for
   the changed training source identity.
2. Materialize the frozen three-stage orders and token streams, including
   document-length ledgers; publish the derived 1T release.
3. Run the exact-roster stream audit and cross-stage consumed-token overlap
   audit against those outputs.
4. Re-run the production-contract validator and stack audit. Launch only when
   both report `PASS`/`READY` and bind the same frozen inputs.
5. Resolve whether the intended 30-day completion target is still required;
   the frozen 4-GPU measurement supports roughly 60.3 days under the
   authorized 75-day ceiling.

## Verification performed

- Final grouped CPU pretrain/data-release/performance-workflow regression —
  133 passed (one existing PyTorch storage deprecation warning).
- Model configuration and model behavior regression — 38 passed, 1 skipped.
- All 168 active formal YAML configurations passed the strict field-schema check.
- The post-freeze plan audit was regenerated and reports `PASS_STATIC`.
- `python3 pretrain/scripts/audit_pretrain_stack.py` — `BLOCKED`, zero static
  defects; blockers listed above.
- Final production-contract validation — `FAIL`: current numerical/recovery/
  reshard qualification is required following the source changes. The initial
  iteration reached and rejected the absent overlap audit.
- Machine-readable final reports: `report/pretrain_stack_audit_20261113.json`
  and `report/pretrain_production_contract_20261113.json`.
- No training process was started.
