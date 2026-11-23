# Authorized full pretrain execution — 2026-11-13

The user explicitly authorized completing all launch gates and then starting
the full eight-GPU 1T pretrain. The 30-day objective and 33-day projected
wall-time ceiling remain in force.

## Active execution

`pretrain/scripts/run_authorized_pretrain.py` is running as a detached process.
The current PID is recorded in `data/health/authorized-pretrain-launch.json`;
the authoritative live phase, child PID, commands, logs, and errors are in
`data/health/authorized-pretrain-pipeline.json`. A generation lease prevents
two instances from executing the workflow concurrently. Per-phase logs are
under `report/logs/authorized-pretrain/`.

Python 3.12.16 was installed under `/opt/yuformer-python`, and the exact
TorchTitan commit was fetched under `/opt/torchtitan-v0.3.0`. Pinned CUDA 12.6 /
PyTorch 2.14 userspace installation completed; preparation, independent
userspace validation, and fresh preflight all passed. Installation details are
in `report/logs/env-recovery-20261113.log`. A binary appearing during atomic
environment publication is insufficient: the worker waits for preparation
PASS and then independently validates userspace and executes fresh preflight.

## Gates executed in order

1. Restore locked runtime; validate userspace and current GPU preflight.
2. Run numerical, distributed, DCP, interruption recovery, and 4-to-8
   qualification on the eight authorized physical devices.
3. Measure FSDP8, HSDP2x4, and HSDP4x2 on the same 200-update contract. Evaluate
   steady p10 with the declared 90% training duty cycle. If none meets 33 days,
   stop for optimization. The 1T budget is not reduced.
4. For the selected topology, compare real eight-GPU chunked-loss accumulated
   gradients against a full-batch reference, with unequal valid-token counts.
   Tied and untied embeddings must both pass. Bind that report to the freeze.
5. Refresh experiment identity evidence, apply only the recorded historical
   source-drift/E1 waiver, and freeze the measured eight-GPU production recipe.
6. Requalify after the freeze because its production config changes identity;
   run the separate sustained production canary.
7. Materialize the three frozen orders, publish their derived release, tokenize
   them with document-length ledgers, and run full stream, overlap, and actual
   consumed-domain mixture audits. Refresh final plan closure against streams.
8. Require production contract PASS and stack audit explicitly READY; then
   invoke the official full pretrain launcher using `pretrain_base.env`.

GPU work waits for all eight authorized devices to be idle. Failed commands or
failed gates record BLOCKED and prevent subsequent phases. Existing stale
probe evidence is archived under the same generation lease protocol rather
than replacing evidence owned by another live job. A completed probe is reused
only when its immutable identity matches the current validated contract.

The initial readiness audit had 11 blocking records representing runtime,
qualification/throughput, freeze, and final-data dependencies. The full pretrain
has not started at the time this execution record was created. The worker is
authorized to launch it automatically once every required gate actually passes.

## Verification

219 grouped CPU regressions passed. New workflow tests verify that a BLOCKED
stack cannot trigger full launch, post-freeze qualification precedes production
canary, actual mixture audit precedes final contract, and a partially published
runtime cannot bypass environment validation. These CPU results do not
substitute for the actual GPU qualification below.

## Initial qualification and baseline probes (completed by 08:28 UTC)

The initial GPU qualification passed its 12 required stages: LR
schedule contract, numerical parity, distributed gradient parity, large-model
accumulation, single GPU, FSDP4 DCP, recovery reference, training/save/load
interrupt-and-resume, and 4-to-2 / 4-to-8 reshard. The maximum observed loss
differences for training/save interruption recovery were approximately
0.000146 / 0.000137; cursor checks passed.

| 200-update candidate | Functional result | Steady p10 tokens/s/device | Projected 1T days, including reserve | Runtime gate |
| --- | --- | ---: | ---: | --- |
| FSDP8 | PASS | 47,485.91 | 33.85 | FAIL |
| HSDP2x4 | PASS | 47,766.02 | 33.65 | FAIL |
| HSDP4x2 | PASS | 47,646.12 | 33.74 | FAIL |

The first two candidates completed validation and checkpoint saving and
destroyed all eight process groups normally. HSDP2x4's step-200 DCP metadata
exists. Current hardware observations show all eight devices occupied,
99–100% GPU utilization, no thermal slowdown flags, and the existing 700W
power limit. GPU utilization alone does not establish efficient model FLOPs.

The full 1T run has not started. All three candidates failed the unchanged
deadline gate. Throughput optimization and new measured evidence are required
before recipe freezing or full launch.

## Recovery gate correction at 08:19 UTC

Read-through of the qualification orchestrator found a false-positive saving
fault trigger: counting two `Staging the checkpoint.` log lines mistakes
repeated output from multiple ranks for a second asynchronous checkpoint.
The earlier save-interruption PASS cannot establish that writing was actually
interrupted.

The runner now waits for an explicit background-process save request for a
checkpoint newer than the latest complete one. After killing the owned
process tree, it verifies that the pending checkpoint did not publish
`.metadata`; a completion race fails the injection rather than certifying it.
The baseline and pending step, post-kill completeness, and original injection
log are retained with the result. Seventeen focused qualification/workflow
regressions passed, including repeated-rank logs, stale requests, and completion
races. A new real GPU qualification is required; the runner hash is already
part of qualification identity, so previous evidence cannot be reused.

An optional FlashAttention 2.8.3.post2 wheel is being compiled outside the
active environment, without GPU access and with four CPU build jobs. Its
verified sdist hash and build state are in `/opt/yuformer-build/`. The isolated
build uses C++20 for PyTorch 2.14 headers and SM90 only. It is not installed in
the training environment and has no measured performance or qualification yet.

## Further fixes and cold-cache checks through 09:16 UTC

Checkpoint retention now counts only directories with DCP `.metadata`. A
pending asynchronous write cannot evict a complete recovery point, including
when retention is one. Four regression cases reproduced the previous unsafe
behavior and passed after the fix. The corrected real save fault was observed
at pending step 8 with complete baseline step 4; pending `.metadata` was absent
after SIGKILL, all four cursor records matched, and step 5–24 recovery loss
differences were at most 0.000133.

A cuDNN SDPA diagnostic on the actual 8×24×4096×64 attention shape measured
about 12.8ms forward+backward versus 30.8ms for efficient SDPA. This is a kernel
measurement, not an end-to-end throughput claim. The first model attempt was
invalidated: the legacy `tests/verify` fixture silently overrode an empty
parent GPU mask and ran children on physical GPU 3 during broad CPU testing.
Ordinary pytest now skips that explicitly opted-in GPU integration directory;
direct child launch also checks opt-in and visibility, and its GPU wait
requires exclusive idleness. Regression coverage prevents hidden GPU children.

With a fresh compiler-cache directory, the earlier process-wide cuDNN toggle
then exposed a math fallback and first-step OOM on the model's strided Q/K/V.
The formal adapter now carries SDPA backend selection into the model config;
the model uses an explicit strict cuDNN context and contiguous Q/K/V. Global
fallback switches are no longer the implementation of backend selection.
The GPU parity gate additionally compiles the actual microbatch-8, 4K gated
GQA module and profiles forward and backward. Both cuDNN operators were
observed, outputs/gradients were finite, and peak allocated memory was
2,088,504,320 bytes. The ordinary mathematical parity cases also passed.

The restarted coordinator is rerunning all 13 current-source qualification
stages before a fresh 200-update cuDNN candidate. Full launch additionally
requires a new production recovery report, using the final model, microbatch,
compile settings, attention backend, and eight-GPU topology. Training, saving,
and loading faults must each match uninterrupted loss and per-rank cursors.
This bounded recovery test accumulates twice; the separate production canary
keeps the complete authorized 16-way accumulation and batch contract.

Data order/release/tokenization/audit subprocesses now use the locked Python
3.12 runtime: broad testing found the default Python 3.10 lacked tokenizers.
Development-only CPU dependencies live in `/opt/yuformer-cpu-tests`, separate
from the active training environment. Forty focused CPU regressions passed;
the expanded pretrain/E4/model suite is still being closed. Initial broad-run
failures included missing default-Python dependencies, GPU-only legacy tests,
and synthetic analysis fixtures reading live E1 replications or omitting
residuals required by their claimed analysis status. These findings are being
fixed without relaxing production gates. Full 1T training remains unstarted.

## CPU regression closure and optimized candidate at 09:41 UTC

The full pretrain/E4/model CPU run completed with **444 passed, 28 skipped,
zero failures/errors**. Its JUnit XML and source-bound summary are
`data/health/pretrain-cpu-regression.xml` / `.json`. Skips cover the explicitly
separate legacy GPU integration suite and unavailable FlashAttention GPU
parity. A watchdog fixture also needed isolated admission telemetry: its
synthetic `sleep` child must not wait for the actual eight training GPUs to
become idle. The real queue resource checks were preserved.

All 13 GPU qualification stages passed on the strict-backend implementation.
The current HSDP2x4/cuDNN candidate reached step 180/200 with approximately
74.5k tokens/s/device, 25.3% measured MFU, and 36.9GiB training allocation.
All eight ranks completed the step-100 DCP save, and post-save throughput
remained stable. Final p10/runtime evidence is still pending completion; this
is not yet a production runtime PASS or a full-pretrain launch.

The four generated production recovery contracts also passed the pinned
validator without publishing run identities, using the selected eight-GPU
topology, microbatch 8, two diagnostic accumulation steps, and cuDNN. Their
actual fault executions remain mandatory after the recipe freeze.


## Production qualification complete; data recovery at 10:45 UTC

The selected strict-cuDNN HSDP2x4 candidate completed 200 updates. The
independent frozen-production canary also completed 200 updates with the
full authorized microbatch 8 and gradient accumulation 16. Its p10 throughput
was **74,440.24 tokens/s/device**, giving **21.59464 days** for 1T tokens at a
0.9 training duty cycle. This estimate covers training and its runtime reserve;
full data preparation is still in progress. Neither diagnostic is full pretrain.
Evidence: `data/health/production-throughput-canary.json`.

Actual final-model, eight-rank recovery passed uninterrupted, training-fault,
saving-fault and loading-fault cases. All rank cursors matched; maximum loss
differences were respectively 0.001010, 0.002872 and 0.002563 for the fault
cases. Recovery uses two accumulation microbatches to bound diagnostic cost;
the separate throughput canary uses all 16. Selected-topology accumulated
gradients also matched the FP32 full-batch reference, including unequal valid
token counts and tied/untied embeddings. The 13 common GPU gates are PASS.

The CPU regression suite now reports **457 passed, 28 skipped, zero failures
or errors**. New coverage checks idempotent freeze/release publication and
outer pipeline evidence reuse, bounded tokenization batches, and native merge
with 1,050 parts under a 16-descriptor limit. Tokenization now partitions by
reference-token and document budgets, limits each encode batch, and caps
worker native thread pools. The merge closes descriptors after mmap and
initializes newly grown part-table entries. These changes address full-1T
memory, descriptor and uninitialized-memory failures before materialization.

The earlier S6 owner heartbeat stopped on November 12. After its existing
24-hour lease timeout expired, the sleeping local retry runner was restarted;
no lease files were removed and no active lease was stolen. The driver
acquired generation 8 normally at approximately 10:44:40 UTC. The restarted
coordinator reused the validated frozen recipe, common GPU qualification,
production recovery and throughput evidence, then resumed S6 order generation.
Full 1T training remains unstarted; order releases, token streams, complete
stream audit, consumed-prefix domain mixture and overlap, refreshed final
preflight, production contract and stack READY remain mandatory launch gates.


## Historical validation metadata restored at 10:53 UTC

Before parallel S6 collection, a proactive identity check found that the old
November 12 job had overwritten all nine validation summaries and the global
summary using the warmup training seed. Actual validation Parquet/ID binaries
still matched every original release hash. Current code already separates
validation seed from training order seed, but the stale on-disk metadata
would force avoidable recutting and cause a later upstream-identity rejection.

The owned local S6 writer was stopped before those writes. All four base and
curriculum configurations independently produced the same validation-config
hash. Applying that hash and the fixed validation seed reconstructed all ten
original metadata files byte-for-byte: both the base manifest and release
hashes matched before publication. Replaced metadata was archived; no data
binaries or original release identities were changed. All 19 upstream control
hashes then matched. Evidence is
`data/health/validation-metadata-restoration-20261113.json`.
The stopped owned generation-8 lease was explicitly released after confirming
its writer dead; the restarted driver acquired generation 9 normally. A new
real tiny-pool regression requires all validation artifact bytes and data
inodes to remain unchanged across base/warmup/stable/anneal order seeds.


The order runner now checks every base-release validation artifact size and
SHA before invoking S6. Same-length replacement and deletion regressions cover
summary, Parquet and ID files; the E4 plan/startup test file reports 28 PASS.
The live restored release passed this new guard. The active S6 driver confirmed
at 10:58:37 UTC that all nine validation domains were already cut and skipped
recutting. No qualified training backend or frozen training configuration
changed during this repair.


## Earlier mixture failure detection and baseline capacity binding

Each tokenized training stage now undergoes the same exact consumed-prefix
domain-mixture check before the next stage starts. This covers both newly
materialized and cached streams, hashes the document-length ledger, binds the
frozen recipe and published order stat identity, and uses the unchanged plan
absolute/relative tolerances. A failed warmup marks BLOCKED_MIXTURE and prevents
800B stable tokenization. Extracted consumed records share the final overlap
audit cache, so early checks avoid duplicate extraction. The complete final
stream/overlap/mixture audit remains mandatory.

A real CPU regression found the static plan test reading the mutable S6 global
summary during publication. Its capacity input now uses a small immutable
aggregate fixture derived from the original released snapshot. Production
static capacity auditing also now reads observed candidate token counts from
the baseline manifest, alongside that manifest's effective top-k; a later
mix's candidate counts cannot masquerade as the baseline quality band.
Regression coverage inflates the current ledger candidate counts and checks
that observed baseline capacity is unchanged. All 34 focused E4 tests pass;
the expanded CPU suite is running again. Training backend/configuration and
GPU qualification identities remain unchanged.


The complete expanded pretrain/E4/model CPU suite finished with **469 PASS,
28 skip, zero failures/errors** in 86.60 seconds. The separate S6/lease suite
remains 26 PASS. Source-bound XML/JSON summaries are current. Warmup's new
795,985,150-byte order table was written at 11:15:29 UTC; its final upstream
snapshot and publication checks remain in progress, so it is not yet counted
as a released stage.


At approximately 11:22 UTC the warmup order commit marker appeared. Published
member/order stat identities and the complete original upstream snapshot were
verified. It selects 42,417,762 documents and 50,796,892,962 reference tokens;
quota shortfall is 20,684 tokens versus the 1,000,000-token ceiling, with zero
collect errors. The driver is cleaning temporary parts before returning to
the three-stage runner. This is the selected reference-token capacity; actual
tokenizer counts and consumed domain mixture still require tokenization and
audit before training.


Warmup returned successfully at 11:23:10 UTC and the order runner recorded
PASS/rc=0. Stable began at 11:23:15 UTC under normally acquired S6 generation
10. No checkpoint or full 1T training process exists yet. The observational
monitor PID 108297 writes current state every 30 seconds to
`data/health/authorized-pretrain-live-monitor.json` and appends
`report/logs/authorized-pretrain-monitor.jsonl`; the dependency coordinator is
PID 106165. Monitoring does not launch or kill tasks.


## Exact accumulation recovery strengthening at 12:08 UTC

While stable order sorting runs on CPU, the otherwise idle eight GPUs now run
production recovery at the actual **16** accumulation microbatches. The earlier
A=2 report remains historical evidence; it was archived before replacement.
`contracts()` inherits accumulation directly from the frozen production base,
and the generated fault budgets/cursor expectations scale accordingly.
Eight focused tests cover A=1/2/16 contract generation and accumulation drift
rejection. Current recovery evidence must match this strengthened identity
before final production validation can pass. No training backend, production
model, frozen recipe or 200-step throughput canary identity changed.
The supplementary runner PID 114117 started at 12:08:46 UTC; launch evidence is
`data/health/production-recovery-full-accumulation-launch.json`. These diagnostic
runs are not the full 1T pretrain. Reference step 9 showed finite loss/gradient,
about 74.6k tokens/s/device, and 36.86GiB training allocation.


## Stable full-scale Arrow domain expansion failure at 12:12 UTC

Stable's first attempt completed candidate collection/sorting but failed while
constructing the member table: `ArrowInvalid: Take operation overflowed binary
array capacity`. This was not an OOM (container memory-limit failure count 0).
Doc IDs already use large-string offsets, but the domain DictionaryArray was
being expanded as one regular string array by `pa.table(..., schema=...)`.
At stable scale, repeated domain labels exceed that representation's capacity.

`_domain_array` now explicitly decodes bounded chunks, at most one million rows
and a 64MiB label-byte budget each. It returns a regular-string ChunkedArray;
published schema, labels, document ordering, quality selection and quotas are
unchanged. A small forced-budget regression checks UTF-8 byte bounds, chunk
boundaries, empty input and Parquet roundtrip. The full S6/lease suite is
**27 PASS**. The runner's normal retry will reuse matching candidate parts;
end-to-end stable publication remains pending and is not marked repaired by
unit tests alone. Completed warmup and the parallel A=16 GPU recovery continue.


At 12:27 UTC, exact-production A=16 recovery validated PASS for all four cases.
Fault maximum absolute loss deltas: {"loading": {"cursor_status": "PASS", "loss_max_abs_delta": 0.012196540832519531, "status": "PASS"}, "reference": {"cursor_status": null, "loss_max_abs_delta": null, "status": "PASS"}, "saving": {"cursor_status": "PASS", "loss_max_abs_delta": 0.006535530090332031, "status": "PASS"}, "training": {"cursor_status": "PASS", "loss_max_abs_delta": 0.00502777099609375, "status": "PASS"}}.
All fault cases verify eight rank cursors at the actual accumulation factor.
The separate real Arrow capacity canary reproduced the old overflow, then
verified 270,000,000 rows / 2,160,000,000 decoded domain bytes with the repaired
270-chunk path and a complete Parquet write/read/value check. Evidence:
`data/health/s6-domain-capacity-canary.json`. Stable attempt 2 reused all
12,391 candidate parts at 93.98 tasks/s and is redoing the merge; its final
publication remains pending.


Latest complete CPU regression after exact-accumulation recovery strengthening
is **472 PASS / 28 skip / zero failure or error**, 88.89 seconds. The separate
S6/lease suite is 27 PASS. The true >2GB capacity reproduction and complete
270-million-row Parquet roundtrip finished in 16.62 seconds, peak RSS about
3.37GiB. Both production fault-recovery and the 200-update production canary
remain current; stable full-scale retry is still merging and must publish
successfully before this data blocker is considered closed.


## Measured stable capacity repair at 13:18–13:30 UTC

Stable attempt 2 successfully wrote the 829,980,858-row member table (17 row
groups, 16.50GB) and 15.41GB order table, proving the Arrow domain repair at
actual scale. The unchanged final quota gate then rejected 340,562,499 missing
reference tokens and removed uncommitted outputs. Candidate-cache diagnosis
isolated multilingual: actual preferred-band capacity 18,815,681,098 versus
quota 19,156,214,478; the full governed domain has about 46.1B tokens. The old
nominal `top_k * pool_tokens` estimate overstates a document-quality band's
token capacity when high-quality documents are shorter.

`deepen_to_quota` now measures candidate tokens immediately after collection
and rescans only insufficient non-full domains. Its full-domain parts live in
a separate generation/threshold-bound `_measured_deepfill` cache, so narrow
parts cannot be mistaken for deep parts on restart. Unaffected domain caches
are preserved, and the final selection remains the exact best-quality prefix.
Strict top-k never deepens. Candidate capacity and selected complete-document
prefix shortfalls are checked before expensive member/order table construction;
failures record per-domain diagnostics. The original one-million-token limit
and final gate remain unchanged.

Thirty S6/lease regressions pass, including negatively correlated quality and
length, fallback cache interruption/reuse, strict-band failure before sorting,
and prefix shortfall before writing. The true >2GB capacity canary passes again
on the current source. The sleeping retry runner was restarted without changing
leases; coordinator PID 130434 resumed at 13:30:05 UTC, reused current frozen
recipe, all GPU qualification/recovery/throughput evidence, and completed
warmup output, then started stable attempt 3. Data publication remains pending.


At 13:41:49 UTC, actual measured deep fill completed all 29 multilingual shards
in about 19 seconds. It produced 31,767,405 candidates / 46,068,632,573 tokens,
exactly the original governed domain minus the immutable validation split,
versus quota 19,156,214,478. Thus capacity is proven before large merge work;
quality-prefix selection and final stage publication are still pending.
Evidence: `data/health/stable-measured-deepfill-verification.json`.


## Source document-index identity closure at 14:19 UTC

A deterministic two-row reproduction demonstrated silent wrong-text fetch:
rewrite one Parquet with reversed rows but identical size, then the old
path+size fingerprint reuses row coordinates and returns the other document.
This can preserve token counts and evade mixture-only checks.

The document index now uses a versioned complete source roster with size,
device, inode and nanosecond mtime; index generations have distinct array
filenames, a single writer lease, strict dtype/shape/stat checks, duplicate-ID
rejection, atomic files and a final per-fingerprint commit manifest. Legacy
cache files are left untouched. Source changes during building prevent commit.
Token part/preparation identity v5 additionally binds the source fingerprint.
Workers read source doc IDs and verify every selected row against the requested
order before encoding, bound local row indices, and check source identity
before/after reading. Source changes before final stream publication are
rejected. A versioned source-document binding is recorded in new stream meta.

Training stream cache reuse, early per-stage mixture preflight, the complete
1T audit, and the final eight-GPU production contract require that binding.
The audit also binds the shared validation helper hash. Existing immutable
released validation assets and GPU qualification inputs are not rewritten.
Targeted source/materialization/E4 tests: **65 PASS**; complete expanded CPU
suite: **489 PASS / 28 skip**, zero failures/errors (93.21 seconds).

A GPU-hidden index primer PID 132793 started at 14:19:23 UTC while S6 sorts;
it scans 12,829 S3 files, builds the fenced v2 cache, and compares the complete
legacy mapping when its old source roster is comparable. Evidence/log:
`data/health/source-document-index-build.json`,
`report/logs/source-document-index-build.log`.


## Stable publication and tokenizer input binding at 2026-11-13T14:49:01.958709+00:00

Stable attempt 3 committed at 14:40:54 UTC: 830,221,696 documents /
812,725,195,319 reference tokens; shortfall 31,730, below the unchanged
1,000,000 ceiling. Member/order artifact stat identities and original upstream snapshot
were independently checked; actual publication closes E-124 and E-125. Anneal is running.

A second deterministic reproduction showed that replacing tokenizer bytes during a
worker load then restoring the path can emit incorrect token IDs with unchanged token
counts. Parent/workers now construct tokenizers from exactly the bytes they hash; each
worker rejects bytes differing from the parent-bound SHA. Final publication also
rechecks the canonical tokenizer. Part/preparation identity v6 and source-document
binding schema 2 require verified source IDs and verified tokenizer bytes. Targeted
regression: 70 PASS, including transient replacement, rollback, interrupted index
publication, source mutation, and bounded parallel reader determinism. The full CPU
suite is rerunning. No formal 1T stream has been published and full training has not
started.

Auxiliary index primer restarted as PID 136690 at 14:48:08 UTC with eight bounded
readers (16-file prefetch, ordered consumption). This overlaps anneal; legacy index
files remain untouched. Producer implementation is held stable during this run.


Complete CPU regression finished exit 0: **494 PASS / 28 skip**, zero errors/failures, 120.22s. The first runner terminated with signal 11 while periodic faulthandler dumping was active; logs were archived under `report/history/cpu-runner-signal11-*`. The rerun removed only the external runner's periodic dump, kept fatal-signal diagnostics, and passed the identical suite. Crash attribution remains unconfirmed. Training source SHA is unchanged (`97d928b9db7f059e2f39712bc4cb47ccf47fc2c8ec34068532790bda4b028953`).


Persistent observational monitor extended at 14:57 UTC (PID 138899), without restarting the pipeline: source-index process liveness/CPU/RSS/read counters and unexpected-exit flag, plus cgroup-v1 memory usage/headroom/allocation-failure counters, are written every 30 seconds into the existing live-monitor JSON and JSONL history. No orchestration or admission gates changed. At 2026-11-13T15:01:26+00:00, status RUNNING, source index index 4200/12829 shards, 740s, anneal merge active, formal pretrain not yet launched.


At 2026-11-13T15:15:55.937622+00:00 all three S6 stages are actually PASS (exit 0). Anneal published 123,742,963 documents / 152,630,432,898 reference tokens, shortfall 200,738 < unchanged 1M; member/order stat identities and exact original upstream snapshot independently verified. Its final post-commit NFS work cleanup completed normally. Main coordinator advanced to `release`, constructing and auditing the frozen 1T order release. Source index primer remains RUNNING with eight bounded readers. Full pretrain has not started. Evidence: `data/health/anneal-order-publication-verification.json`, `data/health/pretrain-1t-order-materialization.json`.


Frozen 1T order release actually PASS at 15:18 UTC: three stages and 40 artifacts, complete order content/hash/stat checks. Coordinator automatically started tokenization PID 139585 at 15:18:46 UTC with 128 workers. Storage preflight PASS: ~349TB free vs ~5.83TB required peak including margin. Warmup tokenization is first; it waits for the fenced source-index build and will reuse its committed result. No formal training launched. Release SHA256: `3be8f450d6ebbe2312f9b18c67ed57ebaca4f7af8b658e58952875099d083f00`.


At 15:54 UTC the new source-index primer completed scanning all 12,829 source files and entered global array concatenation/stable sorting. The legacy full index has 6,485,760,234 rows. A separate 16M-row random-V16 stable-sort timing took 6.64 seconds; N log N extrapolation is ~61 minutes for that row count, only an approximate operational estimate (NUMA/working-set/contention can change it). At 16:22 UTC index CPU remains active, RSS ~346GiB, cgroup headroom ~514GiB and allocation failures zero. Source-index generation 2 remains owned by PID 136690, heartbeat verified live during native sort. Warmup tokenization is correctly waiting for the single index writer. Evidence: `data/health/source-index-sort-timing.json`. Full pretrain not launched.


At ~17:20 UTC the source index committed generation v2 (`2f1027aa040e1ae9…`): 6,485,760,234 unique document IDs, 12,829 files, three arrays totaling ~145GiB. Global stable sort, duplicate-ID rejection and final source-roster recheck completed. Independent array stat and exact size checks match the manifest. Producer SHA remains `e9718d5a0d6b159ac6f2b80d203dbcd1b20236390a71db570183afd8c5e284f9`. Warmup automatically acquired and reused this committed cache and began preparing 42,417,762 documents. The auxiliary primer continues its optional complete legacy-mapping comparison; that comparison has not yet passed. Warmup preparation is reading index pages over NFS, as verified by increasing physical-read counters and `wait_on_page_bit_common`; host available memory is ample and cgroup allocation failures remain zero. Full pretrain not launched. Evidence: `data/health/source-index-commit-verification.json`.


## Open-file order binding repair at 17:45 UTC

A new deterministic regression reproduced an atomic order-path rollback gap: replace the order only during `read_table(path)`, then restore the original inode/mtime/bytes. The previous producer emitted reversed token sequences while retaining the verified order SHA and token counts. The new loader hashes and parses the same opened inode, checks fd/path identity after reading, honors published stat identity where present, and checks the order again before publishing. Order bytes in metadata come from the bound input. Cache version v7 and source-document binding schema 3 require verified order, tokenizer bytes and source IDs. Targeted regression: 78 PASS; full CPU suite is rerunning.

Owned token controller 139585 and auxiliary legacy comparator 136690 were stopped before warmup encoding; the coordinator recorded the expected SIGTERM blockage. All three S6 stages, frozen release and the committed 145GiB v2 source index are preserved. The index algorithm and its fingerprint are unchanged; the restart reuses its cache. Previous comparator launch/report were archived under `report/history/before-order-input-binding-20261113T174515Z`.

The source-change fixture was intermittently failing because sub-millisecond in-place rewrites can retain identical stat timestamps: 97/100 measured trials. Its helper now explicitly advances mtime only when needed so it deterministically represents an input-identity change. Production S3 materialization uses `BatchWriter.close()` with temporary-file `os.replace`, which changes the inode. Production identity checks were not weakened. Evidence: `data/health/source-fixture-timestamp-resolution.json`.


Order-binding fix complete CPU regression: **502 PASS / 28 skip**, zero failures/errors, 106.147 seconds. Training-core SHA unchanged; production GPU numerical/recovery/throughput evidence remains current. Producer is held stable again after this check.


## Source-handle binding follows the complete input review

The same rollback reproduction was applied to worker source reads. Temporarily swap a source Parquet only while opening it, keep the same doc IDs with different text, then restore the original inode/mtime: the prior worker accepted altered text tokens with unchanged counts. Worker reads now open a file, verify the actual fd identity against the bound roster, parse that same handle and check fstat again after reading. Existing per-ID and final path checks remain. Part/preparation cache v8 and source binding schema 4 additionally require `source_handles_verified`. The producer implementation SHA is captured before order loading/long index preparation. Source-index generation v2 is unchanged.

All 82 targeted tests PASS, including source replacement between path check and open and during parser construction. The owned controller/comparator were stopped before worker encoding for this repair; existing S6/release/source index artifacts remain reusable. Full CPU suite is rerunning. The final repair supersedes v7/schema-3 details above. Evidence: `data/health/source-handle-race-reproduction.json`.


## Runtime mapping identity closes the final consumption boundary

The same controlled path rollback reproduced wrong training labels in the TorchTitan reader: expected `[1,2,3,4]`, observed `[101,102,103,104]` after edge-hash validation succeeded. The reader now checks the published file identity, hashes edges and creates mmap from the same open fd, then rechecks fd/path identity. Newly materialized metadata binds the final staging inode/device/mtime/size (preserved by atomic rename); formal 1T stream audit and cache reuse require the binding. A second regression replaces only interior tokens while keeping both 1MiB edges identical and proves published identity rejects it. Old immutable qualification assets may omit the newer metadata field, but still use the same-fd edge/map path.

Targeted regression: **86 PASS**. Training-core source changed from `97d928b9…` to `b9b68ddc8c963da83c77f45672a71a79039cbe91c13aebe97b56b8a22b1bf1d1`; all old eight-GPU numerical/recovery/performance evidence is historical for the prior source and must be regenerated. CPU suite is rerunning before the authorized coordinator restarts GPU qualification. No formal pretrain checkpoints exist. S6 data and v2 source index remain reusable; input producer is v8 / source binding schema 4. Evidence: `data/health/runtime-stream-binding-requalification.json`.


Final full CPU regression after runtime FD mapping fix: **509 PASS / 28 skip**, zero errors/failures. Old GPU/recipe evidence archived under `report/history/before-runtime-stream-binding-20261113T181349Z`; current core SHA `b9b68ddc8c963da83c77f45672a71a79039cbe91c13aebe97b56b8a22b1bf1d1` requires fresh GPU qualification. Source-index comparator restarted as PID 150471 against the same committed v2 cache. Authorized coordinator is restarting with the selected strict-cuDNN eight-GPU configuration and will rebuild all source-bound gates before tokenization/training.


At ~18:23 UTC all **13 common GPU gates PASS** on the new runtime-stream-binding core (`b9b68ddc…`), independently checked against the current complete qualification identity. Includes numerical/gradient/backend parity, LR, large-model accumulation, DCP fault recovery, 4-to-2 and 4-to-8 reshard. Coordinator advanced to the selected HSDP 2x4 strict-cuDNN eight-GPU probe. Fresh production A=16 recovery and 200-update production throughput are still pending.


At 2026-11-13T18:32:35.345864+00:00 source-index comparator completed **PASS**: all 6,485,760,234 doc IDs, file indices and row indices exactly match the comparable legacy cache over 12,829 source files. Current v2 roster fingerprint is unchanged (`2f1027aa…`). No existing mapping drift was found. The new source-bound index and full comparison remain valid on producer `9a9e06221f1d347a0112c459c910ebdda2a15d19ef547ebda5364162a1ffb5e5`. This establishes mapping integrity; new worker/stream FD checks additionally protect future input replacement. Evidence: `data/health/source-document-index-build.json`, `data/health/source-index-commit-verification.json`.


At ~18:48 UTC the new-core selected strict-cuDNN HSDP 2x4 production-shape probe completed **200 updates, accumulation 16**, PASS; fresh selected-topology eight-GPU gradient parity also PASS. Probe p10 74384.84375 tokens/s/device, median 74510.9375, median MFU 25.34933376312256%; 1T forecast 21.61072346227579 days with unchanged duty 0.9, below 33-day ceiling. Recipe was re-frozen on the current source after a source-drift-only experiment audit under the existing authorization; qualification after freeze PASS. Fresh production interruption recovery is RUNNING; independent 200-update production canary remains pending.


At ~19:07 UTC fresh new-core **production A=16 recovery PASS**: reference, training kill/resume, saving kill/resume and loading kill/resume all completed successfully. Report independently validated against the current production contract/source. The coordinator advanced to the independent 200-update frozen-production throughput canary. Full pretrain not launched; data stream generation/audits remain pending.


At 2026-11-13T19:30:56.469368+00:00 the independent new-core frozen-production **200-update canary PASS**. Report independently matches current complete qualification identity, runner/config/env hashes and frozen recipe. p10 74370.171875 tokens/s/device, median 74493.8515625, MFU 25.343520164489746%, 1T forecast 21.614986862847605 days at duty 0.9. All GPU requalification after the FD fixes is complete. Coordinator reused all three S6 stages and advanced through release verification toward tokenization; no formal pretrain launched.


At ~20:09 UTC warmup tokenization is RUNNING with 128 workers: 1,040/1,774 completed parts, 22.182B encoded tokens, ~11.9M tokens/s. All source-ID, source-fd, order-fd and tokenizer-byte checks remain active; no input identity failure occurred. Cgroup memory reached its hard accounting limit through file cache, while anonymous RSS is ~77GiB and ~1.2TiB inactive file cache is reclaimable. Allocation failures, under-OOM and OOM-kill counters remain zero; progress continues under kernel cache reclamation. Warmup publication/actual consumed-mixture gate and stable/anneal streams are still pending. Evidence: `data/health/pretrain-tokenization-runtime-observation.json`.


At ~20:50 UTC complete warmup part measurement exposed E-131 before stable tokenization or formal training: 49,667,693,072 actual EOD-inclusive tokens versus 50,000,297,985 required (shortfall 332,604,913). All-stream web weight is 0.5956085860906843 versus 0.6000319942410366, failing the existing 0.002 absolute tolerance. The producer source binding and source index remain valid; the defect is use of held-out validation tokenizer ratios to budget selected training quality bands. Retaining the original merged stream/parts for exact-ledger verification. Phase-specific training calibration is running with 100M reference tokens/domain/phase; 22 focused regressions PASS. Frozen recipe and recipe-bound throughput canary must be regenerated after calibration; no numerical, recovery, mixture or 33-day gates are waived. An NFS hard-link restriction in calibration archival was handled with an atomic verified copy fallback; original orders remain untouched.


At ~21:08 UTC the original warmup stream committed and its final per-document length ledger exactly matched the independently reconstructed part ledger (SHA-256 `8c6741b444cff30dee63921f40aa58920b032a22be59c19abb7172dc6b2af73a`). The original coordinator stopped before stable with `stream has insufficient tokens for 50000297984 targets`; no formal training started. The inner tokenization controller now records `BLOCKED_PREFLIGHT` and the error/report instead of leaving a materialization PASS as its final job state. Capacity/ledger/mixture failures on cached and freshly materialized streams are covered by 10 passing tests. Phase-specific calibration/freezer/coordinator tests pass 38 cases; calibration is integrated before production freezing, with valid evidence reuse and fail-closed remeasurement. Warmup calibration completed 641,105 documents / 878,896,128 actual tokens; comparing its domain ratios with the complete 49.667B stream gives maximum normalized domain-weight error 0.000586738246, below half of the unchanged formal absolute tolerance. The corrected warmup reference budget is provisionally 52.092B and crosschecks to about 50.931B actual tokens; these are estimates pending corrected full-stream gates. Stable calibration is now encoding 705,754 documents in 107 chunks. The handoff process will revalidate all measurements against current tools, archive the insufficient original stream, re-freeze, run the full CPU suite and restart the authorized eight-GPU chain.


At ~21:26 UTC cgroup-v1 `memory.failcnt` increased from 0 to 1 during stable pilot encoding. Kernel documentation defines this as limit-counter hits followed by reclaim, rather than evidence of a process allocation failure. The external monitor's historical JSON field `allocation_failures` was mislabeled; the live presentation now says `memory_limit_hits`. `under_oom=0` and `oom_kill=0`; anonymous RSS is ~39GiB while more than 1TiB of inactive file cache remains reclaimable, and encoding continues. No cgroup counters were reset and no cache was forcibly purged. Reference: https://kernel.org/doc/html/v6.6/admin-guide/cgroup-v1/memory.html#failcnt .


At ~21:53 UTC E-131 implementation and full CPU regression passed: **532 PASS / 28 skip**, no failures/errors. Phase-specific calibration measured actual EOD-inclusive totals warmup=878,896,128, stable=879,678,165, anneal=874,670,399 over at least 100M reference tokens/domain/phase. The final frozen recipe SHA is `11326994dbdd0b1fc2d3cfdb1e39bb580468c1994320c8c6f8f06a6b826f7692`; reference selection budgets are warmup=52,092,016,778, stable=820,447,760,213, anneal=155,550,771,759. Actual training budgets remain 50B/800B/150B, 2% headroom, WSD steps 11,921/190,735/35,762 and 4,194,304 tokens/update. Base YAML/env are byte-identical to the previous authorized recipe. System PyArrow 12 compatibility was validated over all real measurement ledgers after using Array.to_numpy following combine_chunks. Old release/plan/canary evidence was archived rather than rebinding its hashes; the first full-suite run correctly rejected those stale active artifacts. The original insufficient full stream is retained under data/token_streams/pretrain-1t/_history/e131-original-8069ab2b. Authorized coordinator PID 186272 resumed; common GPU qualification and production recovery independently validate against current inputs. A genuinely fresh eight-GPU 200-update production canary is RUNNING; all eight GPUs showed 100% utilization. Corrected full orders/streams and final production gates remain pending, so formal pretrain has not launched.


At ~22:17 UTC the fresh recipe-bound eight-GPU canary completed **200 updates PASS**. Independent validation confirmed current source qualification identity, frozen recipe `11326994…`, runner/config/env hashes and collected result. p10 throughput **73,897 tokens/s/device**, median **73,933.3125**, MFU **25.152820587158203%**, planned duty 0.9; projected 1T training **21.75339036856494 days**, below the unchanged 33-day ceiling and 30-day target. The step-100 checkpoint was committed and training continued to step 200. Coordinator PID 186272 advanced to corrected S6 order generation; warmup holds source lease generation 14, PID 188592, and is validating upstream gates. Prior stable/anneal job PASS entries are historical and have not yet been revalidated for the new quotas; the external live view labels them accordingly. Read-only actual capacity-plan preview also PASS_STATIC with zero errors; full-tokenizer closure remains pending. No formal training has launched.


At ~22:28 UTC independent reference/final tokenizer provenance confirmed S5 input_contract.tokenizer_sha256=`8d62190be601021899b94757663dfd3e1705fa0615903ce93c155cd301b665f6` matches tokenizer/tokenizers/yf-64k-bpe/tokenizer.json. Formal M3 tokenizer SHA is `5dd6bb3f9f587c301e2bb6bb8c987b725fa04d5c49230849038c7d7ec8c5cf9e`. Thus S5 n_tokens is an exact count in the old reference tokenizer, while formal consumption budgets must use actual M3 + EOD counts. The observed domain/quality-dependent ratios are a conversion between distinct tokenizers, consistent with the phase-specific correction. No reference counts, tokenizer assets or original S5 records were modified. Evidence: data/health/reference-final-tokenizer-provenance.json.


At ~23:04 UTC corrected warmup S6 exited PASS after commit/cleanup. Verified 43,669,607 docs and 52,091,983,950 reference tokens; quota shortfall 32,828 < 1M, upstream snapshot unchanged `8a243b08…`. Stable began as PID 190540 at 23:04:55 under the unchanged production S6 source `3d94d17d…`; by ~23:22 all 12,391 collect tasks completed and measured multilingual deep fill automatically rescanned 29 shards (preferred candidates 18,815,681,098 versus quota 19,789,601,438 reference tokens).

Cleanup diagnosis exposed E-132: final-domain NPZ archives were never closed, leaving thousands of live handles and NFS .nfs silly-renamed candidates until function exit. Staged fix source SHA `b44ab9098e67160781b94cca65ecafe24399695dda2d2e10f95c9a9f8f7b076a` preallocates arrays and closes each archive immediately after copying. Bitwise array comparison and a real 32-FD/96-part reproduction pass; full S6 source/caching/lease/resource regression 38 PASS. The live stable producer remains untouched/running. Only parent order controller PID 188527 is held with SIGSTOP (it owns no lease); coordinator PID 186272 and stable source lease heartbeat remain live. Automatic boundary service PID 192511 will wait for stable commit and normal lease release, verify the reviewed source hash, apply/copy tests, run actual-source 38 cases plus the full CPU suite, and SIGCONT the parent before anneal imports source. Pending hold/resume state is persisted in data/health/s6-fd-fix-phase-switch.json. Production source is still `3d94d17d…` until that boundary; no in-flight producer source bytes were replaced. No formal pretrain has launched.


At ~23:32 UTC the E-132 staged source also passed a **real NFS** canary on the same filesystem device 186 as production data. Deleting a deliberately open legacy NPZ reproduced .nfs silly rename. The bounded loader merged 64 parts under RLIMIT_NOFILE=32; process FD count remained exactly 4 before/after, and all 64 candidates plus the legacy test file were removable with zero .nfs residues. Evidence: data/health/s6-fd-real-nfs-canary.json. Source remains staged until stable producer exits; boundary service 192511 is active. Stable sorting is progressing at ~70GiB anonymous memory, with OOM counters zero.


2026-11-14 ~00:26 UTC corrected stable reference-order publication independently verified PASS: **839,520,433 documents**, requested reference tokens **820,447,760,213**, selected **820,447,704,928**, tail **55,285 < 1,000,000**. New members file is 16,686,714,316 bytes, new stage2 order 15,583,543,842 bytes; row counts match. Manifest SHA `a2325d029f1783aa9e30a27d732343fe76c295744e49a33b5bcef235f1144440`, upstream snapshot remains `8a243b08…`. Actual M3 stream capacity and consumption-mixture gates remain pending tokenization. Old-source stable PID 190540 is cleaning/exiting normally; order controller 188527 remains held until verified source-update service 192511 completes actual-source S6 and full CPU regression. Persistent observer was upgraded to schema 2 as PID 193642 at 00:13 UTC; memory.failcnt is now named memory_limit_hits and the FD-fix service liveness is persisted. No coordinator or active S6 worker was interrupted by that observer change. No formal pretrain has started.


2026-11-14 ~00:33 UTC E-132 safe-boundary update completed PASS. Stable producer exited normally and released the stage lease; reviewed source `b44ab9098e67160781b94cca65ecafe24399695dda2d2e10f95c9a9f8f7b076a` was atomically installed only after exit. Actual-source S6/caching/lease/FD regression **38 PASS**, full CPU suite **532 PASS / 28 skip**, no failures/errors. Parent controller 188527 received SIGCONT automatically, hold state is RESUMED_AFTER_VERIFIED_FIX, and anneal started as PID 197547. Current frozen recipe and eight-GPU canary bindings independently revalidated PASS; model/training core remains unchanged. Real NFS canary proves exact FD count 4 before/after under limit 32 and no residual .nfs files. Original source and CPU evidence preserved in the boundary service history. Source fix is applied and regression-verified; production anneal merge/cleanup observation and all actual M3 token-stream gates remain pending. No formal pretrain has launched.


2026-11-14 ~01:02 UTC corrected anneal reference-order publication independently verified PASS: **126,683,145 documents**, requested reference tokens **155,550,771,759**, selected **155,550,703,026**, tail **68,733 < 1,000,000**. Members/order Parquet row counts match; upstream snapshot remains `8a243b08…`. Manifest SHA `7bf45ff93afac15045b775727b6b73c95474a16e210c1ff215f088a9b0b49567`. The new bounded candidate loader completed the actual anneal merge with only 12–14 observed open FDs, compared with 8,738 in the old stable process. All three corrected reference orders are now published; controller cleanup/exit and derived release handoff are monitored. Actual M3 stream capacity and mixture gates remain pending. No formal pretrain has started.


2026-11-14 ~01:06 UTC all three corrected S6 jobs exited PASS and the derived release independently published PASS: `yuformer-data-86f5860c47e6162e6ba0`, frozen recipe SHA `11326994dbdd0b1fc2d3cfdb1e39bb580468c1994320c8c6f8f06a6b826f7692`, 40 hash-bound artifacts. Coordinator automatically entered tokens at 01:05:39; corrected warmup actual-M3 encoding is RUNNING, stable/anneal PENDING. Warmup job retains historical previous-attempt preflight fields until a new gate result is written; those timestamps are not current evidence. Actual stream capacity/mixture must pass before stable begins. No formal pretrain has started.


2026-11-14 ~01:45 UTC corrected warmup preparation completed; 128 encoding workers are processing 1,779 bounded tasks under prep key `055137c245e2c3b3`. At 220/1,779 tasks, cumulative actual output was ~7.564B tokens at ~19.0M tokens/s. Main process holds 269 bounded process-pool descriptors; observed individual worker maximum 16. Full preparation read ~128.49GiB from NFS, with advancing CPU/fault counters throughout. Kernel memory-limit hits increased from 3 to 4 under file-cache reclamation; under_oom=0 and oom_kill=0. Actual warmup stream capacity/consumed mixture checks and formal launch remain pending.


2026-11-14 ~02:43 UTC user requested an empirical 1B–5B eight-GPU model-size sweep while CPU data preparation continues. Independent runner PID 203151, health `data/health/model-size-sweep.json`; production model/config/frozen recipe remain unchanged pending measured choice. Candidate parameter counts measured on meta tensors: 530,107,904 baseline; 994,652,416 wide 1B; 1,002,005,248 deep 1B; 1,516,600,064; 1,966,208,512; 2,985,462,016; 4,009,568,256; 4,956,593,664. Equal 4K/BF16/AdamW/HSDP2×4 and 4,194,304 tokens/update, 24 updates, first 8 excluded, p10 over 16 points, 90% training duty. Individual run identity/byte hashes and final nine-domain validation are required. First launch failed before GPU use because save_every=0 violated the existing launcher contract; original configs/results archived under report/history/model-size-sweep-save-interval-20261114T023034Z, corrected to final-step-only checkpoint saving, then baseline and 5B contract previews PASS and 8 evaluation regressions PASS. Current concurrent baseline p10=74,767.578125 tokens/s/device →21.5001 days/1T; wide 994.65M p10=48,307.8203125 →33.2764 days, thus outside both 30-day target and 33-day launch ceiling. Deep 1B is RUNNING. This is screening evidence, not production qualification or a model-quality comparison.

Corrected warmup encoding finished all 1,779 parts with rounded 50.939B actual EOD-inclusive tokens and entered native position-order merge around 02:32 UTC. Preliminary capacity exceeds required 50,000,297,985; exact final ledger and consumed-mixture gate remain pending. Main pipeline PID 186272 and token producer 199471 remain live; no formal pretrain has started.


2026-11-14 ~03:14 UTC corrected warmup full-stream publication and consumed-mixture gate **PASS**. Exact actual tokens **50,939,497,047**, docs **43,669,607**; required causal stream **50,000,297,985**, surplus **939,199,062**. Published token SHA `108cfe616dc8a1e7cc94b3591c81637a8aeeb1f802a0ca059c2246ee93134394`; ledger SHA `794f9fea7f55690cbca757fd759297d0ad271f826609b56d95483fae00c2d997`. All four source-document binding flags true, M3 tokenizer/source roster/frozen/release identities current. A distinct open-FD-bound Arrow/NumPy recount independently recomputed every consumed domain token over exact causal interval [1, 50,000,297,985) and matched the preflight counters exactly; full ledger SHA and sum rechecked. Maximum absolute domain error **0.0014967086203835356** < unchanged **0.002**, all nine domains pass. Independent proof: `data/health/e131-corrected-warmup-independent-mixture.json`. Stable automatically started at 03:12:26 as producer PID **214752**; its 800B stream and anneal 150B gates remain pending. E-131 warmup failure is fixed and verified; whole-issue closure still requires stable/anneal. No formal pretrain has started.


2026-11-14 ~03:42 UTC user explicitly selected preference “允许接近 30 天，优先选择更大模型”. Recorded `data/health/model-scale-user-preference.json`; prioritize largest measured architecture within 30 training days at 1T/8 GPUs/duty 0.9, without a 10% slowdown requirement. No formal authorization gate is weakened. Main sweep now reached 5B. Confirmed p10/day pairs: 0.530B 74,767.58/21.50; 0.995B wide 48,307.82/33.28; 1.002B deep 44,472.49/36.15; 1.517B 35,592.68/45.16; 1.966B micro4 26,065.54/61.67; 2.985B micro4 18,603.85/86.41. 4.010B micro4 completed but p10 9,388.04 is allocator-thrashing evidence and excluded pending smaller-batch comparison. Frequent warning lines initially hid progress from a 64KiB tail; read-only monitor tail enlarged to 1MiB and latest actual step recovered, avoiding a false stall diagnosis. No running benchmark or data producer was killed for that observation issue.

Queued follow-up supervisor PID 218118 waits for main sweep completion, then remeasures baseline and ~1B width/depth choices (14L/2304, 12L/2560, 10L/2816, 8L/3072), followed by explicit microbatch2/accum64 4B/5B controls. Only prior idle queued supervisor 209397 was replaced to add controls, with its source/plan/state archived. Main supervisor 203151 and data coordinator 186272 were untouched. New analyzer rejects candidates with allocator OOM retries after warmup until smaller-batch comparison; 11 throughput/accounting/pressure regressions PASS. Analysis sidecar PID 218301 refreshes `data/health/model-scale-sweet-point.json` every 30s; independent follow-up health is `data/health/model-width-followup.json`. No model quality claim or production qualification is inferred from screening.
