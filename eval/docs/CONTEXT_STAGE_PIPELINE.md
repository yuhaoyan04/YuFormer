# Production context stage measurements and progression

The formal eight-GPU 1.516600064B pretrain started on 2026-11-15 at
10:24 UTC. Its active model, backend, environment, data order and WSD schedule
remain frozen. The total allocation is 966254395392 actual labels, including
916258291712 at 4K, 29997662208 at 16K and 19998441472 at 64K. The user prefers
the complete chain within 33 days. Live runtime estimates are forecasts, not
a guarantee that future evaluations or interruptions have already occurred.

The controller configuration is `eval/configs/authorized_context_chain_20261115_v1.json`
(SHA-256 `2283492e473f6d1351838432f41d5dc7900a407064d0e2099f345dc9b9efd857`).
The live state is
`data/health/authorized-total-pretrain-context-chain-20261115-v1.json`.
At deployment it is `WAITING_FOR_ACTUAL_4K_COMPLETION`, PID 440338, without
creating a CUDA context or changing the running formal process.

The controller first requires the actual final 4K attempt and terminal marker
to agree. It then collects the completed run's TensorBoard result and audits
all nine final validation domains, exact approved budget, immutable run identity,
complete native DCP handoff and actual scheduler/data/token counters. A process
exit alone does not admit the parent. The existing qualified production stage
renderer and launcher execute 16K with the approved full window release and
model-only initialization. Optimizer, scheduler and data state start fresh for
that stage; same-stage recovery keeps the qualified full-state resume semantics.

After 16K completes, both its actual checkpoint and the actual complete 4K
baseline run the same frozen cases at the same evaluation context capacity.
Eight independent workers each see one explicit physical GPU. Receipts retain
actual GPU UUIDs, source and checkpoint hashes, exact case assignments and
measurement time. Reconstruction rejects missing workers, repeated devices,
another checkpoint, omitted/repeated cases, changed target masks and incorrect
UTF-8 denominators. Inference uses the qualified native model kernels and DCP;
it may change context capacity/recomputation, while preserving learned
architecture and RoPE. The complete model/backend source digest must equal
the source in the trained checkpoint identity.

Cases and quantitative policy are frozen in
`eval/results/frozen_context_stage_cases_20261115.json`
(SHA-256 `7db7573dfb9458d7ff86ee5692f445a04fe7dc93869f740b068cd3408c396dc6`).
The policy SHA-256 is
`8d27736e4d14bc72bb12e7879beb912988df9dd7f7523e98c598613a152561dd`.

| Suite | Fixed diagnostic cohort | Required paired behavior |
| --- | --- | --- |
| RULER | Original 13 tasks, 32 cases/task/context | Strict full-task macro improvement; non-needle macro does not regress |
| Long BPB | Eight actual raw heldout documents each in books/code/math | Strict equal-domain macro improvement; no domain regression; true UTF-8 byte denominators |
| Code retrieval | Scorable RepoQA v3, six languages × eight functions, context/no-context pairs | Context score and context-minus-control do not regress; at least one context function is recovered |
| Short core | Nine domains × 2816 unchanged 4096-target cases | Every domain NLL does not regress; original zero relative margin |

RULER uses the original case-match metrics and RepoQA retains its original
Tree-sitter function extraction and whitespace BLEU with smoothing method 4,
best annotated function matching and threshold 0.8. These are limited stage
diagnostics. RepoQA public repositories have not been certified absent from
training; report the no-context controls and do not claim clean benchmark data.
The code BPB domain includes repository text/documents under the original data
taxonomy; intrinsic BPB does not replace the distinct code retrieval task.

`admit_context_stage.py` writes the four actual paired decisions. A failed
suite preserves evidence and stops progression. Before 64K, `--verify`
recomputes the admission from actual measurement files, native final checkpoint
audits and frozen cases; a handwritten PASS cannot replace these inputs.
Only a real passing 16K admission enables the qualified 64K renderer. The final
64K checkpoint repeats the same complete measurement/admission process against
the actual 16K baseline. Future capability has not been measured at deployment.

The waiting controller holds a separate generation lease and renews it in the
foreground. It clears inherited formal cursor/attempt/lease variables from
child environments. It checks that all eight authorized GPUs are idle before
starting its own GPU work, tracks worker logs and stops its own descendants on
failure. Existing active 4K jobs are outside its process tree. Partial evaluation
attempts are preserved; a new attempt uses a new output directory. The immutable
configuration binds 114 source/assets files. Do not edit those runtime inputs
in place while this controller is armed.

Validation at deployment: the complete new evaluation CPU suite passed 60 tests
with one explicit isolated-parser-environment skip. The original parser test
was actually executed separately in `/opt/yuformer-repoqa-cpu`, and all 48 v3
reference functions independently passed their original scorer. A CPU-only
preflight in the actual PyTorch 2.14.0+cu126 environment confirmed the CUDA UUID
property, identical learned architecture at 16K/64K, the unchanged 35-file
training source closure and `cuda_initialized=false`. The pre-launch actual
16K native generation/NLL smoke and eight-GPU stage/resume qualification remain
separate system evidence. The newly assembled complete evaluation-worker and
capability chain will be exercised with real final checkpoints when they exist;
CPU tests and case audits do not constitute that future execution.
