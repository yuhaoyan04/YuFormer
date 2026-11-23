# Approved progressive-context production stages

The user approved approximately 1T total labels including context extension, prioritizing a 33-day total runtime and the 1.516600064B model. The measured execution allocation is 966,254,395,392 target labels. It preserves the approximately 50B warmup and 150B decay, and reduces only the 4K stable allocation to fit the measured whole-chain runtime. Counts refer to consumed training labels, including replay, rather than unique source tokens.

| Stage | Updates | Target labels | GPUs × microbatch × accumulation |
|---|---:|---:|---|
| 4K warmup | 11,921 | 50,000,297,984 | 8 × 8 × 16 |
| 4K stable | 170,770 | 716,261,294,080 | 8 × 8 × 16 |
| 4K decay | 35,762 | 149,996,699,648 | 8 × 8 × 16 |
| 16K | 7,152 | 29,997,662,208 | 8 × 2 × 16 |
| 64K | 4,768 | 19,998,441,472 | 8 × 1 × 8 |

Every update consumes exactly 4,194,304 target labels. The actual production canary forecasts the sum of all three context stages at 32.745 days with 90% duty. This estimate is not a completion guarantee; live throughput, checkpoint and validation time must be monitored.

The single 4K run retains its qualified WSD schedule and phase-specific admitted mixture/order. Start it only through `pretrain/scripts/run_authorized_pretrain.py --reuse-prepared-production`. Do not rebuild the original physical release or run generic recipe freezing.

For each new context, first require the actual completed native parent checkpoint, final accounting and finite final validation in all nine domains. Create a new immutable complete-byte handoff using `midtrain/scripts/create_checkpoint_handoff.py`. Never substitute the diagnostic step-200 or step-24 checkpoint for a completed production parent.

`experiments/e4_curriculum/scripts/prepare_production_context_stage.py` accepts the frozen stage plan and its SHA-256, the selected sequence length, actual parent handoff, complete admitted production window release, output configuration directory and new run directory. It performs the real midtrain validation before publishing config/env/receipt. The frozen plan SHA is `653249399c0d54b819b463a28fbdeebcede7244f8976963892bdbfae1ff2ccc4`.

Run the generated environment through `midtrain/scripts/launch.py --env <generated-env> --preflight-only`, then the same entry without `--preflight-only`. New contexts load weights only and reset optimizer, scheduler, RNG and the data cursor. Restarting the same sealed stage uses the same config and run directory and restores full native state.

Both long stages use 25% short replay, AdamW, peak LR 2e-5, cosine decay to 2e-6 and constant labels per update. Warmup is 72 updates at 16K and 48 at 64K. Save and validate every 128 updates; keep the last three checkpoints plus every 1,024th. The final step is saved and validated independently of cadence. The frozen model and runtime choices match the actual eight-GPU system qualifications.

Before progressing from completed 16K to 64K, require a bound actual long/short capability admission for that precise checkpoint and baseline. It must cover RULER, long-document BPB, long QA/code and all nine short-core domains. Pass its immutable file via `--evaluation-admission` when preparing 64K. No system test or two-case inference smoke can issue this capability PASS. The final 64K checkpoint also requires long/short capability evaluation before reporting the full chain complete.

Actual capability measurements necessarily follow training. A failed boundary gate stops progression while retaining the completed parent checkpoint and its evidence; it does not trigger automatic retraining of the completed 4K stage.
