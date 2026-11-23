# Eight-GPU pretrain plan — 2026-11-13

The user now requests all eight H800 GPUs and completion around 30 days.
The production YAML and env declare eight ranks, microbatch 8, and accumulation
16. This preserves 4,194,304 tokens/update, LR 2e-4, 238,418 updates, and the
existing WSD sample/step boundaries. The 1T budget and model shape are preserved.
A new output directory, `data/runs/pretrain-final-1t-8gpu`, prevents accidental
reuse of a four-GPU run identity.

## Runtime requirement

| Quantity | Requirement or estimate |
|---|---:|
| Target pretrain wall time | 30 days |
| Maximum projected wall time allowed at launch | 33 days |
| Planned fraction of wall time spent training | 90% |
| Required effective aggregate throughput for 30 days | 385,802 tokens/s |
| Required steady per-GPU throughput with reserve | 53,584 tokens/s |
| Historical four-GPU per-device p10 | 47,968 tokens/s |
| Ideal eight-GPU training time at that historical rate | 30.16 days |
| Estimate including reserve at that rate | 33.51 days — fails the new ceiling |

The historical rate does not establish eight-GPU scaling. Production canary
evaluation now uses its recorded world size, p10 throughput, and the explicit
90% training duty cycle. Four-GPU results cannot pass the eight-GPU gate.
The watcher and production contract use the same 33-day limit. The previous
75-day allowance is superseded.

## Prepared topology comparisons

Three 200-update contracts under `pretrain/configs/eight_gpu_probe/` preserve
the same model, tokenizer, sequence length, tokens/update, and optimizer:

- `eight_gpu_fsdp8`: FSDP across eight ranks.
- `eight_gpu_hsdp2x4`: two replicated groups of four sharded ranks.
- `eight_gpu_hsdp4x2`: four replicated groups of two sharded ranks.

Select a topology only after numerical/recovery qualification and sustained
throughput measurement. The production default is FSDP8; HSDP is a candidate,
not a measured winner. The canonical production canary is now
`production_canary/production_8gpu.{yaml,env}`.

## Current gates

The four-GPU frozen recipe, plan audit, and throughput canary were preserved
under `report/history/4gpu-before-eight-gpu-20261113/`. Their canonical reports
are marked STALE. Re-freezing requires a current-source eight-GPU throughput
measurement matching the authorized model, microbatch, compile settings, and
tokens/update. The authorization preserves the earlier E1 waiver; correctness
and final-data gates remain mandatory.

The qualification runner defaults to all eight ranks, and production validation
and canary launch explicitly require 4-to-8 reshard qualification PASS. The
watcher waits for eight idle authorized GPUs. Resource policy permits GPUs
0–7 without sharing.

The locked Python/TorchTitan paths and final 1T streams remain absent in this
execution environment. GPU canaries have not run here. No measured claim of
30-day completion is made; the plan is PREPARED_NOT_GPU_VALIDATED.

## Verification

216 grouped CPU tests passed, covering eight-GPU batch arithmetic, runtime
projection with reserve, rejection of four-GPU evidence, GPU ownership,
qualification, collector, production-contract checks, and automation. These
checks also found and fixed a pre-existing missing `e1_seedrep` entry in the
formal-contract count roster.

Machine-readable projection: `report/eight_gpu_runtime_plan_20261113.json`.
