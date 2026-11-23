# Pretrain stack audit — 2026-10-20

Current status: **BLOCKED** (repository contracts pass; runtime qualification is incomplete).

The canonical base run is 1T final-tokenizer tokens: 50B warmup, 800B stable, and 150B
annealing at 4K context. The separate 50B context-extension experiment is excluded from the
1T accounting. Active formal E1/E2 configurations use TorchTitan v0.3.0 and bind both the data
release and tokenizer gate. Historical 1.44T settings remain only in explicitly superseded
history; MFU diagnostics are bounded 200-step `probe_only` compatibility runs.

Implemented repository closure:

- TorchTitan external `BaseModel` adapter with FSDP2 layer wrapping and chunked loss;
- immutable pretokenized stream validation against tokenizer, snapshot, edge hash, EOD count,
  dtype, byte length, and token length;
- topology-independent global sample cursor in DCP state, with alignment checks on reshard;
- YAML-to-typed-config translation for AdamW, WSD/cosine, BF16 mixed precision, asynchronous
  DCP, metrics, and deterministic stream order;
- fail-closed launcher and machine-readable whole-stack audit.

Open runtime gates:

- the host has driver 535.161.08, Python 3.10, and PyTorch 2.2 alpha; the pinned stack requires
  driver >=580.65.06, Python 3.12, PyTorch 2.14.0/cu130, TorchTitan v0.3.0, and its locked commit;
- the final warmup/stable/anneal 1T streams have not yet been materialized;
- single-GPU numerical parity, 4-GPU smoke, SIGKILL recovery, and 4-to-2/8 topology reshard
  canaries must run after the environment and streams exist.

The authoritative JSON report is `report/pretrain_stack_audit_20261020.json`.
