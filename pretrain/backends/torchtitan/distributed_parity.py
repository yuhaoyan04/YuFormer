#!/usr/bin/env python3
"""Compare full-batch gradients against the real FSDP2/chunked-loss adapter."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims

from yuformer_ext.model import YuFormerTitanModel
from yuformer_ext.parallelize import parallelize_yuformer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate-degree", type=int, default=1)
    args = parser.parse_args()
    local_rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if args.replicate_degree < 1 or world % args.replicate_degree:
        raise ValueError("replicate degree must be a positive divisor of WORLD_SIZE")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", local_rank)
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="yuformer-gradient-parity-") as temporary:
            config_path = Path(temporary) / "model.yaml"
            for tied in (True, False):
                config_path.write_text(yaml.safe_dump({"presets": {"tiny": {
                    "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                    "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
                    "max_seq_len": 32, "window_size": 8, "attention_every": 2,
                    "qk_norm": True, "attention_output_gate": True,
                    "tie_word_embeddings": tied,
                }}}))
                config = YuFormerTitanModel.Config(preset="tiny", config_path=str(config_path))
                torch.manual_seed(20261022)
                reference = YuFormerTitanModel(config).to(device)
                candidate = YuFormerTitanModel(config).to(device)
                candidate.load_state_dict(reference.state_dict())
                parallel = ParallelismConfig(data_parallel_shard_degree=world // args.replicate_degree,
                                             data_parallel_replicate_degree=args.replicate_degree,
                                             spmd_backend="partial_dtensor")
                dimensions = ParallelDims.from_config(parallel, world_size=world)
                dimensions.build_mesh()
                candidate = parallelize_yuformer(
                    candidate, parallel_dims=dimensions,
                    training=TrainingConfig(mixed_precision_param="float32",
                                            mixed_precision_reduce="float32"),
                    parallelism=parallel, compile_config=None, ac_config=None,
                    dump_folder=temporary)
                generator = torch.Generator(device=device).manual_seed(13)
                tokens = torch.randint(0, 128, (world * 2, 32), generator=generator, device=device)
                labels = torch.roll(tokens, -1, 1)
                # Unequal valid-token counts expose accidental rank averaging.
                for index in range(world * 2):
                    labels[index, :index + 1] = -100
                valid = (labels != -100).sum()
                loss = F.cross_entropy(reference(tokens).flatten(0, 1), labels.flatten(),
                                       reduction="sum") / valid
                loss.backward()
                candidate._skip_lm_head = True
                chunked = ChunkedLossWrapper.Config(
                    num_chunks=8, loss_fn=CrossEntropyLoss.Config(global_vocab_size=128)).build()
                chunked.set_lm_head(candidate.lm_head)
                local_loss = torch.zeros((), device=device)
                for accumulation_step in range(2):
                    index = accumulation_step * world + dist.get_rank()
                    partial, _ = chunked(candidate(tokens[index:index + 1]),
                                         labels[index:index + 1], global_valid_tokens=valid)
                    partial.backward()
                    local_loss += partial.detach()
                dist.all_reduce(local_loss)
                expected = dict(reference.named_parameters())
                maximum, squared_delta, squared_reference = 0.0, 0.0, 0.0
                count = 0
                for name, parameter in candidate.named_parameters():
                    if parameter.grad is None or expected[name].grad is None:
                        raise RuntimeError(f"missing gradient for {name}")
                    actual = parameter.grad.full_tensor()
                    target = expected[name].grad
                    maximum = max(maximum, float((actual - target).abs().max()))
                    squared_delta += float((actual - target).square().sum())
                    squared_reference += float(target.square().sum())
                    count += 1
                relative = (squared_delta / max(squared_reference, 1e-30)) ** 0.5
                loss_delta = float((local_loss - loss.detach()).abs())
                results.append({"tied_embeddings": tied, "grad_max_abs": maximum,
                                "grad_relative_l2": relative, "loss_abs": loss_delta,
                                "parameters_compared": count,
                                "status": "PASS" if maximum <= 2e-6 and relative <= 2e-5
                                          and loss_delta <= 2e-6 else "FAIL"})
                del candidate, reference, chunked
                torch.cuda.empty_cache()
        passed = all(row["status"] == "PASS" for row in results)
        if dist.get_rank() == 0:
            report = {"schema_version": 1, "status": "PASS" if passed else "FAIL",
                      "world_size": world, "gradient_accumulation": 2,
                      "replicate_degree": args.replicate_degree,
                      "shard_degree": world // args.replicate_degree,
                      "dtype": "float32", "cases": results}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = args.output.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary_path, args.output)
            print(json.dumps(report, sort_keys=True))
        return 0 if passed else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
