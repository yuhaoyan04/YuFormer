"""FSDP2 policy for the YuFormer dense decoder."""
from __future__ import annotations

from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.distributed.compile import apply_compile


def parallelize_yuformer(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled:
        raise ValueError("YuFormer v1 adapter currently qualifies DP/FSDP2 only")
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model.core)
    if (compile_config is not None and compile_config.enable
            and "model" in compile_config.components):
        apply_compile(
            model.core,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    # TorchTitan's partial-DTensor backend exposes the legacy flattened
    # ``fsdp`` mesh.  ``resolve_fsdp_mesh`` is only valid for the newer
    # full-DTensor/SPMD-types backends and asserts on the default partial
    # backend, including the world-size-one qualification case.
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(mesh_names)
        dp_mesh_dims = None
    kwargs = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cast_forward_inputs=False,
        ),
    }
    if dp_mesh_dims is not None:
        kwargs["dp_mesh_dims"] = dp_mesh_dims
    if training.enable_cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()

    reshard = parallelism.fsdp_reshard_after_forward != "never"
    if model.model_config.tie_word_embeddings:
        fully_shard(
            [model.core.embed_tokens, model.core.final_norm, model.core.lm_head],
            **kwargs,
            reshard_after_forward=False,
        )
    else:
        fully_shard(model.core.embed_tokens, **kwargs, reshard_after_forward=reshard)
        fully_shard(
            [model.core.final_norm, model.core.lm_head],
            **kwargs,
            reshard_after_forward=False,
        )
    for block in model.core.layers:
        fully_shard(block, **kwargs, reshard_after_forward=reshard)
    fully_shard(model, **kwargs, reshard_after_forward=False)
    # ChunkedLossWrapper already divides by the global valid-token count.
    # FSDP must sum rank-local gradients without another world-size division.
    disable_fsdp_gradient_division(model)
    return model
