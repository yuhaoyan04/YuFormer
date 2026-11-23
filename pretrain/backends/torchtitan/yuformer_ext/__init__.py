"""YuFormer external model package for the pinned TorchTitan v0.3.0 API."""

from torchtitan.protocols.model_spec import ModelSpec

from .model import YuFormerTitanModel
from .parallelize import parallelize_yuformer


def model_registry(preset: str, config_path: str, sdpa_backend: str = "auto") -> ModelSpec:
    return ModelSpec(
        name="yuformer",
        flavor=preset,
        model=YuFormerTitanModel.Config(preset=preset, config_path=config_path, sdpa_backend=sdpa_backend),
        parallelize_fn=parallelize_yuformer,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


__all__ = ["model_registry", "YuFormerTitanModel"]
