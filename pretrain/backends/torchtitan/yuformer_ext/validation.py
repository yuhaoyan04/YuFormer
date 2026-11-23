"""Per-domain held-out validation for the pinned DP/FSDP2 backend."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.tools import utils

from .stream_math import should_validate_step
from .metrics import isolated_validation_metrics


class DomainValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        domain_names: list[str] = field(default_factory=list)
        final_step: int = 0

    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        if not config.domain_names:
            raise ValueError("domain_names must be non-empty")
        self.domain_names = list(config.domain_names)
        if config.final_step < 1:
            raise ValueError("final_step must be positive")

    def should_validate(self, step: int) -> bool:
        return should_validate_step(step, self.config.freq, self.config.final_step)

    @sl.log_trace_span("eval")
    @isolated_validation_metrics
    @torch.no_grad()
    def validate(self, model_parts: list[torch.nn.Module], step: int) -> None:
        sl.add_step_tag("eval")
        if self.parallel_dims.pp_enabled or self.parallel_dims.cp_enabled:
            raise ValueError("DomainValidator v1 qualifies DP/FSDP2 without PP/CP")
        for model in model_parts:
            model.eval()
        device = utils.device_type
        count = len(self.domain_names)
        loss_by_domain = torch.zeros(count, dtype=torch.float64, device=device)
        tokens_by_domain = torch.zeros(count, dtype=torch.int64, device=device)
        dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size, dp_rank=self.dp_rank,
            tokenizer=self.tokenizer, seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
        )
        iterator = iter(dataloader)
        for _ in range(self.config.steps):
            input_dict, labels = next(iterator)
            stage_ids = input_dict.pop("_yuformer_stage_id")
            unique_domains = torch.unique(stage_ids)
            if unique_domains.numel() != 1:
                raise ValueError("validation batch crosses domain boundaries")
            domain = int(unique_domains.item())
            if not 0 <= domain < count:
                raise ValueError(f"invalid validation domain id: {domain}")
            self.metrics_processor.ntokens_since_last_log += labels.numel()
            for key, value in input_dict.items():
                input_dict[key] = value.to(device)
            labels = labels.to(device)
            valid_tokens = (labels != IGNORE_INDEX).sum()
            inputs, labels, extra_kwargs = self.post_dataloading_process(
                input_dict, labels, model_parts
            )
            with self.validation_context():
                predictions = model_parts[0](inputs, **extra_kwargs)
                loss_sum, _ = self.loss_fn(predictions, labels)
            loss_by_domain[domain] += loss_sum.detach().to(torch.float64)
            tokens_by_domain[domain] += valid_tokens
        if self.parallel_dims.dp_enabled:
            mesh = self.parallel_dims.get_mesh("batch")
            loss_by_domain = dist_utils.dist_sum_tensor(loss_by_domain, mesh, None)
            tokens_by_domain = dist_utils.dist_sum_tensor(tokens_by_domain, mesh, None)
        if bool((tokens_by_domain <= 0).any().item()):
            raise RuntimeError("one or more validation domains received zero tokens")
        averages = loss_by_domain / tokens_by_domain
        total_loss = float(loss_by_domain.sum().item() / tokens_by_domain.sum().item())
        extra = {
            f"validation_metrics/domain/{name}/loss": float(averages[index].item())
            for index, name in enumerate(self.domain_names)
        }
        self.metrics_processor.log_validation(loss=total_loss, step=step, extra_metrics=extra)
        for model in model_parts:
            model.train()
