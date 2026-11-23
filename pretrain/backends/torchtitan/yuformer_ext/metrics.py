"""Hardware-explicit metrics and isolated validation accounting."""
import time
from dataclasses import dataclass
from functools import wraps

from torchtitan.components.metrics import MetricsProcessor


def isolated_validation_metrics(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        metrics = self.metrics_processor
        tokens, loading = metrics.ntokens_since_last_log, metrics.data_loading_times
        last_log, started = metrics.time_last_log, time.perf_counter()
        metrics.ntokens_since_last_log = 0
        metrics.data_loading_times = []
        metrics.time_last_log = started
        try:
            return method(self, *args, **kwargs)
        finally:
            metrics.ntokens_since_last_log = tokens
            metrics.data_loading_times = loading
            metrics.time_last_log = last_log + (time.perf_counter() - started)
    return wrapped


class YuFormerMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        bf16_peak_flops: float = 0.0
        expected_gpu_name: str = ""

    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        if config.bf16_peak_flops <= 0:
            raise ValueError("BF16 dense peak FLOPs must be declared by the hardware contract")
        if config.expected_gpu_name not in self.device_memory_monitor.device_name:
            raise ValueError("GPU device differs from the declared performance contract")
        self.gpu_peak_flops = config.bf16_peak_flops
