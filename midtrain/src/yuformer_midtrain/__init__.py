"""YuFormer long-context midtraining contracts."""

from .contract import audit_ladder

__all__ = ["audit_ladder"]


def midtrain_from_yaml():
    from .registry import midtrain_from_yaml as factory
    return factory()
