import warnings

from tabpfn import model_loading as loading
from tabpfn.architectures.base import (
    attention,
    bar_distribution,
    config,
    encoders,
    layer,
    memory,
    mlp,
    transformer,
)

__all__ = [
    "attention",
    "bar_distribution",
    "config",
    "encoders",
    "layer",
    "loading",
    "memory",
    "mlp",
    "transformer",
]

warnings.warn(
    "tabpfn.model has moved to tabpfn.architectures.base. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
