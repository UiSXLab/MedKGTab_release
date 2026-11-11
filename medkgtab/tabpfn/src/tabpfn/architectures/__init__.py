from __future__ import annotations

from typing import TYPE_CHECKING

from . import base

if TYPE_CHECKING:
    from tabpfn.architectures.interface import ArchitectureModule

ARCHITECTURES: dict[str, ArchitectureModule] = {"base": base}
"""Map from architecture names to the corresponding module."""
