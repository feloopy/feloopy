# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Lazy import proxy for polars.

Defers the actual `import polars` until first attribute access, so that
`import feloopy` does not pay the polars import cost (or fail hard) when
polars is never used.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional, TYPE_CHECKING

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

if TYPE_CHECKING:  # pragma: no cover
    import polars as _polars_stub  # for type checkers only


class _LazyPolars:
    """Module-like proxy that imports polars on first attribute access."""

    __slots__ = ("_module",)

    def __init__(self) -> None:
        self._module: Optional[Any] = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module("polars")
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


#: Shared lazy polars proxy.  Use like ``import polars as pl``.
pl = _LazyPolars()
