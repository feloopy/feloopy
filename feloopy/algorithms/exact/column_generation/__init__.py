"""Column generation for large-scale integer programming.

Decomposes a problem into a master problem and a pricing subproblem,
iteratively adding columns (variables) that improve the objective.

Classes:
    ColumnGeneration -- manual and automatic column generation.

Result:
    ColumnGenerationResult

Error:
    ColumnGenerationError
"""

from .core import (
    ColumnGeneration,
    ColumnGenerationResult,
    ColumnGenerationError,
)
from . import automatic  # noqa: F401  -- binds from_automatic_dw/dw2 etc.

__all__ = [
    'ColumnGeneration',
    'ColumnGenerationResult',
    'ColumnGenerationError',
]
