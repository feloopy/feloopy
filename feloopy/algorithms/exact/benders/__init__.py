"""Benders decomposition for mixed-integer linear programs.

Splits a problem into a master problem and one or more subproblems,
iteratively adding Benders cuts to tighten the master's relaxation.

Classes:
    BendersDecomposition -- manual and automatic Benders decomposition.
    benders()            -- simple one-shot convenience constructor.
    benders_decomposition -- decorator-based API.

Enums:
    BendersCutStrategy, BendersAcceleration, BendersMethod, BendersStatus

Result:
    BendersResult, BendersParams

Callbacks:
    BendersCallback, BendersContext
"""

from .enums import (
    BendersCutStrategy,
    BendersAcceleration,
    BendersMethod,
    BendersStatus,
    BendersError,
)
from .callbacks import (
    BendersCallback,
    BendersContext,
)
from .result import (
    BendersResult,
    BendersParams,
)
from .core import (
    BendersDecomposition,
    benders,
    benders_decomposition,
)
from .automatic import (
    run_automatic_benders,
    select_variant,
    select_complicating_variables,
    split_solver_pair,
)

__all__ = [
    'BendersDecomposition',
    'BendersCutStrategy',
    'BendersAcceleration',
    'BendersMethod',
    'BendersStatus',
    'BendersError',
    'BendersResult',
    'BendersParams',
    'BendersCallback',
    'BendersContext',
    'benders',
    'benders_decomposition',
    'run_automatic_benders',
    'select_variant',
    'select_complicating_variables',
    'split_solver_pair',
]
