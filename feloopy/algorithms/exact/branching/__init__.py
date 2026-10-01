"""Branching algorithms: Branch-and-Bound, Branch-and-Cut, Branch-and-Price.

Provides solver-neutral tree-search methods for mixed-integer programs.
All three classes follow the same manual/automatic API pattern::

    solver = BranchAndBound()          # or BranchAndCut / BranchAndPrice
    result = solver.solve(model_fn, interface='highs', show_log=True)

Automatic (called from feloopy.py)::

    solver = BranchAndBound.from_automatic(model_fn, directions, interface, solver)
    result = solver.solve()
"""

from .enums import BranchingStatus, NodeStrategy, SelectionStrategy
from .result import BranchAndBoundResult, BranchAndCutResult, BranchAndPriceResult
from .callbacks import BranchingCallback, SeparationOracle
from .cuts import Cut, CutPool, LazyConstraint, CutType, SeparationOracles
from .core import BranchAndBound
from .branch_and_cut import BranchAndCut
from .branch_and_price import BranchAndPrice

__all__ = [
    'BranchAndBound',
    'BranchAndCut',
    'BranchAndPrice',
    'BranchingStatus',
    'NodeStrategy',
    'SelectionStrategy',
    'BranchAndBoundResult',
    'BranchAndCutResult',
    'BranchAndPriceResult',
    'BranchingCallback',
    'SeparationOracle',
    'Cut',
    'CutPool',
    'LazyConstraint',
    'CutType',
    'SeparationOracles',
]
