# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Result dataclasses for branching algorithms."""

from dataclasses import dataclass, field

from ..base.result import DecompositionResult
from .enums import BranchingStatus


@dataclass
class BranchAndBoundResult(DecompositionResult):
    """Result of a Branch-and-Bound solve."""
    n_nodes_explored: int = 0
    n_pruned: int = 0
    n_infeasible: int = 0
    best_bound: float = float('-inf')
    gap: float = float('inf')
    solution_count: int = 0
    status: BranchingStatus = BranchingStatus.UNSOLVED


@dataclass
class BranchAndCutResult(BranchAndBoundResult):
    """Result of a Branch-and-Cut solve."""
    n_cuts_added: int = 0
    cut_generation_time: float = 0.0
    separation_time: float = 0.0


@dataclass
class BranchAndPriceResult(BranchAndBoundResult):
    """Result of a Branch-and-Price solve."""
    n_columns_generated: int = 0
    n_pricing_problems: int = 0
    column_generation_time: float = 0.0
