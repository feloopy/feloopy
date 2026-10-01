# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Enumerations for branching algorithms."""

from enum import Enum


class BranchingStatus(Enum):
    """Status of a branching solve."""
    OPTIMAL = 'optimal'
    FEASIBLE = 'feasible'
    INFEASIBLE = 'infeasible'
    UNBOUNDED = 'unbounded'
    MAX_NODES = 'max_nodes'
    MAX_TIME = 'max_time'
    USER_ABORT = 'user_abort'
    UNSOLVED = 'unsolved'


class NodeStrategy(Enum):
    """Node exploration order in the B&B tree."""
    BEST_BOUND = 'best_bound'
    DFS = 'dfs'
    BFS = 'bfs'
    BEST_ESTIMATE = 'best_estimate'


class SelectionStrategy(Enum):
    """Variable selection strategy for branching."""
    MOST_FRACTIONAL = 'most_fractional'
    STRONG_BRANCHING = 'strong'
    PSEUDO_COST = 'pseudo_cost'
    RELIABILITY = 'reliability'
