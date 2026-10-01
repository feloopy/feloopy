# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Shared result abstractions for decomposition algorithms."""

from dataclasses import dataclass, field


@dataclass
class DecompositionResult:
    """Base result class shared by all decomposition algorithms.

    Subclasses add algorithm-specific fields (e.g. ``n_cuts`` for Benders,
    ``columns`` for Column Generation, ``multipliers`` for Lagrangian).
    """
    status: str = 'unsolved'
    objective: float = float('inf')
    variables: dict = field(default_factory=dict)
    iterations: int = 0
    runtime_total: float = 0.0
    history: list = field(default_factory=list)

    @property
    def is_optimal(self):
        return self.status == 'optimal'

    @property
    def is_feasible(self):
        return self.status in ('optimal', 'feasible')
