# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Cut management, lazy constraints, and separation oracles.

Refactored from the original branch_and_cut.py into a solver-neutral form.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from enum import Enum


class CutType(Enum):
    """Type of cutting plane."""
    FRACTIONAL = 'fractional'
    LAZY = 'lazy'
    USER = 'user'


@dataclass
class Cut:
    """A single cutting plane."""
    name: str
    coefficients: Dict[str, float]
    sense: str  # '<=', '>=', '=='
    rhs: float
    cut_type: CutType = CutType.LAZY
    priority: int = 0
    source: str = ''


@dataclass
class LazyConstraint:
    """Lazy constraint definition with separation oracle.

    Parameters
    ----------
    name : str
        Identifier for this constraint family.
    separation : callable or SeparationOracle
        ``(model, solution) -> List[Cut]`` or an object with
        a ``separate(solution)`` method.
    priority : int
        Higher values are processed first.
    max_cuts_per_round : int
        Maximum cuts added per separation round.
    is_valid : callable, optional
        ``(model) -> bool`` — whether this oracle applies now.
    """
    name: str
    separation: Any  # Callable or SeparationOracle
    priority: int = 0
    max_cuts_per_round: int = 100
    is_valid: Optional[Callable] = None


@dataclass
class CutPool:
    """Pool of generated cuts to avoid duplicates."""
    cuts: List[Cut] = field(default_factory=list)
    max_cuts: int = 10000

    def add(self, cut: Cut) -> bool:
        """Add cut if not already present. Returns True if added."""
        for existing in self.cuts:
            if (existing.name == cut.name and
                    existing.coefficients == cut.coefficients and
                    existing.sense == cut.sense and
                    abs(existing.rhs - cut.rhs) < 1e-10):
                return False

        if len(self.cuts) >= self.max_cuts:
            self.cuts.sort(key=lambda c: c.priority)
            self.cuts = self.cuts[len(self.cuts) // 4:]

        self.cuts.append(cut)
        return True

    def get_violated(self, solution: Dict[str, float],
                     tol: float = 1e-6) -> List[Cut]:
        """Return cuts violated by the current solution."""
        violated = []
        for cut in self.cuts:
            lhs = sum(cut.coefficients.get(var, 0) * val
                      for var, val in solution.items())
            if cut.sense == '<=' and lhs > cut.rhs + tol:
                violated.append(cut)
            elif cut.sense == '>=' and lhs < cut.rhs - tol:
                violated.append(cut)
            elif cut.sense == '==' and abs(lhs - cut.rhs) > tol:
                violated.append(cut)
        return violated

    def clear(self):
        self.cuts.clear()


class SeparationOracles:
    """Built-in separation oracles for common problem classes."""

    @staticmethod
    def subtour_elimination_tsp(model, solution, n_cities, x_vars=None):
        """Separate subtour elimination constraints for TSP.

        Parameters
        ----------
        model : feloopy model
        solution : dict
            ``{var_name: value}`` for the current solution.
        n_cities : int
            Number of cities.
        x_vars : optional
            Binary edge variable mapping (unused — derived from solution).

        Returns
        -------
        list of Cut
        """
        edges = []
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    var_name = f'x_{i}_{j}'
                    if var_name in solution and solution[var_name] > 0.5:
                        edges.append((i, j))

        visited = [False] * n_cities
        subtours = []
        for start in range(n_cities):
            if not visited[start]:
                tour = []
                stack = [start]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        tour.append(node)
                        for i, j in edges:
                            if i == node and not visited[j]:
                                stack.append(j)
                subtours.append(tour)

        cuts = []
        for subtour in subtours:
            if 0 < len(subtour) < n_cities:
                coeffs = {}
                for i in subtour:
                    for j in range(n_cities):
                        if j not in subtour and i != j:
                            coeffs[f'x_{i}_{j}'] = 1.0
                if coeffs:
                    cuts.append(Cut(
                        name=f'subtour_{"_".join(map(str, subtour))}',
                        coefficients=coeffs,
                        sense='>=',
                        rhs=2.0,
                        cut_type=CutType.LAZY,
                        priority=1,
                        source='subtour_elimination_tsp',
                    ))
        return cuts

    @staticmethod
    def tsp_2_opt(model, solution, n_cities, dist_matrix):
        """2-opt neighbourhood separation for TSP (stub)."""
        return []

    @staticmethod
    def vrp_capacity(model, solution, n_vehicles, n_customers,
                     demand, capacity):
        """Separate capacity constraints for VRP (stub)."""
        return []

    @staticmethod
    def bin_packing(model, solution, n_items, bin_capacity, item_sizes):
        """Separate cover inequalities for bin packing (stub)."""
        return []


def create_tsp_branch_and_cut(n_cities, dist_matrix):
    """Create a ``LazyConstraint`` pre-configured for TSP subtour elimination."""
    from .callbacks import BranchingCallback
    lc = LazyConstraint(
        name='subtour_elimination',
        separation=lambda m, sol: SeparationOracles.subtour_elimination_tsp(
            m, sol, n_cities),
        priority=1,
        max_cuts_per_round=n_cities,
    )
    return lc


def create_vrp_branch_and_cut(n_vehicles, n_customers, demand, capacity):
    """Create a ``LazyConstraint`` pre-configured for VRP capacity."""
    lc = LazyConstraint(
        name='capacity',
        separation=lambda m, sol: SeparationOracles.vrp_capacity(
            m, sol, n_vehicles, n_customers, demand, capacity),
        priority=1,
    )
    return lc
