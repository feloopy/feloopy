# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Solver-neutral branching callbacks and separation oracle protocol."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .cuts import Cut, CutPool, LazyConstraint


class SeparationOracle(ABC):
    """Protocol for separation oracles.

    Subclass this and implement ``separate`` to define custom cut
    separation logic.  Attach instances to a ``BranchAndCut`` via
    ``add_separation_oracle``.
    """

    @abstractmethod
    def separate(self, solution: Dict[str, float]) -> List[Cut]:
        """Given a feasible solution, return a list of violated cuts."""

    def is_valid(self, model) -> bool:
        """Return True if this oracle applies to the current model state."""
        return True


class BranchingCallback:
    """Solver-neutral branching callback.

    Wraps feloopy's ``features['callback']`` mechanism.

    * **Gurobi** — full ``cbLazy`` support (MIPSOL / MIPNODE).
    * **CPLEX** — ``LazyConstraintCallback`` support.
    * **Other solvers** — polling fallback: solve, separate, re-solve.

    Parameters
    ----------
    lazy_constraints : list of LazyConstraint, optional
        Initial lazy constraints to register.
    """

    def __init__(self, lazy_constraints=None):
        self.lazy_constraints: List[LazyConstraint] = list(lazy_constraints or [])
        self.cut_pool = CutPool()
        self._iteration = 0
        self._cuts_added_total = 0
        self._cuts_added_per_iteration: List[int] = []
        self._solution_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_lazy_constraint(self, lc: LazyConstraint):
        self.lazy_constraints.append(lc)
        self.lazy_constraints.sort(key=lambda c: -c.priority)

    # ------------------------------------------------------------------
    # Gurobi callback
    # ------------------------------------------------------------------

    def gurobi_callback(self, model, where):
        """Gurobi-native callback — handles MIPSOL and MIPNODE."""
        try:
            import gurobipy as gurobi
        except ImportError:
            return

        if where == gurobi.GRB.Callback.MIPSOL:
            self._process_integer_solution_gurobi(model)
        elif where == gurobi.GRB.Callback.MIPNODE:
            pass  # fractional cuts (optional, future)

    def _process_integer_solution_gurobi(self, model):
        """Extract solution, run separation, inject cuts via cbLazy."""
        import gurobipy as gurobi

        solution = {v.VarName: v.X for v in model.getVars()}
        self._solution_history.append(solution)
        self._iteration += 1
        cuts_this_round = 0

        for lc in self.lazy_constraints:
            if lc.is_valid is not None and not lc.is_valid(None):
                continue
            violated = self._run_separation(lc, solution)
            for cut in violated[:lc.max_cuts_per_round]:
                if self.cut_pool.add(cut):
                    self._inject_cut_gurobi(model, cut)
                    cuts_this_round += 1
                    self._cuts_added_total += 1

        self._cuts_added_per_iteration.append(cuts_this_round)

    def _inject_cut_gurobi(self, model, cut: Cut):
        """Inject a single cut into the Gurobi model via cbLazy."""
        import gurobipy as gurobi

        expr = gurobi.LinExpr()
        for var_name, coeff in cut.coefficients.items():
            var = model.getVarByName(var_name)
            if var is not None:
                expr += coeff * var

        if cut.sense == '<=':
            model.cbLazy(expr <= cut.rhs)
        elif cut.sense == '>=':
            model.cbLazy(expr >= cut.rhs)
        elif cut.sense == '==':
            model.cbLazy(expr == cut.rhs)

    # ------------------------------------------------------------------
    # CPLEX callback
    # ------------------------------------------------------------------

    def cplex_callback(self, context):
        """CPLEX LazyConstraintCallback context handler.

        Expects *context* to be a ``cplex.callbacks.LazyConstraintCallback``
        subclass instance.
        """
        try:
            import cplex.callbacks as cpx_callbacks
            if not isinstance(context, cpx_callbacks.LazyConstraintCallback):
                return
        except ImportError:
            return

        solution = {}
        try:
            cols = context.get_values()
            names = context.get_var_names()
            solution = dict(zip(names, cols))
        except Exception:
            return

        self._solution_history.append(solution)
        self._iteration += 1
        cuts_this_round = 0

        for lc in self.lazy_constraints:
            if lc.is_valid is not None and not lc.is_valid(None):
                continue
            violated = self._run_separation(lc, solution)
            for cut in violated[:lc.max_cuts_per_round]:
                if self.cut_pool.add(cut):
                    self._inject_cut_cplex(context, cut)
                    cuts_this_round += 1
                    self._cuts_added_total += 1

        self._cuts_added_per_iteration.append(cuts_this_round)

    def _inject_cut_cplex(self, context, cut: Cut):
        """Inject a cut via CPLEX LazyConstraintCallback."""
        try:
            import cplex as cpx
            names = list(cut.coefficients.keys())
            coeffs = list(cut.coefficients.values())
            sense_map = {'<=': 'L', '>=': 'G', '==': 'E'}
            context.add(
                linear_expr=[names, coeffs],
                sense=sense_map.get(cut.sense, 'L'),
                rhs=cut.rhs,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generic polling (fallback for solvers without native callbacks)
    # ------------------------------------------------------------------

    def generic_poll(self, solution: Dict[str, float]) -> List[Cut]:
        """Run separation on a solution and return new cuts.

        For solvers without native lazy-constraint callbacks (HiGHS,
        OR-Tools, PuLP, …).  Call this after each solve, add the
        returned cuts as regular constraints, then re-solve.
        """
        self._solution_history.append(solution)
        self._iteration += 1
        new_cuts: List[Cut] = []
        cuts_this_round = 0

        for lc in self.lazy_constraints:
            if lc.is_valid is not None and not lc.is_valid(None):
                continue
            violated = self._run_separation(lc, solution)
            for cut in violated[:lc.max_cuts_per_round]:
                if self.cut_pool.add(cut):
                    new_cuts.append(cut)
                    cuts_this_round += 1
                    self._cuts_added_total += 1

        self._cuts_added_per_iteration.append(cuts_this_round)
        return new_cuts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_separation(self, lc: LazyConstraint,
                        solution: Dict[str, float]) -> List[Cut]:
        """Invoke a lazy constraint's separation oracle."""
        try:
            sep = lc.separation
            if hasattr(sep, 'separate'):
                return sep.separate(solution)
            return sep(None, solution)
        except Exception:
            return []

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'iterations': self._iteration,
            'total_cuts': self._cuts_added_total,
            'cuts_per_iteration': list(self._cuts_added_per_iteration),
            'cut_pool_size': len(self.cut_pool.cuts),
            'num_separation_oracles': len(self.lazy_constraints),
            'solutions_evaluated': len(self._solution_history),
        }

    def reset(self):
        self._iteration = 0
        self._cuts_added_total = 0
        self._cuts_added_per_iteration = []
        self._solution_history = []
        self.cut_pool.clear()
