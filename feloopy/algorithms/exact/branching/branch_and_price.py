# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Branch-and-Price: Branch-and-Bound with column generation.

Integrates ColumnGeneration with a branch-and-bound tree.  At each
node the LP relaxation is solved via column generation, and branching
is performed on integer variables that appear in the pricing problem.

Manual API
----------
>>> bp = BranchAndPrice()
>>> bp.set_master(master_fn)
>>> bp.set_pricing(pricing_fn)
>>> result = bp.solve(model_fn, interface='highs', show_log=True)

Automatic API
-------------
>>> bp = BranchAndPrice.from_automatic(
...     model_fn, directions, interface='highs', solver='highs')
>>> result = bp.solve(interface='highs')
"""

import heapq
import time

from .enums import BranchingStatus, NodeStrategy, SelectionStrategy
from .result import BranchAndPriceResult
from ..column_generation import ColumnGeneration, ColumnGenerationResult
from ..base.logging import make_logger


class _Node:
    """A single node in the branch-and-price tree."""

    __slots__ = ('depth', 'fixings', 'bound', 'node_id')

    def __init__(self, depth, fixings, bound, node_id):
        self.depth = depth
        self.fixings = fixings
        self.bound = bound
        self.node_id = node_id

    def __lt__(self, other):
        return self.bound < other.bound


class BranchAndPrice:
    """Branch-and-Price: Branch-and-Bound with column generation.

    Integrates ``ColumnGeneration`` with a best-bound branch-and-bound
    tree.  At every node, column generation solves the LP relaxation
    and branching is imposed on fractional integer variables.

    Manual API::

        bp = BranchAndPrice()
        bp.set_master(master_fn)       # (columns) -> dict
        bp.set_pricing(pricing_fn)     # (duals) -> [column_dicts] | None
        bp.add_branching_variable('x')
        result = bp.solve(model_fn, interface='highs')

    Automatic API::

        bp = BranchAndPrice.from_automatic(
            model_fn, directions, interface='highs', solver='highs')
        result = bp.solve()
    """

    def __init__(self):
        """Initialise a Branch-and-Price solver.

        Configure with ``set_master``, ``set_pricing``, and optionally
        ``add_branching_variable``, then call ``solve()``.
        """
        self._master_fn = None
        self._pricing_fn = None
        self._branching_vars = []
        self._cg = None
        self._cg_kwargs = {}
        self._node_counter = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_master(self, master_fn):
        """Set the master problem factory.

        Parameters
        ----------
        master_fn : callable
            ``(columns) -> dict`` where the dict contains at least
            ``objective`` and ``duals`` keys.
        """
        self._master_fn = master_fn
        return self

    def set_pricing(self, pricing_fn):
        """Set the pricing oracle.

        Parameters
        ----------
        pricing_fn : callable
            ``(duals) -> list[dict] | None`` -- returns a list of
            column dictionaries (each with at least ``reduced_cost``)
            or ``None`` when no improving column exists.
        """
        self._pricing_fn = pricing_fn
        return self

    def add_branching_variable(self, var_name):
        """Mark a variable for branching decisions.

        Parameters
        ----------
        var_name : str
            Name of the integer variable to branch on.
        """
        if var_name not in self._branching_vars:
            self._branching_vars.append(var_name)
        return self

    # ------------------------------------------------------------------
    # Automatic construction
    # ------------------------------------------------------------------

    @classmethod
    def from_automatic(cls, model_fn, directions, interface='highs',
                       solver='highs', master_constraints=None,
                       pricing_constraints=None, show_log=False,
                       method='exact', time_limit=None, cpu_threads=None,
                       absolute_gap=None, relative_gap=None,
                       captured=None, em=None):
        """Auto-configure Branch-and-Price from a feloopy model function.

        Uses ``ColumnGeneration.from_automatic`` to derive master and
        pricing callbacks from the model's captured structure.

        Parameters
        ----------
        model_fn : callable
            ``feloopy.model`` factory that defines the full problem.
        directions : list of str
            Optimization direction(s), e.g. ``['min']``.
        interface : str
            Solver interface for the master LP.
        solver : str
            Solver engine name.
        master_constraints : list of int, optional
            Constraint indices for the master problem.  When *None*
            all constraints are used.
        pricing_constraints : list of int, optional
            Constraint indices for the pricing subproblem.
        show_log : bool
            Print CG iteration log.
        method : str
            feloopy method keyword.
        time_limit, cpu_threads,         absolute_gap, relative_gap : optional
            Solver options forwarded to the column generation callbacks.
        captured : _AutomaticCaptureModel, optional
            Pre-built capture. When omitted it is derived from ``model_fn``.
        em : optional
            Environment model forwarded to the column generation callbacks.

        Returns
        -------
        BranchAndPrice
            Configured instance ready to call ``.solve()``.
        """
        from ..automatic_capture import _capture_from_model_fn

        bp = cls()
        bp._cg_kwargs = dict(
            method=method, time_limit=time_limit,
            cpu_threads=cpu_threads, absolute_gap=absolute_gap,
            relative_gap=relative_gap,
        )

        _if = interface if isinstance(interface, str) else (
            interface[0] if isinstance(interface, (list, tuple)) else 'highs')
        _sv = solver if isinstance(solver, str) else (
            solver[0] if isinstance(solver, (list, tuple)) else 'highs')

        if captured is None:
            captured = _capture_from_model_fn(model_fn)

        build_master, pricing_oracle = ColumnGeneration.from_automatic(
            captured, em, _if, _sv, directions,
            method=method, time_limit=time_limit,
            cpu_threads=cpu_threads, absolute_gap=absolute_gap,
            relative_gap=relative_gap,
        )

        bp._master_fn = build_master
        bp._pricing_fn = pricing_oracle

        int_var_names = set()
        for (kind, name) in captured.variables:
            if kind in ('ivar', 'bvar'):
                int_var_names.add(name)

        for (kind, name) in captured.variables:
            if name in int_var_names:
                bp.add_branching_variable(name)

        return bp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_node_id(self):
        self._node_counter += 1
        return self._node_counter

    def _build_cg(self, master_fn, pricing_fn, columns):
        """Create and configure a fresh ``ColumnGeneration`` instance."""
        cg = ColumnGeneration(master_solver=master_fn,
                              pricing_oracle=pricing_fn)
        cg.columns = list(columns)
        return cg

    def _run_cg_at_node(self, master_fn, pricing_fn, columns,
                        fixings, tolerance, show_log):
        """Run column generation at a single tree node.

        Parameters
        ----------
        master_fn : callable
            Master solver callback.
        pricing_fn : callable
            Pricing oracle callback.
        columns : list
            Current column pool.
        fixings : dict
            ``{var_name: value}`` -- variables fixed at this node.
        tolerance : float
            CG convergence tolerance.
        show_log : bool
            Enable CG logging.

        Returns
        -------
        tuple of (ColumnGenerationResult, list)
        """
        wrapped_master = master_fn
        if fixings:
            _original = master_fn
            _fix = dict(fixings)

            def wrapped_master(columns_input):
                result = _original(columns_input)
                if not isinstance(result, dict):
                    return result
                sol = result.get('solution', {})
                for var_name, val in _fix.items():
                    if var_name in sol:
                        sol[var_name] = val
                return result

        cg = self._build_cg(wrapped_master, pricing_fn, columns)
        max_iters = self._cg_kwargs.get('max_cg_iterations', 100)
        result = cg.solve(
            master_solver=wrapped_master,
            pricing_oracle=pricing_fn,
            max_iterations=max_iters,
            tolerance=tolerance,
            show_log=show_log,
        )
        return result, cg.columns

    def _select_branching_variable(self, solution):
        """Pick the most fractional branching variable.

        Prefers variables that were explicitly registered via
        ``add_branching_variable``; falls back to any integer
        variable in the solution.
        """
        best_var = None
        best_frac = 0.5

        for var_name in self._branching_vars:
            val = solution.get(var_name)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            frac = abs(fval - round(fval))
            if frac > best_frac:
                best_frac = frac
                best_var = var_name

        if best_var is not None:
            return best_var

        for var_name, val in solution.items():
            if var_name in self._branching_vars:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            frac = abs(fval - round(fval))
            if frac > best_frac:
                best_frac = frac
                best_var = var_name

        return best_var

    def _is_integer(self, value, tol=1e-6):
        """Check whether *value* is effectively integer."""
        try:
            return abs(float(value) - round(float(value))) < tol
        except (TypeError, ValueError):
            return True

    def _check_solution_integer(self, solution):
        """Return True if every branching variable is integer-valued."""
        for var_name in self._branching_vars:
            val = solution.get(var_name)
            if val is None:
                continue
            if not self._is_integer(val):
                return False
        return True

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, model_fn=None, interface='highs', solver=None,
              directions=None, max_nodes=1000, max_time=None,
              tolerance=1e-6, show_log=False, save_vars=True, **kwargs):
        """Run Branch-and-Price and return a ``BranchAndPriceResult``.

        Parameters
        ----------
        model_fn : callable, optional
            ``feloopy.model`` factory -- used only when a manual master
            or pricing callback has not been set via ``set_master`` /
            ``set_pricing``.
        interface : str
            Solver interface name.
        solver : str, optional
            Solver engine name (defaults to *interface*).
        directions : list of str, optional
            ``['min']`` or ``['max']``.
        max_nodes : int
            Maximum number of tree nodes to explore.
        max_time : float, optional
            Wall-clock time limit in seconds.
        tolerance : float
            Convergence tolerance for column generation.
        show_log : bool
            Print node-level log messages.
        save_vars : bool
            Populate ``result.variables`` from the best solution.
        **kwargs
            Extra keyword arguments forwarded to the CG solver call
            (e.g. ``time_limit``, ``cpu_threads``).

        Returns
        -------
        BranchAndPriceResult
        """
        if self._master_fn is None or self._pricing_fn is None:
            raise ValueError(
                "Master and pricing callbacks must be set. "
                "Use set_master()/set_pricing() or from_automatic().")

        log = make_logger('BranchAndPrice', show_log)
        wall_start = time.time()

        self._node_counter = 0
        total_columns_generated = 0
        total_pricing_problems = 0
        cg_time_total = 0.0
        n_nodes_explored = 0
        n_pruned = 0
        n_infeasible = 0

        best_obj = float('inf')
        best_bound = float('-inf')
        best_solution = None
        best_columns = []
        cg_history = []
        status = BranchingStatus.UNSOLVED

        root = _Node(depth=0, fixings={}, bound=float('-inf'),
                      node_id=self._next_node_id())
        queue = [root]
        heapq.heapify(queue)

        while queue:
            if max_time is not None and (time.time() - wall_start) > max_time:
                log(f"Time limit reached ({max_time}s)")
                status = BranchingStatus.MAX_TIME
                break

            if n_nodes_explored >= max_nodes:
                log(f"Node limit reached ({max_nodes})")
                status = BranchingStatus.MAX_NODES
                break

            node = heapq.heappop(queue)
            n_nodes_explored += 1

            log(f"Node {node.node_id} (depth={node.depth}, "
                f"bound={node.bound:.6f}, fixings={len(node.fixings)})")

            cg_start = time.time()
            try:
                cg_result, current_columns = self._run_cg_at_node(
                    self._master_fn, self._pricing_fn,
                    best_columns, node.fixings,
                    tolerance, show_log=False)
            except Exception as exc:
                log(f"Node {node.node_id} infeasible: {exc}")
                n_infeasible += 1
                continue
            cg_time_total += time.time() - cg_start

            if cg_result is None:
                n_infeasible += 1
                continue

            node_obj = cg_result.objective
            if node_obj is None:
                n_infeasible += 1
                continue

            node_obj = float(node_obj)

            total_columns_generated += len(cg_result.columns) - len(best_columns)
            if cg_result.history:
                total_pricing_problems += sum(
                    h.get('new_columns', 0) for h in cg_result.history
                )
            cg_history.append({
                'node_id': node.node_id,
                'objective': node_obj,
                'columns': len(cg_result.columns),
                'iterations': cg_result.iterations,
            })

            if node_obj >= best_obj:
                log(f"Node {node.node_id}: bound {node_obj:.6f} >= "
                    f"best {best_obj:.6f}, pruned")
                n_pruned += 1
                continue

            solution = {}
            if isinstance(cg_result.variables, dict):
                solution = cg_result.variables
            elif hasattr(cg_result.variables, 'items'):
                try:
                    solution = dict(cg_result.variables)
                except (TypeError, ValueError):
                    pass

            if self._check_solution_integer(solution):
                best_obj = node_obj
                best_solution = solution
                best_columns = list(current_columns)
                log(f"Node {node.node_id}: integer solution "
                    f"obj={best_obj:.6f}")
            else:
                branch_var = self._select_branching_variable(solution)
                if branch_var is None:
                    best_obj = node_obj
                    best_solution = solution
                    best_columns = list(current_columns)
                    log(f"Node {node.node_id}: no branching var, "
                        f"accepting obj={best_obj:.6f}")
                else:
                    branch_val = float(solution.get(branch_var, 0.0))
                    floor_val = int(branch_val)
                    ceil_val = floor_val + 1

                    log(f"Node {node.node_id}: branching on "
                        f"'{branch_var}'={branch_val:.4f} "
                        f"-> [{floor_val}, {ceil_val}]")

                    left_fixings = dict(node.fixings)
                    left_fixings[branch_var] = float(floor_val)
                    left_node = _Node(
                        depth=node.depth + 1,
                        fixings=left_fixings,
                        bound=node_obj,
                        node_id=self._next_node_id(),
                    )

                    right_fixings = dict(node.fixings)
                    right_fixings[branch_var] = float(ceil_val)
                    right_node = _Node(
                        depth=node.depth + 1,
                        fixings=right_fixings,
                        bound=node_obj,
                        node_id=self._next_node_id(),
                    )

                    heapq.heappush(queue, left_node)
                    heapq.heappush(queue, right_node)

        if best_solution is None:
            best_solution = {}

        if status == BranchingStatus.UNSOLVED:
            if best_obj < float('inf'):
                status = BranchingStatus.OPTIMAL
            else:
                status = BranchingStatus.INFEASIBLE

        result = BranchAndPriceResult(
            status=status,
            objective=best_obj if best_obj < float('inf') else None,
            variables=best_solution if save_vars else {},
            iterations=n_nodes_explored,
            runtime_total=time.time() - wall_start,
            history=cg_history,
            n_nodes_explored=n_nodes_explored,
            n_pruned=n_pruned,
            n_infeasible=n_infeasible,
            best_bound=best_bound,
            gap=((best_obj - best_bound) / abs(best_obj)
                 if best_obj != 0 and best_bound > float('-inf')
                 else float('inf')),
            solution_count=1 if best_solution else 0,
            n_columns_generated=total_columns_generated,
            n_pricing_problems=total_pricing_problems,
            column_generation_time=cg_time_total,
        )

        log(f"Finished: status={status.value} objective={best_obj:.6f} "
            f"nodes={n_nodes_explored} cg_time={cg_time_total:.2f}s")

        return result
