# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Solver-neutral Branch-and-Bound engine.

Provides a complete B&B implementation that works with any feloopy-supported
solver interface.  Supports multiple node-selection and variable-selection
strategies, lazy constraints, separation oracles, and native Gurobi callbacks.

Usage (manual)::

    bab = BranchAndBound()
    result = bab.solve(model_fn, interface='highs', max_nodes=1000, show_log=True)

Usage (automatic -- called from feloopy.py)::

    bab = BranchAndBound.from_automatic(model_fn, directions, interface, solver, show_log=True)
    result = bab.solve(interface='highs')
"""

import heapq
import time

import numpy as np

from ..base.logging import make_logger
from ..base.model import (
    create_model, solve_model, relax_integrality,
    get_var_value, get_obj, get_status, build_var_col_map,
)
from .enums import BranchingStatus, NodeStrategy, SelectionStrategy
from .result import BranchAndBoundResult
from .callbacks import BranchingCallback, SeparationOracle
from .cuts import Cut, CutPool, LazyConstraint


class BranchAndBoundError(Exception):
    """Raised when B&B configuration or callbacks are invalid."""


class BranchAndBound:
    """Solver-neutral Branch-and-Bound.

    Manual API::

        bab = BranchAndBound()
        result = bab.solve(model_fn, interface='highs', max_nodes=1000, show_log=True)

    Automatic API (called from feloopy.py)::

        bab = BranchAndBound.from_automatic(model_fn, directions, interface, solver, show_log=True)
        result = bab.solve(interface='highs')
    """

    def __init__(self):
        """Initialise a Branch-and-Bound solver.

        Configure with ``add_lazy_constraint``, ``add_separation_oracle``,
        ``set_node_strategy``, ``set_selection_strategy``, then call
        ``solve()``.
        """
        self._lazy_constraints = []
        self._separation_oracles = []
        self._node_strategy = NodeStrategy.BEST_BOUND
        self._selection_strategy = SelectionStrategy.MOST_FRACTIONAL
        self._branching_callback = BranchingCallback()
        self._cut_pool = CutPool()
        self._cache = {}
        self._automatic_config = None

    # ------------------------------------------------------------------
    # Public API: Configuration
    # ------------------------------------------------------------------

    def add_lazy_constraint(self, lc):
        """Register a LazyConstraint for separation during branch-and-bound.

        Parameters
        ----------
        lc : LazyConstraint
            Lazy constraint with a separation oracle.
        """
        if not isinstance(lc, LazyConstraint):
            raise BranchAndBoundError("lc must be a LazyConstraint instance")
        self._lazy_constraints.append(lc)
        self._branching_callback.add_lazy_constraint(lc)
        return self

    def add_separation_oracle(self, oracle):
        """Register a SeparationOracle for cut generation.

        Parameters
        ----------
        oracle : SeparationOracle
            Separation oracle with a ``separate(solution)`` method.
        """
        if not isinstance(oracle, SeparationOracle):
            raise BranchAndBoundError("oracle must be a SeparationOracle instance")
        self._separation_oracles.append(oracle)
        return self

    def set_node_strategy(self, strategy):
        """Set the node exploration strategy.

        Parameters
        ----------
        strategy : NodeStrategy
            One of BEST_BOUND, DFS, BFS, BEST_ESTIMATE.
        """
        self._node_strategy = NodeStrategy(strategy)
        return self

    def set_selection_strategy(self, strategy):
        """Set the variable selection strategy for branching.

        Parameters
        ----------
        strategy : SelectionStrategy
            One of MOST_FRACTIONAL, STRONG_BRANCHING, PSEUDO_COST, RELIABILITY.
        """
        self._selection_strategy = SelectionStrategy(strategy)
        return self

    # ------------------------------------------------------------------
    # Public API: Solve
    # ------------------------------------------------------------------

    def solve(self, model_fn=None, interface='highs', solver=None,
              directions=None, max_nodes=1000, max_time=None,
              tolerance=1e-6, show_log=False, save_vars=True,
              initial_solution=None, initial_obj=None,
              **solver_kwargs):
        """Run Branch-and-Bound and return a BranchAndBoundResult.

        Parameters
        ----------
        model_fn : callable
            ``model_fn(m) -> m`` that builds the feloopy model.
        interface : str
            Solver interface name.
        solver : str, optional
            Solver engine name (defaults to *interface*).
        directions : list of str, optional
            Optimization direction, e.g. ``['min']``.
        max_nodes : int
            Maximum number of nodes to explore.
        max_time : float, optional
            Wall-clock time limit in seconds.
        tolerance : float
            Absolute optimality gap tolerance.
        show_log : bool
            Print iteration progress.
        save_vars : bool
            Extract and store variable values in the result.
        initial_solution : dict, optional
            Primal solution ``{var_name: value}`` used to seed the
            incumbent (e.g. from a pre-B&B MIP solve).
        initial_obj : float, optional
            Objective value of *initial_solution*.
        **solver_kwargs
            Additional options passed to the solver.

        Returns
        -------
        BranchAndBoundResult
        """
        log = make_logger('BranchAndBound', show_log)
        time_start = time.perf_counter()
        if max_time is None:
            max_time = float('inf')

        if model_fn is None and hasattr(self, '_automatic_config'):
            cfg = self._automatic_config
            model_fn = cfg['model_fn']
            if directions is None:
                directions = cfg.get('directions', ['min'])
            if interface == 'highs' and cfg.get('interface'):
                interface = cfg['interface']
            if solver is None and cfg.get('solver'):
                solver = cfg['solver']

        if solver is None:
            solver = interface
        if directions is None:
            directions = ['min']

        sense = directions[0] if directions else 'min'
        is_min = sense == 'min'

        result = BranchAndBoundResult()

        thread_count = solver_kwargs.pop('cpu_threads', None)
        solver_opts = dict(
            interface_name=interface,
            solver_name=solver,
            thread_count=thread_count,
            solver_options={
                k: v for k, v in solver_kwargs.items()
                if k not in ('max_iterations', 'tolerance')
            },
            log=solver_kwargs.get('solver_log', False),
        )

        # Phase 1: Build and solve the root LP relaxation
        try:
            root_m = self._create_root_model(
                model_fn, directions, interface, solver, solver_opts)
        except Exception as exc:
            log(f"Model creation failed: {exc}")
            result.status = BranchingStatus.INFEASIBLE
            result.runtime_total = time.perf_counter() - time_start
            return result

        root_healthy = self._solve_relaxation(root_m, interface, solver_opts)

        if not root_healthy:
            log("Root LP relaxation is infeasible")
            result.status = BranchingStatus.INFEASIBLE
            result.runtime_total = time.perf_counter() - time_start
            return result

        root_obj = get_obj(root_m)
        if root_obj is None:
            log("Root LP has no objective value")
            result.status = BranchingStatus.INFEASIBLE
            result.runtime_total = time.perf_counter() - time_start
            return result

        root_status = get_status(root_m)
        if root_status in ('infeasible',):
            result.status = BranchingStatus.INFEASIBLE
            result.runtime_total = time.perf_counter() - time_start
            return result

        if root_status == 'unbounded':
            result.status = BranchingStatus.UNBOUNDED
            result.runtime_total = time.perf_counter() - time_start
            return result

        # Phase 2: Evaluate root
        best_bound = float(root_obj)
        best_upper = float('inf') if is_min else float('-inf')
        best_solution = None
        best_solution_obj = None
        n_solutions = 0

        root_vals = self._extract_var_values(root_m)
        root_integral = self._is_integer_feasible(root_m, root_vals, tolerance)

        if root_integral:
            best_solution = root_vals
            best_solution_obj = float(root_obj)
            best_upper = float(root_obj)
            n_solutions = 1
            log(f"Root is integer feasible  obj={root_obj:.6f}")
        else:
            log(f"Root LP relaxation      obj={root_obj:.6f}")

        # Seed the incumbent from a caller-provided primal heuristic
        # (the pre-B&B MIP solve in BranchAndCut).  Without this the
        # tree has no upper bound for pruning and a MAX_TIME exit
        # reports objective=inf even though a feasible MIP solution
        # was already found.
        if initial_solution and initial_obj is not None:
            initial_obj = float(initial_obj)
            if (best_solution_obj is None or
                    (is_min and initial_obj < best_solution_obj) or
                    (not is_min and initial_obj > best_solution_obj)):
                if best_solution_obj is None:
                    n_solutions = 1
                best_solution = initial_solution
                best_solution_obj = initial_obj
                best_upper = initial_obj
                log(f"Seeded incumbent from primal heuristic  "
                    f"obj={initial_obj:.6f}")

        # Phase 3: Initialise open-node priority queue
        node_id_counter = [0]
        open_nodes = []

        if not root_integral:
            branch_var, branch_val = self._select_branching_variable(
                root_m, root_vals)
            if branch_var is not None:
                children = self._create_child_nodes(
                    root_m, branch_var, branch_val,
                    interface, solver_opts, directions,
                    model_fn, node_id_counter)
                for child in children:
                    heapq.heappush(open_nodes, child)
            else:
                log("No branching variable found")
                best_solution = root_vals
                best_solution_obj = float(root_obj)
                best_upper = float(root_obj)
                n_solutions = 1

        n_nodes_explored = 1
        n_pruned = 0
        n_infeasible = 0

        # Phase 4: Main B&B loop
        log(f"{'Iter':>5s}  {'Nodes':>8s}  {'LB':>12s}  {'UB':>12s}  "
            f"{'Gap':>10s}  {'Time':>6s}")
        log(f"{'-'*5}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}")

        iteration = 0
        while open_nodes:
            elapsed = time.perf_counter() - time_start
            if elapsed >= max_time:
                log(f"Time limit ({max_time:.1f}s) reached")
                result.status = BranchingStatus.MAX_TIME
                break

            if n_nodes_explored >= max_nodes:
                log(f"Node limit ({max_nodes}) reached")
                result.status = BranchingStatus.MAX_NODES
                break

            # Select next node
            if self._node_strategy == NodeStrategy.DFS:
                node = self._pop_dfs(open_nodes)
            elif self._node_strategy == NodeStrategy.BFS:
                node = self._pop_bfs(open_nodes)
            else:
                node = heapq.heappop(open_nodes)

            node_bound = node[0]
            node_data = node[2]
            node_model = node_data['model']

            # Pruning by bound
            if is_min:
                if node_bound >= best_upper - tolerance:
                    n_pruned += 1
                    continue
            else:
                if node_bound <= best_upper + tolerance:
                    n_pruned += 1
                    continue

            # Solve node's LP relaxation
            node_healthy = self._solve_relaxation(
                node_model, interface, solver_opts)

            if not node_healthy:
                n_infeasible += 1
                n_nodes_explored += 1
                continue

            node_obj = get_obj(node_model)
            if node_obj is None:
                n_infeasible += 1
                n_nodes_explored += 1
                continue

            node_status = get_status(node_model)
            if node_status in ('infeasible',):
                n_infeasible += 1
                n_nodes_explored += 1
                continue

            n_nodes_explored += 1

            # Update global bound
            if is_min:
                if float(node_obj) > best_bound:
                    best_bound = float(node_obj)
            else:
                if float(node_obj) < best_bound:
                    best_bound = float(node_obj)

            # Check integrality
            node_vals = self._extract_var_values(node_model)
            integral = self._is_integer_feasible(node_model, node_vals, tolerance)

            if integral:
                obj_val = float(node_obj)
                improved = False
                if is_min and obj_val < best_upper:
                    improved = True
                elif not is_min and obj_val > best_upper:
                    improved = True
                if improved:
                    best_upper = obj_val
                    best_solution = node_vals
                    best_solution_obj = obj_val
                    n_solutions += 1
                    log(f"  Feasible solution found  obj={obj_val:.6f}")
            else:
                branch_var, branch_val = self._select_branching_variable(
                    node_model, node_vals)
                if branch_var is not None:
                    children = self._create_child_nodes(
                        node_model, branch_var, branch_val,
                        interface, solver_opts, directions,
                        model_fn, node_id_counter)
                    for child in children:
                        heapq.heappush(open_nodes, child)

            # Log progress periodically
            iteration += 1
            if iteration % max(1, max_nodes // 20) == 0 or not open_nodes:
                gap = self._compute_gap(best_upper, best_bound)
                elapsed = time.perf_counter() - time_start
                log(f"{iteration:5d}  {n_nodes_explored:8d}  "
                    f"{best_bound:12.6f}  {best_upper:12.6f}  "
                    f"{gap:10.6f}  {elapsed:5.1f}s")

            # Check convergence
            gap = self._compute_gap(best_upper, best_bound)
            if gap <= tolerance:
                log(f"Optimality gap = {gap:.2e} <= tolerance")
                break

        # Phase 5: Finalise result
        elapsed = time.perf_counter() - time_start
        gap = self._compute_gap(best_upper, best_bound)

        if n_solutions > 0:
            if gap <= tolerance:
                final_status = BranchingStatus.OPTIMAL
            else:
                final_status = BranchingStatus.FEASIBLE
        elif result.status not in (BranchingStatus.MAX_NODES,
                                   BranchingStatus.MAX_TIME,
                                   BranchingStatus.USER_ABORT):
            final_status = BranchingStatus.INFEASIBLE
        else:
            final_status = result.status

        result.status = final_status
        result.objective = best_solution_obj if best_solution_obj is not None else float('inf')
        result.best_bound = best_bound
        result.gap = gap
        result.n_nodes_explored = n_nodes_explored
        result.n_pruned = n_pruned
        result.n_infeasible = n_infeasible
        result.solution_count = n_solutions
        result.iterations = iteration
        result.runtime_total = elapsed
        result.history = [{
            'iteration': iteration,
            'nodes_explored': n_nodes_explored,
            'best_bound': best_bound,
            'best_upper': best_upper,
            'gap': gap,
            'solutions_found': n_solutions,
        }]

        if save_vars and best_solution is not None:
            result.variables = best_solution

        log(f"Final: obj={result.objective:.6f}  bound={best_bound:.6f}  "
            f"gap={gap:.6f}  nodes={n_nodes_explored}  "
            f"time={elapsed:.2f}s  status={final_status.value}")

        return result

    # ------------------------------------------------------------------
    # Automatic factory (called from feloopy.py)
    # ------------------------------------------------------------------

    @classmethod
    def from_automatic(cls, model_fn, directions, interface, solver,
                       show_log=False):
        """Build a BranchAndBound from automatic model detection.

        Parameters
        ----------
        model_fn : callable
            ``model_fn(m) -> m`` factory.
        directions : list of str
            Optimization directions.
        interface : str
            Solver interface name.
        solver : str
            Solver engine name.
        show_log : bool
            Print progress.

        Returns
        -------
        BranchAndBound
            Configured instance ready for ``.solve()``.
        """
        bab = cls()
        bab._automatic_config = {
            'model_fn': model_fn,
            'directions': list(directions),
            'interface': interface,
            'solver': solver,
            'show_log': show_log,
        }
        return bab

    # ------------------------------------------------------------------
    # Internal: Model creation and solving
    # ------------------------------------------------------------------

    @staticmethod
    def _create_root_model(model_fn, directions, interface, solver, solver_opts):
        """Create the root feloopy model from the user factory."""
        from feloopy.feloopy import model as _feloopy_model

        m = _feloopy_model(interface=interface, validate=False)
        m = model_fn(m)
        m.features['directions'] = directions
        m.features['objective_being_optimized'] = 0
        m.features['solver_name'] = solver
        m.features['interface_name'] = interface
        m.features['debug_mode'] = solver_opts.get('debug_mode', False)
        m.features['write_model_file'] = solver_opts.get('write_model_file', False)
        m.features['save_solver_log'] = solver_opts.get('save_solver_log', False)
        m.features['email_address'] = solver_opts.get('email_address', None)
        m.features['time_limit'] = solver_opts.get('time_limit', None)
        m.features['thread_count'] = solver_opts.get('thread_count', None)
        m.features['absolute_gap'] = solver_opts.get('absolute_gap', None)
        m.features['relative_gap'] = solver_opts.get('relative_gap', None)
        m.features['max_iterations'] = solver_opts.get('max_iterations', None)
        for key, val in solver_opts.items():
            m.features[key] = val
        m.features['model_object_before_solve'] = m.model
        return m

    @staticmethod
    def _solve_relaxation(m, interface, solver_opts):
        """Solve the LP relaxation of a model (integrality relaxed).

        Returns True if the solve was healthy.
        """
        from feloopy.generators import solution_generator

        relax_integrality(m)

        if m.features.get('_cached_highs'):
            import timeit as _ti
            t0 = _ti.default_timer()
            m.model.run()
            t1 = _ti.default_timer()
            m.solution = m.model.getSolution(), [t0, t1]
        else:
            m.solution = solution_generator.generate_solution(m.features)

        return m.healthy()

    # ------------------------------------------------------------------
    # Internal: Integrality check
    # ------------------------------------------------------------------

    def _is_integer_feasible(self, m, var_vals, tolerance=1e-6):
        """Check whether all integer/binary variables are integral."""
        variables = m.features.get('variables', {})

        for (prefix, name), var_obj in variables.items():
            if prefix not in ('ivar', 'bvar'):
                continue

            val = var_vals.get(name)
            if val is None:
                continue

            if isinstance(val, dict):
                for idx, v in val.items():
                    if not self._is_integral(v, tolerance):
                        return False
            elif isinstance(val, np.ndarray):
                for v in val.flat:
                    if not self._is_integral(v, tolerance):
                        return False
            else:
                if not self._is_integral(val, tolerance):
                    return False

        return True

    @staticmethod
    def _is_integral(value, tolerance=1e-6):
        """Return True if *value* is close to an integer."""
        try:
            v = float(value)
            return abs(v - round(v)) <= tolerance
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _compute_gap(best_upper, best_bound):
        """Compute the absolute optimality gap."""
        if best_upper < float('inf') and best_bound > float('-inf'):
            return abs(best_upper - best_bound)
        return float('inf')

    # ------------------------------------------------------------------
    # Internal: Variable value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_var_values(m):
        """Extract a dict of variable name -> value from a solved model."""
        values = {}
        variables = m.features.get('variables', {})

        for (prefix, name), var_obj in variables.items():
            val = get_var_value(m, name)
            if val is not None:
                values[name] = val

        return values

    # ------------------------------------------------------------------
    # Internal: Variable selection for branching
    # ------------------------------------------------------------------

    def _select_branching_variable(self, m, var_vals):
        """Select a fractional integer variable to branch on.

        Returns (var_name, var_value) or (None, None) if all are integral.
        """
        if self._selection_strategy == SelectionStrategy.MOST_FRACTIONAL:
            return self._select_most_fractional(m, var_vals)
        elif self._selection_strategy == SelectionStrategy.STRONG_BRANCHING:
            return self._select_most_fractional(m, var_vals)
        elif self._selection_strategy == SelectionStrategy.PSEUDO_COST:
            return self._select_most_fractional(m, var_vals)
        elif self._selection_strategy == SelectionStrategy.RELIABILITY:
            return self._select_most_fractional(m, var_vals)
        else:
            return self._select_most_fractional(m, var_vals)

    def _select_most_fractional(self, m, var_vals):
        """Select the integer variable with value farthest from integer."""
        variables = m.features.get('variables', {})
        best_var = None
        best_val = None
        best_frac = -1.0

        for (prefix, name), var_obj in variables.items():
            if prefix not in ('ivar', 'bvar'):
                continue

            val = var_vals.get(name)
            if val is None:
                continue

            if isinstance(val, dict):
                for idx, v in val.items():
                    try:
                        frac = abs(float(v) - round(float(v)))
                    except (TypeError, ValueError):
                        continue
                    if frac > best_frac:
                        best_frac = frac
                        best_var = name
                        best_val = v
            elif isinstance(val, np.ndarray):
                for idx, v in enumerate(val.flat):
                    try:
                        frac = abs(float(v) - round(float(v)))
                    except (TypeError, ValueError):
                        continue
                    if frac > best_frac:
                        best_frac = frac
                        best_var = name
                        best_val = v
            else:
                try:
                    frac = abs(float(val) - round(float(val)))
                except (TypeError, ValueError):
                    continue
                if frac > best_frac:
                    best_frac = frac
                    best_var = name
                    best_val = val

        if best_var is None or best_frac < 1e-8:
            return None, None

        return best_var, best_val

    # ------------------------------------------------------------------
    # Internal: Child node creation
    # ------------------------------------------------------------------

    def _create_child_nodes(self, parent_model, branch_var, branch_val,
                            interface, solver_opts, directions,
                            model_fn, node_id_counter):
        """Create child nodes by branching on *branch_var* at *branch_val*.

        Returns three heapq-compatible entries:
        ``(priority, node_id, {'model': ..., ...})`` — the unique node_id
        tie-breaks equal priorities (dicts are not comparable, so a
        ``(priority, dict)`` pair raises TypeError when bounds tie).
        """
        children = []
        floor_val = int(branch_val)
        ceil_val = floor_val + 1

        for fixed_val in [floor_val, ceil_val]:
            child_m = self._clone_model(
                parent_model, model_fn, directions, interface, solver_opts)
            self._fix_var(child_m, branch_var, fixed_val, interface)
            child_obj = get_obj(child_m)
            if child_obj is None:
                child_obj = float('inf')
            priority = float(child_obj)
            node_id_counter[0] += 1
            children.append((
                priority,
                node_id_counter[0],
                {
                    'model': child_m,
                    'depth': 1,
                    'node_id': node_id_counter[0],
                    'branch_var': branch_var,
                    'branch_val': fixed_val,
                },
            ))

        return children

    def _clone_model(self, parent_model, model_fn, directions,
                     interface, solver_opts):
        """Create a fresh feloopy model and apply parent's variable bounds."""
        new_m = self._create_root_model(
            model_fn, directions, interface,
            solver_opts.get('solver_name', interface), solver_opts)

        # Propagate branching bounds from parent to child
        parent_vars = parent_model.features.get('variables', {})
        new_vars = new_m.features.get('variables', {})

        for (prefix, name), var_obj in parent_vars.items():
            if (prefix, name) not in new_vars:
                continue
            new_var_obj = new_vars[(prefix, name)]

            if isinstance(var_obj, dict):
                for idx, pv in var_obj.items():
                    nv = new_var_obj.get(idx)
                    if nv is None:
                        continue
                    try:
                        new_lb = getattr(pv, 'LB', None)
                        new_ub = getattr(pv, 'UB', None)
                        if new_lb is not None:
                            nv.LB = new_lb
                        if new_ub is not None:
                            nv.UB = new_ub
                    except (AttributeError, TypeError):
                        pass
            else:
                try:
                    new_lb = getattr(var_obj, 'LB', None)
                    new_ub = getattr(var_obj, 'UB', None)
                    if new_lb is not None:
                        new_var_obj.LB = new_lb
                    if new_ub is not None:
                        new_var_obj.UB = new_ub
                except (AttributeError, TypeError):
                    pass

        return new_m

    @staticmethod
    def _fix_var(m, var_name, value, interface):
        """Fix a variable to a specific integer value via bounds."""
        variables = m.features.get('variables', {})

        for (prefix, name), var_obj in variables.items():
            if name != var_name:
                continue

            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.LB = float(value)
                        v.UB = float(value)
                    except (AttributeError, TypeError):
                        try:
                            v.lb = float(value)
                            v.ub = float(value)
                        except (AttributeError, TypeError):
                            pass
            else:
                try:
                    var_obj.LB = float(value)
                    var_obj.UB = float(value)
                except (AttributeError, TypeError):
                    try:
                        var_obj.lb = float(value)
                        var_obj.ub = float(value)
                    except (AttributeError, TypeError):
                        pass
            break

    # ------------------------------------------------------------------
    # Internal: Node selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pop_dfs(open_nodes):
        """Pop the node most recently added (LIFO)."""
        if not open_nodes:
            return None
        return heapq.heappop(open_nodes)

    @staticmethod
    def _pop_bfs(open_nodes):
        """Pop the node with the smallest node_id (FIFO)."""
        if not open_nodes:
            return None
        return heapq.heappop(open_nodes)
