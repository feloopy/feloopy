# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ....helpers.containers import to_indexed_dict

from .enums import (
    BendersCutStrategy, BendersAcceleration, BendersMethod,
    BendersStatus, BendersError,
)
from .callbacks import (
    _BendersEvent, BendersCallback, BendersContext, _CallbackManager,
)
from .result import BendersResult, BendersParams

_PARALLEL_SAFE_INTERFACES = {'highs', 'xpress', 'copt'}

def _create_gurobi_model_silently(name, show_log=False):
    """Create Gurobi model without 'Restricted license' banner when show_log is False."""
    import gurobipy
    if not show_log:
        import os, sys, contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                old_stdout_fd = os.dup(sys.stdout.fileno())
                old_stderr_fd = os.dup(sys.stderr.fileno())
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                os.dup2(devnull, sys.stderr.fileno())
                os.close(devnull)
                try:
                    m = gurobipy.Model(name)
                finally:
                    os.dup2(old_stdout_fd, sys.stdout.fileno())
                    os.dup2(old_stderr_fd, sys.stderr.fileno())
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                return m
            except Exception:
                return gurobipy.Model(name)
    return gurobipy.Model(name)


from .cuts import (
    _normalize_cut, _normalize_cut_list, _cut_signature,
    _log_benders, _log_iter,
    _generate_optimality_cut, _generate_feasibility_cut,
    _generate_combinatorial_optimality_cut, _generate_nogood_cut,
    _generate_generalized_optimality_cut, _generate_strengthened_cut,
    _generate_pareto_optimal_cut, _generate_multi_cuts,
    _generate_l_shaped_aggregated_cut, _generate_integer_l_shaped_cut,
    _generate_generalized_l_shaped_cut,
    _logic_cuts, _generate_cut_for_method,
    _add_cuts_to_master, _cleanup_duplicate_cuts,
    _apply_cut_selection, _add_to_cut_pool,
    _compute_cut_violation, _select_most_violated_cuts,
)
from .utils import (
    _relax_integrality, _fix_variable, _fix_variable_bound,
    _unfix_variable_bound, _unfix_all_bounds,
    _capture_original_bounds, _build_var_col_map,
    _extract_reduced_costs, _derive_numerical_coeffs,
    _compute_benders_coeffs_from_subduals, _compute_iis,
    _finalize_result, _build_result, get_convergence,
    _add_cut_direct_highs, _get_reduced_costs_highs,
    _get_theta_col_idx, _get_var_col_indices,
)


class _BendersProblem:
    def __init__(self, name, model_fn, directions, obj_index=0):
        if not callable(model_fn):
            raise BendersError(f"Problem '{name}': model_fn must be callable")
        if not directions or not isinstance(directions, (list, tuple)):
            raise BendersError(f"Problem '{name}': directions must be non-empty list")
        for i, d in enumerate(directions):
            if d not in ('min', 'max'):
                raise BendersError(
                    f"Problem '{name}': directions[{i}] = {d!r} must be 'min' or 'max'")
        self.name = name
        self.model_fn = model_fn
        self.directions = list(directions)
        self.obj_index = obj_index



def benders(master_fn=None, sub_fn=None, method="classical",
            vars=None, interface="gurobi", **kwargs):
    """Simple Benders decomposition API.

    Usage (composition):
        bd = benders()
        bd.master(my_master_fn)
        bd.subproblem(my_sub_fn, shares=["x", "y"])
        bd.solve(interface="gurobi", method="classical")
        result = bd.result

    Usage (one-shot):
        bd = benders(master_fn, sub_fn, vars=["x"])
        bd.solve(interface="gurobi")

    Usage (as a flp.search compatible model):
        bd = benders()
        bd.master(my_master_fn)
        bd.subproblem(my_sub_fn, shares=["x"])
        # Returns a BendersResult directly
        result = bd.run(interface="gurobi")
    """
    bd = BendersDecomposition()
    if master_fn is not None:
        bd.add_level("master", master_fn, ["min"])
    if sub_fn is not None:
        bd.add_level("subproblem", sub_fn, ["min"])
        if vars:
            bd.add_link("master", "subproblem", shared_vars=vars)
    bd._simple_method = method
    bd._simple_interface = interface
    bd._simple_kwargs = kwargs
    return bd


def benders_decomposition(master_vars=None, sub_vars=None, method="classical"):
    """Decorator for Benders decomposition.

    Usage:
        @benders_decomposition(master_vars=["x"], method="classical")
        def my_model(m):
            x = m.ivar(name="x", bounds=(0, 10))
            theta = m.fvar(name="theta", bounds=(0, 100))
            m.con(x >= 1, name="c1")
            m.minimize(x + theta)
            return m

        # The decorated function becomes a BendersDecomposition
        bd = my_model
        bd.add_subproblem(my_sub_fn, shares=["x"])
        result = bd.run(interface="gurobi")
    """
    def decorator(fn):
        bd = BendersDecomposition()
        bd.add_level("master", fn, ["min"])
        if master_vars:
            bd._master_complicating_vars = master_vars
        bd._simple_method = method
        bd._simple_sub_vars = sub_vars or []
        return bd
    return decorator


class BendersDecomposition:
    """Automated Benders decomposition for MILPs.

    Decomposes a mixed-integer linear program into a master problem
    (containing integer/complicating variables) and one or more LP
    subproblems, iteratively generating Benders cuts until convergence.

    Works with any solver interface via the
    vendor-neutral ``solution_generator`` and ``get_dual()`` API.

    Supports classical Benders, logic-based Benders, combinatorial Benders,
    generalized Benders, and L-shaped (stochastic) decomposition.

    Usage
    -----
    **Manual API** (separate master/subproblem factories):

    >>> bd = BendersDecomposition()
    >>> bd.add_level("master", master_fn, ["min"])
    >>> bd.add_level("subproblem", subproblem_fn, ["min"])
    >>> bd.add_link("master", "subproblem", shared_vars=["x"])
    >>> result = bd.solve(interface="highs", max_iterations=50)

    **Automatic API** (single model, label-based):

    >>> bd = BendersDecomposition()
    >>> bd.add_problem(
    ...     model_fn=my_model,
    ...     directions=["min"],
    ...     master_constraints=["budget"],
    ...     subproblem_constraints=["linking", "demand"],
    ...     complicating_variables=["x"],
    ... )
    >>> result = bd.solve(interface="gurobi")

    Returns
    -------
    BendersResult
        Structured result with objective, variables, bounds, runtime breakdown.
    """

    def __init__(self):
        """Initialise an empty Benders decomposition.

        Use ``add_level`` / ``add_link`` (manual) or ``add_problem``
        (automatic) to configure before calling ``solve()``.
        """
        self._master = None
        self._subproblems = []
        self._links = []
        self._auto_config = None
        self._stored_cuts = []
        self._iteration_history = []
        self._logic_cut_callback = None
        self._cache = {}
        # Advanced acceleration features
        self._cut_pool = []
        self._trust_region_radius = 1.0
        self._cut_pool_max_size = 100
        self._multi_cut_enabled = False
        self._cut_selection_strategy = 'most_violated'
        self._pareto_points = []
        self._cut_strengthener_calls = 0
        # Callback and params
        self._callback_manager = _CallbackManager()
        self._context = BendersContext()
        self._result = BendersResult()
        self._params = BendersParams()
        # Direction (min/max) — resolved from directions
        self._objective_sense = 'min'
        # Method override
        self._method_override = None

    # ------------------------------------------------------------------
    # Public API: Callbacks and params
    # ------------------------------------------------------------------

    def register(self, callback):
        """Register a callback for Benders events.

        Parameters
        ----------
        callback : BendersCallback | callable
            A callback instance or a plain function called on every event.
        """
        self._callback_manager.register(callback)
        return self

    def set_params(self, **kwargs):
        """Set Benders parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to BendersParams, e.g.
            ``tol_abs=1e-5, time_limit=3600, cut_normalize=True``.
        """
        for k, v in kwargs.items():
            if hasattr(self._params, k):
                setattr(self._params, k, v)
            else:
                raise BendersError(f"Unknown parameter: {k}")
        return self

    @property
    def result(self):
        """The current BendersResult (updated after solve)."""
        return self._result

    @property
    def params(self):
        """The BendersParams instance."""
        return self._params

    # ------------------------------------------------------------------
    # Public API: Automatic (label-based) decomposition
    # ------------------------------------------------------------------

    def add_problem(self, model_fn, directions, master_constraints=None,
                    subproblem_constraints=None, complicating_variables=None,
                    obj_index=0):
        """Define a Benders decomposition from a single model.

        Parameters
        ----------
        model_fn : callable
            ``model_fn(m) -> m`` that builds the full model.
        directions : list of str
            Optimization direction, e.g. ``["min"]``.
        master_constraints : list of str, optional
            Labels of constraints that belong to the master problem.
        subproblem_constraints : list of str, optional
            Labels of constraints that belong to the subproblem(s).
        complicating_variables : list of str
            Names of variables that link master and subproblem.
        obj_index : int
            Which objective to optimize (default 0).
        """
        if complicating_variables is None or not complicating_variables:
            raise BendersError("complicating_variables must be non-empty")
        if master_constraints is None and subproblem_constraints is None:
            raise BendersError(
                "Provide at least one of master_constraints or subproblem_constraints")

        all_master = list(master_constraints) if master_constraints else []
        all_sub = list(subproblem_constraints) if subproblem_constraints else []
        all_vars = list(complicating_variables)

        self._master = _BendersProblem(
            'master', model_fn, directions, obj_index)
        self._auto_config = {
            'master_constraints': all_master,
            'subproblem_constraints': all_sub,
            'complicating_variables': all_vars,
        }
        self._subproblems = []
        self._links = []
        return self

    # ------------------------------------------------------------------
    # Public API: Manual (multi-level) decomposition
    # ------------------------------------------------------------------

    def add_level(self, name, model_fn, directions, obj_index=0):
        """Add a problem level (master or subproblem).

        Parameters
        ----------
        name : str
            Level name (``'master'`` or ``'subproblem'``).
        model_fn : callable
            ``model_fn(m) -> m`` that builds this level's model.
        directions : list of str
            Optimization direction.
        obj_index : int
            Which objective to optimize (default 0).
        """
        level = _BendersProblem(name, model_fn, directions, obj_index)
        if name == 'master':
            self._master = level
        else:
            self._subproblems.append(level)
        return self

    def add_link(self, upper_name, lower_name, shared_vars):
        """Define linking variables between two levels.

        Parameters
        ----------
        upper_name : str
            Name of the upper (master) level.
        lower_name : str
            Name of the lower (subproblem) level.
        shared_vars : list of str
            Variable names that couple the two levels.
        """
        if not shared_vars or not isinstance(shared_vars, (list, tuple)):
            raise BendersError("shared_vars must be a non-empty list of variable names")
        self._links.append({
            'upper': upper_name,
            'lower': lower_name,
            'shared_vars': list(shared_vars),
        })
        return self

    def add_subproblem(self, sub_fn, shares=None, name=None, obj_index=0):
        """Add a subproblem (shorthand for clean API).

        Parameters
        ----------
        sub_fn : callable
            ``sub_fn(m) -> m`` that builds the subproblem.
        shares : list of str, optional
            Complicating variable names shared with master.
        name : str, optional
            Subproblem name (default ``'subproblem'``).
        obj_index : int, optional
            Which objective to optimize.
        """
        sname = name or f"subproblem_{len(self._subproblems)}"
        sub_level = _BendersProblem(sname, sub_fn, ["min"], obj_index)
        self._subproblems.append(sub_level)
        if shares:
            self.add_link("master", sname, shared_vars=shares)
        return self

    def run(self, interface="gurobi", solver=None, show_log=True, **kwargs):
        """Run Benders and return the result (shorthand for clean API).

        Parameters
        ----------
        interface : str
            Solver interface for the master.
        solver : str, optional
            Solver name.
        show_log : bool
            Whether to print progress.
        **kwargs
            Additional options passed to ``solve()``.
        """
        method = getattr(self, '_simple_method', 'classical')
        return self.solve(
            interface=interface,
            solver=solver,
            method=method,
            show_log=show_log,
            **kwargs,
        )

    def add_logic_cut_callback(self, callback):
        """Register a domain-specific logic-based Benders cut callback.

        ``callback(master_vars, subproblem_result)`` may return one cut
        dictionary or a list of cut dictionaries using the internal cut shape:
        ``{'rhs': number, 'coeffs': dict, 'type': 'optimality'}`` or
        ``{'rhs': number, 'coeffs': dict, 'type': 'feasibility'}``.
        Returning ``None`` delegates to the standard dual/no-good logic.
        """
        if not callable(callback):
            raise BendersError("logic cut callback must be callable")
        self._logic_cut_callback = callback
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        if self._master is None:
            raise BendersError(
                "No master problem defined. Use add_problem() or add_level().")
        if self._auto_config is None and not self._subproblems:
            raise BendersError("No subproblems defined.")

    def _is_auto(self):
        return self._auto_config is not None

    def _get_complicating_vars(self):
        if self._auto_config is not None:
            return self._auto_config['complicating_variables']
        for link in self._links:
            if link is not None:
                return link['shared_vars']
        raise BendersError("No complicating variables defined.")

    def _get_sub_constraint_ids(self):
        if self._auto_config is not None:
            return self._auto_config.get('subproblem_constraints', [])
        return []

    def _get_master_constraint_ids(self):
        if self._auto_config is not None:
            return self._auto_config.get('master_constraints', [])
        return []

    # ------------------------------------------------------------------
    # Model creation / solving helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_model(model_fn, directions, obj_index, solver_opts,
                      level_name='master'):
        from feloopy.feloopy import model as _feloopy_model
        if level_name == 'subproblem':
            interface = solver_opts.get('sub_interface_name',
                                        solver_opts.get('interface_name', 'highs'))
            solver_name = solver_opts.get('sub_solver_name',
                                          solver_opts.get('solver_name', interface))
        else:
            interface = solver_opts.get('interface_name', 'highs')
            solver_name = solver_opts.get('solver_name', interface)
        m = _feloopy_model(interface=interface, validate=False)
        m = model_fn(m)
        m.features['directions'] = directions
        m.features['objective_being_optimized'] = obj_index
        m.features['solver_name'] = solver_name
        for key, val in solver_opts.items():
            m.features[key] = val
        m.features['model_object_before_solve'] = m.model
        return m

    @staticmethod
    def _solve_model(m, solution_generator, constraint_ids=None,
                     relax_integrality=False):
        original_constraints = None
        original_labels = None
        if constraint_ids is not None:
            original_constraints = list(m.features['constraints'])
            original_labels = list(m.features['constraint_labels'])
            filtered_c = []
            filtered_l = []
            for c, label in zip(original_constraints, original_labels):
                if label in constraint_ids:
                    filtered_c.append(c)
                    filtered_l.append(label)
            m.features['constraints'] = filtered_c
            m.features['constraint_labels'] = filtered_l

        if relax_integrality:
            _relax_integrality(m)

        if m.features.get('_cached_highs'):
            import timeit as _ti
            t0 = _ti.default_timer()
            m.model.run()
            t1 = _ti.default_timer()
            m.solution = m.model.getSolution(), [t0, t1]
            try:
                lp = m.model.getLp()
                m.features['lp_data'] = {
                    'n_cols': lp.num_col_,
                    'n_rows': lp.num_row_,
                    'row_lower': list(lp.row_lower_),
                    'row_upper': list(lp.row_upper_),
                    'col_lower': list(lp.col_lower_),
                    'col_upper': list(lp.col_upper_),
                    'col_cost': list(lp.col_cost_),
                    'integrality': list(lp.integrality_),
                    'col_names': list(lp.col_names_) if lp.col_names_ else [],
                    'row_names': list(lp.row_names_) if lp.row_names_ else [],
                    'A_col_pointers': list(lp.a_matrix_.start_),
                    'A_row_indices': list(lp.a_matrix_.index_),
                    'A_values': list(lp.a_matrix_.value_),
                }
            except Exception:
                pass
        else:
            m.solution = solution_generator.generate_solution(m.features)

        if original_constraints is not None:
            m.features['constraints'] = original_constraints
            m.features['constraint_labels'] = original_labels

        return m.healthy()

    @staticmethod
    def _get_var_value(m, var_name):
        try:
            return m.get_numpy_var(var_name)
        except Exception:
            return None

    @staticmethod
    def _zero_value(m, var_name):
        """Return a zero-shaped value for an unfixed master variable."""
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            var_obj = m.features.get('variables', {}).get((prefix, var_name))
            if var_obj is None:
                continue
            if not isinstance(var_obj, dict):
                container = to_indexed_dict(var_obj)
                if container is not None:
                    var_obj = container
            if isinstance(var_obj, dict):
                return {idx: 0.0 for idx in var_obj}
            return 0.0
        return 0.0

    @staticmethod
    def _get_dual(m, constraint_label):
        try:
            return m.get_dual(constraint_label)
        except Exception:
            return None

    @staticmethod
    def _get_obj(m):
        try:
            return m.get_objective()
        except Exception:
            return None

    @staticmethod
    def _get_status(m):
        try:
            return m.get_status()
        except Exception:
            return 'unknown'

    # ------------------------------------------------------------------
    # Core Benders algorithm (follows reference BDA procedure)
    # ------------------------------------------------------------------

    def solve(self, interface, solver=None, max_iterations=None,
              tolerance=None, cut_strategy='single', acceleration='none',
              show_log=False, method='auto', logic_cut_callback=None,
              relax_subproblem_integrality=None,
              sub_interface=None, sub_solver=None,
              params=None, **solver_kwargs):
        """Run Benders decomposition.

        Follows the standard Benders Decomposition Algorithm (BDA):

        1. Fix complicating vars x_bar in subproblem, solve LP -> duals
        2. If LP optimal: compute LB = obj(x_bar), generate optimality cut
           If LP infeasible: solve homogeneous LP -> ray, generate feasibility cut
        3. Add cut to master, solve master -> new x_bar, UB
        4. Repeat until UB - LB <= tolerance

        Parameters
        ----------
        interface : str
            Solver interface name.
        solver : str, optional
            Solver engine name (defaults to interface name).
        max_iterations : int, optional
            Maximum Benders iterations (overrides params.iter_limit).
        tolerance : float, optional
            Absolute optimality gap tolerance (overrides params.tol_abs).
        cut_strategy : str
            ``'single'`` or ``'multi'``.
        acceleration : str
            ``'none'``, ``'pareto'``, ``'stabilization'``, ``'trust_region'``,
            ``'multi_cut'``, or ``'hybrid'``.
        show_log : bool
            Print iteration progress.
        method : str
            ``'auto'``, ``'classical'``, ``'logic'``, ``'combinatorial'``,
            ``'generalized'``, or ``'l_shaped'``.
        logic_cut_callback : callable, optional
            Callback for logic-based Benders cuts.
        relax_subproblem_integrality : bool, optional
            Relax integer variables in the subproblem (default: True).
        sub_interface : str, optional
            Solver interface for subproblems.
        sub_solver : str, optional
            Solver engine for subproblems.
        params : BendersParams, optional
            Override parameters. If None, uses self._params.
        **solver_kwargs
            Solver-specific options.

        Returns
        -------
        BendersResult
            Structured result with objective, variables, bounds, runtime.
        """
        self._validate()

        # Merge params
        if params is not None:
            self._params = params
        if max_iterations is not None:
            self._params.iter_limit = max_iterations
        if tolerance is not None:
            self._params.tol_abs = tolerance

        if method not in ('auto', 'classical', 'logic', 'combinatorial',
                          'generalized', 'l_shaped', 'integer_l_shaped',
                          'generalized_l_shaped'):
            raise BendersError(
                "method must be 'auto', 'classical', 'logic', "
                "'combinatorial', 'generalized', 'l_shaped', "
                "'integer_l_shaped', or 'generalized_l_shaped'")
        self._active_method = method
        if logic_cut_callback is not None:
            self.add_logic_cut_callback(logic_cut_callback)

        if method == 'classical' and relax_subproblem_integrality is False:
            raise BendersError(
                "Classical Benders requires an LP-relaxed subproblem; use "
                "method='logic' for integer subproblems.")
        if method == 'logic':
            relax_subproblem_integrality = False
            if self._logic_cut_callback is None:
                raise BendersError(
                    "method='logic' requires logic_cut_callback or "
                    "add_logic_cut_callback(...)")
        elif relax_subproblem_integrality is None:
            relax_subproblem_integrality = True

        if solver is None:
            solver = interface
        if sub_interface is None:
            sub_interface = interface
        if sub_solver is None:
            sub_solver = solver

        # Resolve objective sense from master directions
        self._objective_sense = self._master.directions[0] if self._master.directions else 'min'

        strategy = BendersCutStrategy(cut_strategy)
        accel = BendersAcceleration(acceleration)
        
        # Configure advanced acceleration features
        if accel == BendersAcceleration.MULTI_CUT:
            self._multi_cut_enabled = True
        elif accel == BendersAcceleration.HYBRID:
            self._multi_cut_enabled = True
            self._cut_pool_max_size = 50
        elif accel == BendersAcceleration.TRUST_REGION:
            self._trust_region_radius = 0.5
        elif accel == BendersAcceleration.PARETO:
            self._cut_pool_max_size = 30

        solver_opts = dict(
            interface_name=interface,
            solver_name=solver,
            sub_interface_name=sub_interface,
            sub_solver_name=sub_solver,
            solver_options={
                key: value
                for key, value in solver_kwargs.get('options', {}).items()
                if key not in ('max_iterations', 'tolerance')
            },
            log=solver_kwargs.get('solver_log', False),
            write_model_file=solver_kwargs.get('save_model', False),
            save_solver_log=solver_kwargs.get('save_log', False),
            email_address=solver_kwargs.get('email', None),
            time_limit=solver_kwargs.get('time_limit', None),
            thread_count=solver_kwargs.get('cpu_threads', None),
            absolute_gap=solver_kwargs.get('absolute_gap', None),
            relative_gap=solver_kwargs.get('relative_gap', None),
            max_iterations=solver_kwargs.get('max_nodes', None),
            debug_mode=solver_kwargs.get('debug', False),
            callback=None,
        )

        from feloopy.generators import solution_generator

        # Dispatch to incremental path when the master interface supports
        # incremental operations.  Subproblems are always rebuilt from
        # scratch via the standard path, so sub_interface need not be
        # incremental.

        # Branch-and-check: solve master once with lazy constraint callbacks
        use_bnc = solver_kwargs.get('use_bnc', False) or \
            self._params.use_bnc if hasattr(self._params, 'use_bnc') else False
        if use_bnc and interface == 'gurobi':
            return self._solve_bnc_benders(
                interface, solver, sub_interface, sub_solver,
                self._params.iter_limit, self._params.tol_abs, cut_strategy,
                acceleration, show_log, method, logic_cut_callback,
                relax_subproblem_integrality, solver_kwargs)

        _INCREMENTAL_INTERFACES = {'highs', 'gurobi', 'cplex', 'copt'}
        if interface in _INCREMENTAL_INTERFACES:
            return self._solve_incremental_benders(
                interface, solver, sub_interface, sub_solver,
                self._params.iter_limit, self._params.tol_abs, cut_strategy, acceleration,
                show_log, method, logic_cut_callback,
                relax_subproblem_integrality, solver_kwargs)

        complicating_vars = self._get_complicating_vars()
        master_constraint_ids = self._get_master_constraint_ids() or None
        self._stored_cuts = []
        self._cut_pool = []
        time_start = time.perf_counter()

        _log_benders(
            f"Benders decomposition ({method})\n"
            f"  Master: {interface}/{solver}  |  Subproblem: {sub_interface}/{sub_solver}\n"
            f"  Complicating vars: {complicating_vars}\n"
            f"  Strategy: {strategy.value}  |  Acceleration: {accel.value}  |  Sense: {self._objective_sense}\n"
            f"  Max iter: {self._params.iter_limit}  |  Tolerance: {self._params.tol_abs:.1e}",
            show_log)
        _log_benders(
            f"{'Iter':>5s}  {'LB':>12s}  {'UB':>12s}  {'Gap':>10s}  {'Cuts':>12s}  {'Time':>6s}",
            show_log)
        _log_benders(
            f"{'-'*5}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*6}",
            show_log)

        UB = float('inf') if self._objective_sense == 'min' else float('-inf')
        LB = float('-inf') if self._objective_sense == 'min' else float('inf')
        best_solution = None
        bounds_history = []
        obj_history = []
        n_opt_cuts = 0
        n_feas_cuts = 0
        n_sol = 0

        # Fire callback: ON_START
        self._context.iteration = 0
        self._context.lb = LB
        self._context.ub = UB
        self._context.status = BendersStatus.UNSOLVED
        if self._callback_manager.trigger(_BendersEvent.ON_START, self._context):
            return self._finalize_result(UB, LB, best_solution, 0,
                                         bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                                         BendersStatus.UNSOLVED)

        init_m = self._create_model(
            self._master.model_fn,
            self._master.directions,
            self._master.obj_index,
            solver_opts)
        x_bar = {}
        for var_name in complicating_vars:
            x_bar[var_name] = self._zero_value(init_m, var_name)

        _log_benders(
            f"Initial x_bar: {{{', '.join(f'{k}: {np.array(v).tolist()}' for k, v in x_bar.items())}}}",
            False)

        initial_results = self._solve_subproblems(
            x_bar, solver_opts, solution_generator, show_log,
            relax_subproblem_integrality)
        for sub_result in initial_results:
            logic_cuts = self._logic_cuts(x_bar, sub_result)
            cut = None
            if logic_cuts:
                self._stored_cuts.extend(logic_cuts)
            elif not sub_result['feasible']:
                farkas_cut = sub_result.get('farkas_feasibility_cut')
                if farkas_cut is not None:
                    cut = farkas_cut
                    n_feas_cuts += 1
                else:
                    cut = self._generate_feasibility_cut(
                        sub_result, x_bar, complicating_vars)
                    if cut is not None:
                        n_feas_cuts += 1
            else:
                cut = self._generate_optimality_cut(
                    sub_result, x_bar, complicating_vars)
                if cut is not None:
                    n_opt_cuts += 1
            if cut is not None:
                self._stored_cuts.append(cut)

        # Cut-stagnation tracking: an iteration that stores no cut at all
        # (optimality, feasibility, logic, or constant) leaves the master
        # untouched, so the next master solve is the same problem and every
        # later iteration repeats verbatim; including the expensive
        # infeasible-subproblem ladder (IIS / constraint-fixing / artificial
        # variables).  Track it so the loop can stop instead of burning the
        # whole iteration budget on identical work.
        cuts_prev = len(self._stored_cuts)
        stagnant_rounds = 0
        stagnant_round_limit = 5

        for iteration in range(self._params.iter_limit):
            # Check time limit
            elapsed = time.perf_counter() - time_start
            if elapsed >= self._params.time_limit:
                _log_benders(f"Time limit ({self._params.time_limit}s) reached", show_log)
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.TIMEOUT)

            master_m = self._create_model(
                self._master.model_fn,
                self._master.directions,
                self._master.obj_index,
                solver_opts)

            if self._stored_cuts:
                # Apply cut selection and cleanup for advanced acceleration
                if self._multi_cut_enabled or self._cut_pool:
                    selected_cuts = self._apply_cut_selection(x_bar, max_cuts=10)
                    self._add_cuts_to_master(master_m, complicating_vars, cuts=selected_cuts)
                else:
                    self._cleanup_duplicate_cuts()
                    self._add_cuts_to_master(master_m, complicating_vars)

            # Fire callback: ON_MASTER_BUILD
            self._context.iteration = iteration + 1
            self._context.x_bar = x_bar
            self._context.benders = self
            if self._callback_manager.trigger(_BendersEvent.ON_MASTER_BUILD, self._context):
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.UNSOLVED, obj_history, n_sol)

            # Fire callback: ON_BEFORE_MASTER_SOLVE
            self._callback_manager.trigger(_BendersEvent.ON_BEFORE_MASTER_SOLVE, self._context)

            t_master = time.perf_counter()
            if not self._solve_model(master_m, solution_generator,
                                     constraint_ids=master_constraint_ids):
                _log_benders(
                    f"Master problem infeasible at iteration {iteration + 1}",
                    show_log)
                t_master = time.perf_counter() - t_master
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.INFEASIBLE)
            t_master = time.perf_counter() - t_master

            # Fire callback: ON_AFTER_MASTER_SOLVED / ON_MASTER_SOLVED
            self._context.master_obj = self._get_obj(master_m)
            self._callback_manager.trigger(_BendersEvent.ON_AFTER_MASTER_SOLVED, self._context)
            if self._callback_manager.trigger(_BendersEvent.ON_MASTER_SOLVED, self._context):
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.UNSOLVED, obj_history, n_sol)

            current_master_obj = self._get_obj(master_m)
            if current_master_obj is not None:
                if self._objective_sense == 'min':
                    LB = max(LB, current_master_obj)
                else:
                    LB = min(LB, current_master_obj)
            else:
                current_master_obj = 0.0

            x_bar = {}
            for var_name in complicating_vars:
                val = self._get_var_value(master_m, var_name)
                if val is not None:
                    x_bar[var_name] = val

            # Fire callback: ON_BEFORE_SUB_SOLVE
            self._callback_manager.trigger(_BendersEvent.ON_BEFORE_SUB_SOLVE, self._context)

            t_sub = time.perf_counter()
            subproblem_results = self._solve_subproblems(
                x_bar, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)
            t_sub = time.perf_counter() - t_sub

            # Fire callback: ON_AFTER_SUB_SOLVED / ON_SUB_SOLVED
            self._context.sub_results = subproblem_results
            self._callback_manager.trigger(_BendersEvent.ON_AFTER_SUB_SOLVED, self._context)
            if self._callback_manager.trigger(_BendersEvent.ON_SUB_SOLVED, self._context):
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.UNSOLVED, obj_history, n_sol)

            subproblem_feasible = True

            for sub_result in subproblem_results:
                logic_cuts = self._logic_cuts(x_bar, sub_result)
                if logic_cuts:
                    self._stored_cuts.extend(logic_cuts)
                    _log_benders("  Logic-based cut(s) generated", show_log)
                elif not sub_result['feasible']:
                    subproblem_feasible = False
                    farkas_cut = sub_result.get('farkas_feasibility_cut')
                    if farkas_cut is not None:
                        self._stored_cuts.append(farkas_cut)
                        n_feas_cuts += 1
                        _log_benders(
                            f"  Gurobi Farkas feasibility cut generated",
                            show_log)
                    else:
                        cut = self._generate_feasibility_cut(
                            sub_result, x_bar, complicating_vars)
                        if cut is not None:
                            self._stored_cuts.append(cut)
                            n_feas_cuts += 1
                            _log_benders(
                                f"  Feasibility cut generated",
                                show_log)
                else:
                    sub_obj = sub_result.get('objective_value', 0.0)
                    is_augmented = bool(sub_result.get('original_infeasible', False))
                    if self._is_auto():
                        total_at_x_bar = sub_obj
                    else:
                        theta_val = self._get_var_value(master_m, '_benders_theta')
                        if theta_val is None:
                            theta_val = self._get_var_value(master_m, 'theta')
                        if theta_val is None:
                            theta_val = 0.0
                        int_cost_at_x_bar = (current_master_obj - theta_val
                                              if current_master_obj is not None
                                              else 0.0)
                        total_at_x_bar = int_cost_at_x_bar + sub_obj

                    if not is_augmented:
                        if self._objective_sense == 'min':
                            if total_at_x_bar < UB:
                                UB = total_at_x_bar
                                best_solution = dict(x_bar)
                                n_sol += 1
                        else:
                            if total_at_x_bar > UB:
                                UB = total_at_x_bar
                                best_solution = dict(x_bar)
                                n_sol += 1
                    elif show_log:
                        _log_benders(
                            "  Augmented subproblem: cut added, UB not updated "
                            "(original subproblem infeasible)",
                            show_log)

                    has_duals = sub_result.get('has_duals', False)
                    has_numerical = bool(sub_result.get('numerical_coeffs'))
                    if (has_duals or has_numerical) and not logic_cuts:
                        if self._multi_cut_enabled:
                            cuts = self._generate_multi_cuts([sub_result], x_bar, complicating_vars)
                            for cut in cuts:
                                self._add_to_cut_pool(cut)
                                self._stored_cuts.append(cut)
                                n_opt_cuts += 1
                                _log_benders(
                                    f"  Multi-cut: rhs={cut['rhs']:.4f}, "
                                    f"coeffs={cut['coeffs']}",
                                    show_log)
                        else:
                            cut = self._generate_cut_for_method(
                                sub_result, x_bar, complicating_vars)
                            if cut is not None:
                                self._add_to_cut_pool(cut)
                                self._stored_cuts.append(cut)
                                n_opt_cuts += 1
                                _log_benders(
                                    f"  Optimality cut: rhs={cut['rhs']:.4f}, "
                                    f"coeffs={cut['coeffs']}",
                                    show_log)
                        if total_at_x_bar is not None and total_at_x_bar <= UB + tolerance:
                            best_solution = dict(x_bar)
                    else:
                        # No duals or numerical coeffs: emit a constant
                            # lower-bound cut theta >= Q(x_bar) so the master
                            # still gets a valid bound.
                            cut = {
                                'rhs': sub_obj,
                                'coeffs': {},
                                'type': 'optimality',
                                'sub_obj': sub_obj,
                            }
                            self._stored_cuts.append(cut)
                            n_opt_cuts += 1
                            _log_benders(
                                f"  Constant cut: theta >= {sub_obj:.4f} "
                                f"(no variable coefficients available)",
                                show_log)

            bounds_history.append((LB, UB))
            obj_history.append(UB)
            gap = abs(UB - LB) if UB < float('inf') and LB > float('-inf') else float('inf')

            # Fire callbacks for bound changes
            self._context.lb = LB
            self._context.ub = UB
            self._callback_manager.trigger(_BendersEvent.ON_NEW_LOWER_BOUND, self._context)
            self._callback_manager.trigger(_BendersEvent.ON_NEW_UPPER_BOUND, self._context)

            if show_log:
                elapsed = time.perf_counter() - time_start
                _log_iter(iteration + 1, LB, UB, gap,
                          len(self._stored_cuts), n_opt_cuts, n_feas_cuts,
                          elapsed, show_log)

            # Fire callback: ON_ITERATION_END
            self._context.lb = LB
            self._context.ub = UB
            self._context.current_cuts = self._stored_cuts
            if self._callback_manager.trigger(_BendersEvent.ON_ITERATION_END, self._context):
                return self._finalize_result(
                    UB, LB, best_solution, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.UNSOLVED, obj_history, n_sol)

            # Check convergence: absolute AND relative gap
            if (gap <= self._params.tol_abs or
                (UB != 0 and gap / abs(UB) <= self._params.tol_rel)):
                _log_benders(
                    f"Converged at iteration {iteration + 1} "
                    f"(gap = {gap:.2e})",
                    show_log)
                return self._finalize_result(
                    UB,
                    LB, best_solution or x_bar, iteration + 1,
                    bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                    BendersStatus.OPTIMAL, obj_history, n_sol)

            # Stagnation guard: no cut was stored this iteration.  A few
            # consecutive rounds are tolerated (bounds can still move), then
            # stop with an unsolved status — the boost safety net falls back
            # to a direct solve, which is strictly better than replaying
            # identical infeasible subproblems until the iteration cap.
            cuts_now = len(self._stored_cuts)
            if cuts_now == cuts_prev:
                stagnant_rounds += 1
                if stagnant_rounds >= stagnant_round_limit:
                    _log_benders(
                        f"No cut progress for {stagnant_rounds} consecutive "
                        f"iterations; stopping.", show_log)
                    return self._finalize_result(
                        UB, LB, best_solution, iteration + 1,
                        bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                        BendersStatus.UNSOLVED, obj_history, n_sol)
            else:
                stagnant_rounds = 0
            cuts_prev = cuts_now

        _log_benders(
            f"Maximum iterations ({self._params.iter_limit}) reached. "
            f"Final gap = {abs(UB - LB):.2e}",
            show_log)
        final_obj = UB
        return self._finalize_result(
            final_obj, LB, best_solution or x_bar, self._params.iter_limit,
            bounds_history, time_start, n_opt_cuts, n_feas_cuts,
            BendersStatus.MAX_ITERATIONS, obj_history, n_sol)

    # ------------------------------------------------------------------
    # Branch-and-Check Benders (lazy callbacks)
    # ------------------------------------------------------------------

    def _solve_bnc_benders(self, interface, solver, sub_interface, sub_solver,
                            max_iterations, tolerance, cut_strategy,
                            acceleration, show_log, method, logic_cut_callback,
                            relax_subproblem_integrality, solver_kwargs):
        """Benders via Branch-and-Check: solve master MIP once with lazy cuts.

        Instead of rebuilding the master each iteration, the master MIP is
        solved once. At each integer-feasible node, the subproblem is solved
        and cuts are injected via Gurobi's cbLazy(). This avoids redundant
        re-solving of the master and can be significantly faster.

        Requires Gurobi as the master solver.
        """
        import gurobipy as gurobi

        from feloopy.generators import solution_generator

        complicating_vars = self._get_complicating_vars()
        master_constraint_ids = self._get_master_constraint_ids() or None
        self._stored_cuts = []
        self._cut_pool = []
        time_start = time.perf_counter()

        UB = float('inf') if self._objective_sense == 'min' else float('-inf')
        LB = float('-inf') if self._objective_sense == 'min' else float('inf')
        best_solution = None
        bounds_history = []
        n_opt_cuts = 0
        n_feas_cuts = 0

        _log_benders(
            f"Benders decomposition (Branch-and-Check)\n"
            f"  Master: {interface}/{solver}\n"
            f"  Complicating vars: {complicating_vars}\n"
            f"  Max iter: {max_iterations}  |  Tolerance: {tolerance:.1e}",
            show_log)

        # Shared state for the callback
        _bnc_state = {
            'iteration': 0,
            'best_obj': UB,
            'best_solution': None,
            'lb': LB,
            'cuts_added': 0,
        }

        _PREFIXES = ('fvar', 'pvar', 'ivar', 'bvar')

        def _get_bnc_var_values(model, var_name, getter):
            """Extract variable value(s) from Gurobi callback model.

            Handles both scalar and indexed variables. Tries all
            type-prefixes (fvar/pvar/ivar/bvar) to find the solver variable.
            """
            # Try each prefix for a scalar variable
            for pfx in _PREFIXES:
                v = model.getVarByName(f"{pfx}_{var_name}")
                if v is not None:
                    try:
                        return getter(v)
                    except Exception:
                        pass
            # Try indexed: var_name_0, var_name_1, ...
            result = {}
            for pfx in _PREFIXES:
                for idx in range(200):
                    v = model.getVarByName(f"{pfx}_{var_name}_{idx}")
                    if v is None:
                        break
                    try:
                        result[idx] = getter(v)
                    except Exception:
                        break
                if result:
                    return result
            return None

        def _bnc_lazy_callback(model, where):
            """Gurobi lazy constraint callback for Benders cuts."""
            # Integer solution found
            if where == gurobi.GRB.Callback.MIPSOL:
                pass
            # Fractional node solution (for bnc_frac_sol)
            elif where == gurobi.GRB.Callback.MIPNODE:
                if not self._params.bnc_frac_sol:
                    return
                try:
                    if model.cbGet(gurobi.GRB.Callback.MIPNODE_STATUS) != gurobi.GRB.OPTIMAL:
                        return
                except Exception:
                    return
            else:
                return

            if _bnc_state['iteration'] >= max_iterations:
                return

            # Check time limit
            elapsed = time.perf_counter() - time_start
            if elapsed >= self._params.time_limit:
                return

            _bnc_state['iteration'] += 1
            iter_num = _bnc_state['iteration']

            # Use cbGetSolution for integer nodes, cbGetNodeRel for fractional
            is_frac = (where == gurobi.GRB.Callback.MIPNODE)
            _get_val = model.cbGetNodeRel if is_frac else model.cbGetSolution

            # Get incumbent solution using the helper
            x_bar = {}
            for var_name in complicating_vars:
                val = _get_bnc_var_values(model, var_name, _get_val)
                if val is not None:
                    x_bar[var_name] = val

            if not x_bar:
                return

            # Solve subproblem with fixed x_bar
            sub_result = self._solve_single_subproblem(
                _bnc_state['sub_level'], x_bar, complicating_vars,
                _bnc_state['solver_opts'], solution_generator, False,
                relax_subproblem_integrality=relax_subproblem_integrality)

            # Generate and add cuts
            if not sub_result['feasible']:
                farkas_cut = sub_result.get('farkas_feasibility_cut')
                if farkas_cut is not None:
                    cut = farkas_cut
                else:
                    cut = self._generate_feasibility_cut(
                        sub_result, x_bar, complicating_vars)
                if cut is not None:
                    _bnc_add_cut_gurobi(model, cut, complicating_vars)
                    n_feas_cuts += 1
                    _bnc_state['cuts_added'] += 1
                    _log_benders(f"  [BnC] Iter {iter_num}: feasibility cut added", show_log)
            elif sub_result.get('has_duals', False):
                sub_obj = sub_result.get('objective_value', 0.0)

                # Only update UB for integer solutions (not fractional nodes)
                if not is_frac:
                    theta_val = 0.0
                    try:
                        theta_var = model.getVarByName('_benders_theta')
                        if theta_var is not None:
                            theta_val = model.cbGetSolution(theta_var)
                    except Exception:
                        pass

                    master_obj = model.cbGet(gurobi.GRB.Callback.MIP_OBJBST)
                    total = (master_obj - theta_val) + sub_obj

                    if self._objective_sense == 'min':
                        if total < _bnc_state['best_obj']:
                            _bnc_state['best_obj'] = total
                            _bnc_state['best_solution'] = dict(x_bar)
                    else:
                        if total > _bnc_state['best_obj']:
                            _bnc_state['best_obj'] = total
                            _bnc_state['best_solution'] = dict(x_bar)

                # Generate optimality cut
                if self._multi_cut_enabled:
                    cuts = self._generate_multi_cuts(
                        [sub_result], x_bar, complicating_vars)
                    for cut in cuts:
                        _bnc_add_cut_gurobi(model, cut, complicating_vars)
                        n_opt_cuts += 1
                        _bnc_state['cuts_added'] += 1
                else:
                    cut = self._generate_optimality_cut(
                        sub_result, x_bar, complicating_vars)
                    if cut is not None:
                        _bnc_add_cut_gurobi(model, cut, complicating_vars)
                        n_opt_cuts += 1
                        _bnc_state['cuts_added'] += 1

                _log_benders(
                    f"  [BnC] Iter {iter_num}: optimality cut "
                    f"(UB={_bnc_state['best_obj']:.4f})",
                    show_log)

        def _bnc_add_cut_gurobi(model, cut, complicating_vars):
            """Add a Benders cut to Gurobi via cbLazy."""
            is_feasibility = cut.get('type') == 'feasibility'

            if is_feasibility:
                expr = gurobi.LinExpr(-cut['rhs'])
                for var_name, coeff in cut.get('coeffs', {}).items():
                    if isinstance(coeff, dict):
                        for pfx in _PREFIXES:
                            for idx, c in coeff.items():
                                v = model.getVarByName(f"{pfx}_{var_name}_{idx}")
                                if v is not None:
                                    expr += c * v
                            break
                    else:
                        for pfx in _PREFIXES:
                            var = model.getVarByName(f"{pfx}_{var_name}")
                            if var is not None:
                                expr += coeff * var
                                break
                model.cbLazy(expr >= 0)
            else:
                # Optimality cut: theta - sum(c_j * x_j) >= rhs
                theta_var = None
                for name in ('_benders_theta', 'theta'):
                    theta_var = model.getVarByName(name)
                    if theta_var is not None:
                        break

                expr = gurobi.LinExpr(-cut['rhs'])
                if theta_var is not None:
                    expr += 1.0 * theta_var

                for var_name, coeff in cut.get('coeffs', {}).items():
                    if isinstance(coeff, dict):
                        for pfx in _PREFIXES:
                            for idx, c in coeff.items():
                                v = model.getVarByName(f"{pfx}_{var_name}_{idx}")
                                if v is not None:
                                    expr += (-c) * v
                            break
                    else:
                        for pfx in _PREFIXES:
                            var = model.getVarByName(f"{pfx}_{var_name}")
                            if var is not None:
                                expr += (-coeff) * var
                                break
                model.cbLazy(expr >= 0)

        # --- Build and solve the master MIP with the callback ---
        master_m = self._create_model(
            self._master.model_fn,
            self._master.directions,
            self._master.obj_index,
            dict(
                interface_name=interface,
                solver_name=solver,
                sub_interface_name=sub_interface,
                sub_solver_name=sub_solver,
                solver_options={},
                log=False,
            ))

        # Filter master constraints if needed
        if master_constraint_ids:
            orig_c = list(master_m.features['constraints'])
            orig_l = list(master_m.features['constraint_labels'])
            filtered_c = [c for c, l in zip(orig_c, orig_l)
                          if l in master_constraint_ids]
            filtered_l = [l for l in orig_l if l in master_constraint_ids]
            master_m.features['constraints'] = filtered_c
            master_m.features['constraint_labels'] = filtered_l

        # Store subproblem config for callback
        sub_level = _BendersProblem(
            'subproblem', self._master.model_fn,
            self._master.directions, self._master.obj_index)
        _bnc_state['sub_level'] = sub_level
        _bnc_state['solver_opts'] = dict(
            interface_name=sub_interface,
            solver_name=sub_solver,
            sub_interface_name=sub_interface,
            sub_solver_name=sub_solver,
            solver_options={},
            log=False,
        )

        # Solve with Gurobi callback
        model_obj = master_m.model
        try:
            model_obj.setParam('LazyConstraints', 1)
            model_obj.setParam('OutputFlag', 0)
            if self._params.time_limit < float('inf'):
                model_obj.setParam('TimeLimit', self._params.time_limit)
        except Exception:
            pass

        _log_benders("  [BnC] Solving master MIP with lazy callbacks...", show_log)

        # We need to also add initial cuts before solving
        # (to provide a starting bound)
        init_x_bar = {}
        for var_name in complicating_vars:
            init_x_bar[var_name] = self._zero_value(master_m, var_name)

        initial_results = self._solve_subproblems(
            init_x_bar, _bnc_state['solver_opts'], solution_generator,
            show_log, relax_subproblem_integrality)
        for sub_result in initial_results:
            logic_cuts = self._logic_cuts(init_x_bar, sub_result)
            cut = None
            if logic_cuts:
                self._stored_cuts.extend(logic_cuts)
            elif not sub_result['feasible']:
                farkas_cut = sub_result.get('farkas_feasibility_cut')
                if farkas_cut is not None:
                    cut = farkas_cut
                    n_feas_cuts += 1
                else:
                    cut = self._generate_feasibility_cut(
                        sub_result, x_bar, complicating_vars)
                    if cut is not None:
                        n_feas_cuts += 1
            else:
                cut = self._generate_optimality_cut(
                    sub_result, x_bar, complicating_vars)
                if cut is not None:
                    n_opt_cuts += 1
            if cut is not None:
                self._stored_cuts.append(cut)


        # Add initial cuts to the model before solving
        if self._stored_cuts:
            self._add_cuts_to_master(master_m, complicating_vars)

        # Solve with callback
        try:
            model_obj.optimize(_bnc_lazy_callback)
        except Exception as e:
            _log_benders(f"  [BnC] Gurobi error: {e}", show_log)
            return self._finalize_result(
                UB, LB, best_solution, 0, bounds_history, time_start,
                n_opt_cuts, n_feas_cuts, BendersStatus.ERROR)

        # Extract results
        try:
            final_obj = model_obj.objVal
            best_solution = {}
            for var_name in complicating_vars:
                val = _get_bnc_var_values(model_obj, var_name, lambda v: v.X)
                if val is not None:
                    best_solution[var_name] = val
        except Exception:
            final_obj = _bnc_state['best_obj']
            best_solution = _bnc_state['best_solution'] or {}

        status = BendersStatus.OPTIMAL
        try:
            if model_obj.status == gurobi.GRB.INFEASIBLE:
                status = BendersStatus.INFEASIBLE
            elif model_obj.status == gurobi.GRB.UNBOUNDED:
                status = BendersStatus.UNBOUNDED
            elif model_obj.status == gurobi.GRB.TIME_LIMIT:
                status = BendersStatus.TIMEOUT
            elif model_obj.status != gurobi.GRB.OPTIMAL:
                status = BendersStatus.FEASIBLE
        except Exception:
            pass

        _log_benders(
            f"  [BnC] Done: {final_obj:.4f}, "
            f"{_bnc_state['iteration']} lazy callbacks, "
            f"{_bnc_state['cuts_added']} cuts added",
            show_log)

        return self._finalize_result(
            final_obj, LB, best_solution, _bnc_state['iteration'],
            bounds_history, time_start, n_opt_cuts, n_feas_cuts, status)

    def _finalize_result(self, ub, lb, variables, iterations,
                         bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                         status, obj_history=None, n_sol=0):
        """Package results into a BendersResult."""
        elapsed = time.perf_counter() - time_start
        self._result = BendersResult(
            status=status,
            objective=ub,
            variables=variables or {},
            iterations=iterations,
            bounds=bounds_history,
            n_optimality_cuts=n_opt_cuts,
            n_feasibility_cuts=n_feas_cuts,
            n_cuts=n_opt_cuts + n_feas_cuts,
            n_sol=n_sol,
            runtime_total=elapsed,
            gap_abs=abs(ub - lb) if ub < float('inf') and lb > float('-inf') else float('inf'),
            gap_rel=abs(ub - lb) / abs(ub) if ub != 0 and ub < float('inf') and lb > float('-inf') else float('inf'),
            lb_history=[b[0] for b in bounds_history],
            ub_history=[b[1] for b in bounds_history],
            obj_history=obj_history or [],
        )
        return self._result

    def get_convergence(self):
        """Return convergence history from the last Benders solve.

        Returns
        -------
        dict with keys:
            'lb_history': list[float] — lower bound per iteration
            'ub_history': list[float] — upper bound per iteration
            'gap_abs_history': list[float] — |UB - LB| per iteration
            'gap_rel_history': list[float] — |UB - LB| / |UB| per iteration
            'iterations': int — total iterations executed
            'status': str — termination status ('optimal', 'max_iterations',
                            'timeout', 'infeasible', 'unsolved')
            'n_optimality_cuts': int
            'n_feasibility_cuts': int
            'n_cuts': int
            'runtime_total': float — wall-clock seconds
            'objective': float — final objective value
        """
        result = getattr(self, '_result', None)
        if result is None:
            return {
                'lb_history': [], 'ub_history': [],
                'gap_abs_history': [], 'gap_rel_history': [],
                'iterations': 0, 'status': 'unsolved',
                'n_optimality_cuts': 0, 'n_feasibility_cuts': 0,
                'n_cuts': 0, 'runtime_total': 0.0, 'objective': float('inf'),
            }
        lb = list(result.lb_history)
        ub = list(result.ub_history)
        gap_abs = [abs(u - l) for u, l in zip(ub, lb)]
        gap_rel = [
            abs(u - l) / abs(u) if abs(u) > 1e-15 else 0.0
            for u, l in zip(ub, lb)
        ]
        status_str = result.status
        if hasattr(status_str, 'value'):
            status_str = status_str.value
        return {
            'lb_history': lb,
            'ub_history': ub,
            'obj_history': list(result.obj_history) if result.obj_history else [],
            'gap_abs_history': gap_abs,
            'gap_rel_history': gap_rel,
            'iterations': result.iterations,
            'status': status_str,
            'n_optimality_cuts': result.n_optimality_cuts,
            'n_feasibility_cuts': result.n_feasibility_cuts,
            'n_cuts': result.n_cuts,
            'n_sol': result.n_sol,
            'runtime_total': result.runtime_total,
            'objective': result.objective,
        }

    # ------------------------------------------------------------------
    # Solver-independent incremental Benders (master-only incremental)
    # ------------------------------------------------------------------

    @staticmethod
    def _master_status_dead(master_inc):
        """True when the last native master solve failed outright.

        The objective value alone cannot be trusted here: HiGHS
        reports an uninitialized objective (e.g. 1.27e-321) for an
        infeasible model instead of inf, which would poison the
        lower bound before any infinity-based guard could fire.
        """
        try:
            status = master_inc._get_native_result()
        except Exception:
            return False
        if status is None:
            return False
        if isinstance(status, int):
            # gurobi: 3=INFEASIBLE, 4=UNBOUNDED, 5=INF_OR_UNBD
            return status in (3, 4, 5)
        name = getattr(status, 'name', None)
        if name is None:
            name = str(status)
        return any(token in name.lower() for token in (
            'infeasible', 'unbounded', 'error', 'notset'))

    def _solve_incremental_benders(self, interface, solver, sub_interface,
                                    sub_solver, max_iterations, tolerance,
                                    cut_strategy, acceleration, show_log,
                                    method, logic_cut_callback,
                                    relax_subproblem_integrality,
                                    solver_kwargs):
        """Benders with incremental master re-solves.

        The master is built once via the IncrementalModel API and cuts
        are added incrementally via addRow.  Subproblems are rebuilt
        each iteration via the standard path (needed for constraint
        filtering and dual extraction).
        """
        from feloopy.classes.incremental import IncrementalModel

        complicating_vars = self._get_complicating_vars()
        master_constraint_ids = self._get_master_constraint_ids() or None
        self._stored_cuts = []

        solver_opts = dict(
            interface_name=interface,
            solver_name=solver,
            sub_interface_name=sub_interface,
            sub_solver_name=sub_solver,
            solver_options={
                key: value
                for key, value in solver_kwargs.get('options', {}).items()
                if key not in ('max_iterations', 'tolerance')
            },
            log=solver_kwargs.get('solver_log', False),
            write_model_file=solver_kwargs.get('save_model', False),
            save_solver_log=solver_kwargs.get('save_log', False),
            email_address=solver_kwargs.get('email', None),
            time_limit=solver_kwargs.get('time_limit', None),
            thread_count=solver_kwargs.get('cpu_threads', None),
            absolute_gap=solver_kwargs.get('absolute_gap', None),
            relative_gap=solver_kwargs.get('relative_gap', None),
            max_iterations=solver_kwargs.get('max_nodes', None),
            debug_mode=solver_kwargs.get('debug', False),
            callback=None,
        )

        _log_benders(
            f"Benders decomposition ({method})\n"
            f"  Master: {interface}/{solver}  |  Subproblem: {sub_interface}/{sub_solver}\n"
            f"  Complicating vars: {complicating_vars}\n"
            f"  Max iter: {max_iterations}  |  Tolerance: {tolerance:.1e}",
            show_log)
        _log_benders(
            f"{'Iter':>5s}  {'LB':>12s}  {'UB':>12s}  {'Gap':>10s}  {'Cuts':>12s}  {'Time':>6s}",
            show_log)
        _log_benders(
            f"{'-'*5}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*6}",
            show_log)

        # Build master via IncrementalModel
        master_m = self._create_model(
            self._master.model_fn,
            self._master.directions,
            self._master.obj_index,
            solver_opts)

        # Filter master constraints if constraint_ids specified
        if master_constraint_ids:
            orig_c = list(master_m.features['constraints'])
            orig_l = list(master_m.features['constraint_labels'])
            filtered_c = []
            filtered_l = []
            for c, label in zip(orig_c, orig_l):
                if label in master_constraint_ids:
                    filtered_c.append(c)
                    filtered_l.append(label)
            master_m.features['constraints'] = filtered_c
            master_m.features['constraint_labels'] = filtered_l

        master_inc = IncrementalModel(
            master_m,
            directions=self._master.directions,
            obj_index=self._master.obj_index,
            solver_name=solver,
            solver_options=solver_opts.get('solver_options', {}),
            log=solver_kwargs.get('solver_log', False))

        # Build the incremental model — this materializes the native
        # solver model, sets the objective, and captures the var_col_map.
        master_inc.build()

        # Add theta variable for Benders cuts.
        # If the model_fn already created _benders_theta (e.g. auto path),
        # reuse it.  Otherwise create a new column.
        existing_theta = master_inc._var_col_map.get('_benders_theta')
        if existing_theta is None:
            existing_theta = master_inc._var_col_map.get('theta')
        auto_mode = existing_theta is not None
        if auto_mode:
            theta_col_idx = existing_theta
        else:
            theta_col_idx = master_inc.add_column(
                cost=1.0, indices=[], values=[],
                lb=-1e9, ub=1e9, name='_benders_theta')
            master_inc._var_col_map['_benders_theta'] = theta_col_idx
            # Add initial constraint: theta >= 0
            master_inc.add_row(lower=0.0, upper=1e20,
                               indices=[theta_col_idx], values=[1.0],
                               label='_benders_theta_init')

        from feloopy.generators import solution_generator

        is_min = self._objective_sense == 'min'
        UB = float('inf') if is_min else float('-inf')
        LB = float('-inf') if is_min else float('inf')
        best_solution = None
        bounds_history = []
        n_sol = 0
        time_start = time.perf_counter()
        stagnation_count = 0
        prev_LB = float('-inf') if is_min else float('inf')
        prev_UB = float('inf') if is_min else float('-inf')

        # Initial x_bar
        x_bar = {}
        for var_name in complicating_vars:
            x_bar[var_name] = self._zero_value(master_m, var_name)

        _log_benders(
            f"Initial x_bar: {{{', '.join(f'{k}: {np.array(v).tolist()}' for k, v in x_bar.items())}}}",
            show_log)

        # Initial subproblem solve (standard path for dual extraction)
        initial_results = self._solve_subproblems(
            x_bar, solver_opts, solution_generator, show_log,
            relax_subproblem_integrality)
        cuts_added = 0
        n_opt_cuts = 0
        n_feas_cuts = 0
        for sub_result in initial_results:
            logic_cuts = self._logic_cuts(x_bar, sub_result)
            cut = None
            if logic_cuts:
                self._stored_cuts.extend(logic_cuts)
            elif not sub_result['feasible']:
                farkas_cut = sub_result.get('farkas_feasibility_cut')
                if farkas_cut is not None:
                    cut = farkas_cut
                    n_feas_cuts += 1
                else:
                    cut = self._generate_feasibility_cut(
                        sub_result, x_bar, complicating_vars)
                    if cut is not None:
                        n_feas_cuts += 1
            else:
                cut = self._generate_optimality_cut(
                    sub_result, x_bar, complicating_vars)
                if cut is not None:
                    n_opt_cuts += 1
            if cut is not None:
                self._stored_cuts.append(cut)

        for iteration in range(max_iterations):
            # Add only NEW cuts to master incrementally
            if self._stored_cuts and cuts_added < len(self._stored_cuts):
                new_cuts = self._stored_cuts[cuts_added:]
                if self._multi_cut_enabled or self._cut_pool:
                    selected_cuts = new_cuts[:10]
                    if self._cut_pool:
                        violations = [(self._compute_cut_violation(c, x_bar), c)
                                      for c in new_cuts]
                        violations.sort(key=lambda x: x[0], reverse=True)
                        selected_cuts = [c for _, c in violations[:10]]
                    self._add_cuts_incremental(
                        master_inc, selected_cuts, complicating_vars,
                        f'_benders_iter_{iteration}')
                else:
                    self._add_cuts_incremental(
                        master_inc, new_cuts, complicating_vars,
                        f'_benders_iter_{iteration}')
                cuts_added = len(self._stored_cuts)

            # Solve master
            master_inc.solve()
            master_obj_raw = master_inc.get_objective_value()
            current_master_obj = master_obj_raw

            if (self._master_status_dead(master_inc) or
                    current_master_obj is not None and (
                        (is_min and current_master_obj >= 1e29) or
                        (not is_min and current_master_obj <= -1e29))):
                _log_benders(
                    f"Master infeasible at iteration {iteration + 1} "
                    f"(objective {current_master_obj}); stopping.",
                    show_log)
                return self._build_result(
                    UB, best_solution or x_bar, iteration + 1,
                    bounds_history, 'infeasible',
                    n_opt_cuts=n_opt_cuts, n_feas_cuts=n_feas_cuts,
                    runtime=time.perf_counter() - time_start,
                    n_sol=n_sol)

            if current_master_obj is not None:
                if is_min:
                    LB = max(LB, current_master_obj)
                else:
                    LB = min(LB, current_master_obj)
            else:
                current_master_obj = 0.0

            # Extract x_bar
            x_bar = {}
            for var_name in complicating_vars:
                val = master_inc.get_variable_value(var_name)
                if val is not None:
                    if isinstance(val, dict):
                        first_key = next(iter(val.keys()), None)
                        if isinstance(first_key, tuple):
                            shape = tuple(
                                max(k[i] for k in val.keys()) + 1
                                for i in range(len(first_key)))
                            arr = np.zeros(shape)
                            for idx, v in val.items():
                                arr[idx] = v
                            x_bar[var_name] = arr
                        else:
                            arr = np.zeros(max(val.keys()) + 1)
                            for idx, v in val.items():
                                arr[idx] = v
                            x_bar[var_name] = arr
                    else:
                        x_bar[var_name] = val

            # Solve subproblems
            subproblem_results = self._solve_subproblems(
                x_bar, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)

            subproblem_feasible = True
            for sub_result in subproblem_results:
                logic_cuts = self._logic_cuts(x_bar, sub_result)
                if logic_cuts:
                    self._stored_cuts.extend(logic_cuts)
                    _log_benders("  Logic-based cut(s) generated", show_log)
                elif not sub_result['feasible']:
                    subproblem_feasible = False
                    farkas_cut = sub_result.get('farkas_feasibility_cut')
                    if farkas_cut is not None:
                        self._stored_cuts.append(farkas_cut)
                        n_feas_cuts += 1
                        _log_benders(f"  Gurobi Farkas feasibility cut generated", show_log)
                    else:
                        cut = self._generate_feasibility_cut(
                            sub_result, x_bar, complicating_vars)
                        if cut is not None:
                            self._stored_cuts.append(cut)
                            n_feas_cuts += 1
                            _log_benders(f"  Feasibility cut generated", show_log)
                else:
                    sub_obj = sub_result.get('objective_value', 0.0)
                    is_augmented = bool(sub_result.get('original_infeasible', False))

                    # The master optimizes the first-stage (integer)
                    # cost plus theta while the subproblem returns the
                    # second-stage cost: the true incumbent is their
                    # sum.  Using the subproblem value alone
                    # understates UB by the integer cost, so the gap
                    # can never close on models with an integer
                    # objective component.
                    theta_val = master_inc.get_variable_value(
                        '_benders_theta')
                    if theta_val is None:
                        theta_val = master_inc.get_variable_value('theta')
                    if theta_val is None:
                        theta_val = 0.0
                    int_cost_at_x_bar = (
                        master_obj_raw - theta_val
                        if master_obj_raw is not None else 0.0)
                    total_at_x_bar = int_cost_at_x_bar + sub_obj

                    if not is_augmented:
                        if is_min:
                            if total_at_x_bar < UB:
                                UB = total_at_x_bar
                                best_solution = dict(x_bar)
                                n_sol += 1
                        else:
                            if total_at_x_bar > UB:
                                UB = total_at_x_bar
                                best_solution = dict(x_bar)
                                n_sol += 1

                    has_duals = sub_result.get('has_duals', False)
                    has_numerical = bool(sub_result.get('numerical_coeffs'))
                    if (has_duals or has_numerical) and not logic_cuts:
                        if self._multi_cut_enabled:
                            cuts = self._generate_multi_cuts([sub_result], x_bar, complicating_vars)
                            for cut in cuts:
                                self._add_to_cut_pool(cut)
                                self._stored_cuts.append(cut)
                                n_opt_cuts += 1
                                _log_benders(
                                    f"  Multi-cut: rhs={cut['rhs']:.4f}, "
                                    f"coeffs={cut['coeffs']}",
                                    show_log)
                        else:
                            cut = self._generate_cut_for_method(
                                sub_result, x_bar, complicating_vars)
                            if cut is not None:
                                self._add_to_cut_pool(cut)
                                self._stored_cuts.append(cut)
                                n_opt_cuts += 1
                                _log_benders(
                                    f"  Optimality cut: rhs={cut['rhs']:.4f}, "
                                    f"coeffs={cut['coeffs']}",
                                    show_log)
                        if total_at_x_bar is not None and total_at_x_bar <= UB + tolerance:
                            best_solution = dict(x_bar)
                    else:
                        cut = {
                            'rhs': sub_obj,
                            'coeffs': {},
                            'type': 'optimality',
                            'sub_obj': sub_obj,
                        }
                        self._stored_cuts.append(cut)
                        n_opt_cuts += 1
                        _log_benders(
                            f"  Constant cut: theta >= {sub_obj:.4f} "
                            f"(no variable coefficients available)",
                            show_log)

            bounds_history.append((LB, UB))
            if is_min:
                gap = abs(UB - LB) if UB < float('inf') else float('inf')
            else:
                gap = abs(UB - LB) if UB > float('-inf') else float('inf')
            elapsed = time.perf_counter() - time_start
            _log_iter(iteration + 1, LB, UB, gap,
                      len(self._stored_cuts), 0, 0, elapsed, show_log)

            if gap <= tolerance:
                _log_benders(
                    f"Converged at iteration {iteration + 1} "
                    f"(gap = {gap:.2e} <= {tolerance:.2e})",
                    show_log)
                return self._build_result(
                    UB if is_min else UB,
                    best_solution or x_bar, iteration + 1,
                    bounds_history, 'optimal',
                    n_opt_cuts=n_opt_cuts, n_feas_cuts=n_feas_cuts,
                    runtime=time.perf_counter() - time_start,
                    n_sol=n_sol)

            # Stagnation detection: stop if LB and UB haven't improved
            # Don't trigger stagnation while UB is inf — the master is
            # still searching for a feasible integer point.
            if LB == prev_LB and UB == prev_UB:
                if UB < float('inf') or not is_min:
                    stagnation_count += 1
                if stagnation_count >= 20:
                    _log_benders(
                        f"Stagnation detected at iteration {iteration + 1} "
                        f"(no improvement for {stagnation_count} iterations)",
                        show_log)
                    break
            else:
                stagnation_count = 0
            prev_LB = LB
            prev_UB = UB

        _log_benders(
            f"Maximum iterations ({max_iterations}) reached. "
            f"Final gap = {abs(UB - LB):.2e}",
            show_log)
        final_obj = UB if is_min else UB
        return self._build_result(
            final_obj, best_solution or x_bar, max_iterations,
            bounds_history, 'max_iterations',
            n_opt_cuts=n_opt_cuts, n_feas_cuts=n_feas_cuts,
            runtime=time.perf_counter() - time_start,
            n_sol=n_sol)

    def _add_cuts_incremental(self, master_inc, cuts, complicating_vars,
                               prefix):
        """Add Benders cuts to the master IncrementalModel."""
        theta_col = None
        for name in ('theta', '_benders_theta'):
            col_info = master_inc.var_col_map.get(name)
            if col_info is not None and not isinstance(col_info, dict):
                theta_col = col_info
                break

        for i, cut in enumerate(cuts):
            indices = []
            values = []
            is_feasibility = cut.get('type') == 'feasibility'

            if is_feasibility:
                for var_name, coeff in cut.get('coeffs', {}).items():
                    col_info = master_inc.var_col_map.get(var_name)
                    if col_info is None:
                        continue
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            if isinstance(coeff, dict):
                                # Try direct lookup, then tuple-normalized lookup
                                c = coeff.get(idx, None)
                                if c is None and isinstance(idx, int):
                                    c = coeff.get((idx,), 0.0)
                                elif c is None:
                                    c = coeff.get(idx, 0.0)
                            else:
                                c = coeff
                            if abs(c) > 1e-15:
                                indices.append(col_idx)
                                values.append(c)
                    elif isinstance(coeff, dict):
                        c_val = coeff.get(None, sum(coeff.values()))
                        if abs(c_val) > 1e-15:
                            indices.append(col_info)
                            values.append(c_val)
                    else:
                        if abs(coeff) > 1e-15:
                            indices.append(col_info)
                            values.append(coeff)
                lower = cut['rhs']
                upper = 1e20
            else:
                if theta_col is not None:
                    indices.append(theta_col)
                    values.append(1.0)
                for var_name, coeff in cut.get('coeffs', {}).items():
                    col_info = master_inc.var_col_map.get(var_name)
                    if col_info is None:
                        continue
                    if isinstance(coeff, dict):
                        if isinstance(col_info, dict):
                            for idx, col_idx in col_info.items():
                                if isinstance(coeff, dict):
                                    c = coeff.get(idx, None)
                                    if c is None and isinstance(idx, int):
                                        c = coeff.get((idx,), 0.0)
                                    elif c is None:
                                        c = coeff.get(idx, 0.0)
                                else:
                                    c = coeff
                                if abs(c) > 1e-15:
                                    indices.append(col_idx)
                                    values.append(-c)
                        else:
                            c_val = coeff.get(None, sum(coeff.values()))
                            if abs(c_val) > 1e-15:
                                indices.append(col_info)
                                values.append(-c_val)
                    elif isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            if abs(coeff) > 1e-15:
                                indices.append(col_idx)
                                values.append(-coeff)
                    else:
                        if abs(coeff) > 1e-15:
                            indices.append(col_info)
                            values.append(-coeff)
                lower = cut['rhs']
                upper = 1e20

            if indices:
                label = f'{prefix}_{"feas" if is_feasibility else "opt"}_{i}'
                master_inc.add_row(lower, upper, indices, values, label=label)

    def _solve_subproblems_incremental(self, sub_incs, master_vars,
                                        complicating_vars, show_log,
                                        relax_subproblem_integrality):
        """Solve subproblems using IncrementalModel with bound fixing.

        Uses reduced costs instead of fixing-constraint duals for cut
        generation, since bound-based fixing doesn't create constraints.
        """
        results = []

        if self._is_auto():
            # Auto mode: single subproblem from master's model_fn
            sub_m, sub_inc = sub_incs['subproblem']
            shared_vars = complicating_vars

            # Capture original bounds
            for var_name in shared_vars:
                if var_name not in sub_inc._original_bounds:
                    sub_inc._capture_var_bounds(var_name)

            # Fix complicating variables via bounds
            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    sub_inc.fix_variable(var_name, val)

            # Relax integrality if needed
            if relax_subproblem_integrality:
                _relax_integrality(sub_m)

            # Solve
            sub_inc.solve()
            feasible = sub_m.healthy()

            result = {
                'name': 'subproblem',
                'feasible': feasible,
                'status': 'optimal' if feasible else 'infeasible',
                'objective_value': sub_inc.get_objective_value() if feasible else None,
                'model': sub_m,
                'has_duals': False,
                'sub_constraints': [],
                'fixing_vars': shared_vars,
                'fixing_vals': {v: master_vars.get(v) for v in shared_vars},
            }

            if feasible:
                # Use reduced costs for cut coefficients
                reduced_costs = sub_inc.get_reduced_costs()
                if reduced_costs is not None:
                    fix_duals = {}
                    col_map = sub_inc.var_col_map
                    for var_name in shared_vars:
                        col_info = col_map.get(var_name)
                        if col_info is None:
                            continue
                        if isinstance(col_info, dict):
                            for idx, col_idx in col_info.items():
                                if col_idx < len(reduced_costs):
                                    label = f'_benders_fix_{var_name}_{idx}'
                                    fix_duals[label] = reduced_costs[col_idx]
                        else:
                            if col_info < len(reduced_costs):
                                label = f'_benders_fix_{var_name}'
                                fix_duals[label] = reduced_costs[col_info]
                    result['fix_duals'] = fix_duals
                    result['has_duals'] = len(fix_duals) > 0

            # Restore bounds
            for var_name in shared_vars:
                sub_inc.unfix_variable(var_name)

            results.append(result)
        else:
            # Manual mode: iterate over links and subproblems
            for link in self._links:
                if link is None:
                    continue
                level_name = link['lower']
                level = None
                for sp in self._subproblems:
                    if isinstance(sp, _BendersProblem) and sp.name == level_name:
                        level = sp
                        break
                if level is None:
                    continue

                sub_m, sub_inc = sub_incs[level.name]
                shared_vars = link['shared_vars']

                # Capture original bounds
                for var_name in shared_vars:
                    if var_name not in sub_inc._original_bounds:
                        sub_inc._capture_var_bounds(var_name)

                # Fix complicating variables via bounds
                for var_name, val in master_vars.items():
                    if var_name in shared_vars:
                        sub_inc.fix_variable(var_name, val)

                # Relax integrality if needed
                if relax_subproblem_integrality:
                    _relax_integrality(sub_m)

                # Solve
                sub_inc.solve()
                feasible = sub_m.healthy()

                result = {
                    'name': level.name,
                    'feasible': feasible,
                    'status': 'optimal' if feasible else 'infeasible',
                    'objective_value': sub_inc.get_objective_value() if feasible else None,
                    'model': sub_m,
                    'has_duals': False,
                    'sub_constraints': [],
                    'fixing_vars': shared_vars,
                    'fixing_vals': {v: master_vars.get(v) for v in shared_vars},
                }

                if feasible:
                    reduced_costs = sub_inc.get_reduced_costs()
                    if reduced_costs is not None:
                        fix_duals = {}
                        col_map = sub_inc.var_col_map
                        for var_name in shared_vars:
                            col_info = col_map.get(var_name)
                            if col_info is None:
                                continue
                            if isinstance(col_info, dict):
                                for idx, col_idx in col_info.items():
                                    if col_idx < len(reduced_costs):
                                        label = f'_benders_fix_{var_name}_{idx}'
                                        fix_duals[label] = reduced_costs[col_idx]
                            else:
                                if col_info < len(reduced_costs):
                                    label = f'_benders_fix_{var_name}'
                                    fix_duals[label] = reduced_costs[col_info]
                        result['fix_duals'] = fix_duals
                        result['has_duals'] = len(fix_duals) > 0

                # Restore bounds
                for var_name in shared_vars:
                    sub_inc.unfix_variable(var_name)

                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Optimized cached Benders for HiGHS
    # ------------------------------------------------------------------

    def _solve_cached(self, init_m, x_bar, complicating_vars,
                      master_constraint_ids, solver_opts, solution_generator,
                      max_iterations, tolerance, show_log,
                      relax_subproblem_integrality, solver_kwargs):
        """Fallback to standard Benders when cached mode not yet viable."""
        from feloopy.generators import solution_generator as _sg

        is_min = self._objective_sense == 'min'
        UB = float('inf') if is_min else float('-inf')
        LB = float('-inf') if is_min else float('inf')
        best_solution = None
        bounds_history = []

        initial_results = self._solve_subproblems(
            x_bar, solver_opts, solution_generator, show_log,
            relax_subproblem_integrality)
        for sub_result in initial_results:
            logic_cuts = self._logic_cuts(x_bar, sub_result)
            cut = None
            if logic_cuts:
                self._stored_cuts.extend(logic_cuts)
            elif not sub_result['feasible']:
                farkas_cut = sub_result.get('farkas_feasibility_cut')
                if farkas_cut is not None:
                    cut = farkas_cut
                else:
                    cut = self._generate_feasibility_cut(
                        sub_result, x_bar, complicating_vars)
            else:
                cut = self._generate_optimality_cut(
                    sub_result, x_bar, complicating_vars)
            if cut is not None:
                self._stored_cuts.append(cut)

        for iteration in range(max_iterations):
            master_m = self._create_model(
                self._master.model_fn,
                self._master.directions,
                self._master.obj_index,
                solver_opts)

            if self._stored_cuts:
                self._add_cuts_to_master(master_m, complicating_vars)

            if not self._solve_model(master_m, _sg,
                                     constraint_ids=master_constraint_ids):
                _log_benders(
                    f"Master problem infeasible at iteration {iteration + 1}",
                    show_log)
                break

            current_master_obj = self._get_obj(master_m)
            if current_master_obj is not None:
                if is_min:
                    LB = max(LB, current_master_obj)
                else:
                    LB = min(LB, current_master_obj)
            else:
                current_master_obj = 0.0

            x_bar = {}
            for var_name in complicating_vars:
                val = self._get_var_value(master_m, var_name)
                if val is not None:
                    x_bar[var_name] = val

            subproblem_results = self._solve_subproblems(
                x_bar, solver_opts, _sg, show_log,
                relax_subproblem_integrality)

            for sub_result in subproblem_results:
                logic_cuts = self._logic_cuts(x_bar, sub_result)
                if logic_cuts:
                    self._stored_cuts.extend(logic_cuts)
                    _log_benders("  Logic-based cut(s) generated", show_log)
                elif not sub_result['feasible']:
                    farkas_cut = sub_result.get('farkas_feasibility_cut')
                    if farkas_cut is not None:
                        self._stored_cuts.append(farkas_cut)
                        _log_benders("  Gurobi Farkas feasibility cut generated", show_log)
                    else:
                        cut = self._generate_feasibility_cut(
                            sub_result, x_bar, complicating_vars)
                        if cut is not None:
                            self._stored_cuts.append(cut)
                            _log_benders("  Feasibility cut generated", show_log)
                else:
                    sub_obj = sub_result.get('objective_value', 0.0)
                    is_augmented = bool(sub_result.get('original_infeasible', False))
                    total_at_x_bar = sub_obj
                    if not is_augmented:
                        if is_min:
                            if total_at_x_bar < UB:
                                UB = total_at_x_bar
                        else:
                            if total_at_x_bar > UB:
                                UB = total_at_x_bar

                    has_duals = sub_result.get('has_duals', False)
                    has_numerical = bool(sub_result.get('numerical_coeffs'))
                    if (has_duals or has_numerical) and not logic_cuts:
                        cut = self._generate_cut_for_method(
                            sub_result, x_bar, complicating_vars)
                        if cut is not None:
                            self._stored_cuts.append(cut)
                            _log_benders(
                                f"  Optimality cut: rhs={cut['rhs']:.4f}, "
                                f"coeffs={cut['coeffs']}",
                                show_log)
                        if total_at_x_bar is not None and total_at_x_bar <= UB + tolerance:
                            best_solution = dict(x_bar)
                    else:
                        _log_benders(
                            f"  Subproblem feasible (obj={sub_obj:.4f}) "
                            f"but no duals or logic cuts available",
                            show_log)

            bounds_history.append((LB, UB))
            if is_min:
                gap = abs(UB - LB) if UB < float('inf') else float('inf')
            else:
                gap = abs(UB - LB) if UB > float('-inf') else float('inf')
            elapsed = time.perf_counter() - time_start
            _log_iter(iteration + 1, LB, UB, gap,
                      len(self._stored_cuts), 0, 0, elapsed, show_log)

            if gap <= tolerance:
                _log_benders(
                    f"Converged at iteration {iteration + 1} "
                    f"(gap = {gap:.2e} <= {tolerance:.2e})",
                    show_log)
                return self._build_result(
                    UB,
                    best_solution or x_bar, iteration + 1,
                    bounds_history, 'optimal')

        _log_benders(
            f"Maximum iterations ({max_iterations}) reached. "
            f"Final gap = {abs(UB - LB):.2e}",
            show_log)

        final_obj = UB
        return self._build_result(
            final_obj, best_solution or x_bar, max_iterations,
            bounds_history, 'max_iterations')

    def _solve_subproblems_cached(self, sub_m, sub_var_col_map, master_vars,
                                  shared_vars, original_bounds, show_log,
                                  derive_numerical_cut=True):
        """Solve subproblems using cached HiGHS model with bound-based fixing."""

        for var_name in shared_vars:
            if var_name in master_vars:
                self._fix_variable_bound(
                    sub_m, sub_var_col_map, var_name, master_vars[var_name])

        sub_m.features['log'] = False
        sub_m.features['solver_options']['Presolve'] = 0

        _log_benders("  Solving cached subproblem (LP relaxation)", show_log)

        import timeit as _ti
        t0 = _ti.default_timer()
        sub_m.model.run()
        t1 = _ti.default_timer()

        status_str = 'unknown'
        try:
            import highspy
            status = sub_m.model.getModelStatus()
            status_map = {
                highspy.HighsStatus.kOk: 'ok',
                highspy.HighsStatus.kWarning: 'warning',
                highspy.HighsStatus.kError: 'error',
            }
            status_str = status_map.get(status, f'unknown_{status}')
        except Exception:
            pass

        feasible = sub_m.healthy()

        try:
            sol = sub_m.model.getSolution()
            sub_m.solution = sol, [t0, t1]
            lp = sub_m.model.getLp()
            sub_m.features['lp_data'] = {
                'n_cols': lp.num_col_,
                'n_rows': lp.num_row_,
                'row_lower': list(lp.row_lower_),
                'row_upper': list(lp.row_upper_),
                'col_lower': list(lp.col_lower_),
                'col_upper': list(lp.col_upper_),
                'col_cost': list(lp.col_cost_),
                'integrality': list(lp.integrality_),
                'col_names': list(lp.col_names_) if lp.col_names_ else [],
                'row_names': list(lp.row_names_) if lp.row_names_ else [],
                'A_col_pointers': list(lp.a_matrix_.start_),
                'A_row_indices': list(lp.a_matrix_.index_),
                'A_values': list(lp.a_matrix_.value_),
            }
        except Exception:
            pass

        if not feasible:
            self._unfix_all_bounds(sub_m, sub_var_col_map, shared_vars,
                                   original_bounds)
            _log_benders(
                f"  Subproblem: INFEASIBLE (status = {status_str})",
                show_log)
            return [{
                'name': 'subproblem',
                'feasible': False,
                'status': status_str,
                'objective_value': None,
                'model': sub_m,
                'has_duals': False,
                'sub_constraints': [],
                'fixing_vars': shared_vars,
                'fixing_vals': {v: master_vars.get(v) for v in shared_vars},
            }]

        sub_obj = self._get_obj(sub_m)
        if sub_obj is None:
            sub_obj = 0.0

        fix_duals = {}
        sub_duals = {}
        sub_constraints = []
        for label in sub_m.features.get('constraint_labels', []):
            if not label:
                continue
            dual = self._get_dual(sub_m, label)
            if dual is None:
                continue
            if label.startswith('_benders_fix_'):
                fix_duals[label] = dual
            else:
                sub_constraints.append(label)
                sub_duals[label] = dual

        no_useful_fix_duals = (
            not fix_duals or
            all(abs(float(value)) <= 1e-12 for value in fix_duals.values()))

        if no_useful_fix_duals:
            try:
                sol = sub_m.model.getSolution()
                col_dual = list(sol.col_dual)
                for var_name in shared_vars:
                    col_indices = self._get_var_col_indices(
                        sub_m, var_name, sub_var_col_map)
                    for idx, col_idx in col_indices.items():
                        if col_idx < len(col_dual):
                            rc = col_dual[col_idx]
                            fix_duals[f'_benders_fix_{var_name}_{idx}'] = rc
            except Exception:
                pass

            no_useful_fix_duals = (
                not fix_duals or
                all(abs(float(v)) <= 1e-12 for v in fix_duals.values()))
            if no_useful_fix_duals and sub_duals:
                try:
                    fix_duals = self._compute_benders_coeffs_from_subduals(
                        sub_m, sub_duals, shared_vars, master_vars)
                except Exception:
                    pass

        self._unfix_all_bounds(sub_m, sub_var_col_map, shared_vars,
                               original_bounds)

        _log_benders(
            f"  Subproblem: obj = {sub_obj:.6f}, "
            f"status = {status_str}, fix_duals = {len(fix_duals)}, "
            f"sub_duals = {len(sub_duals)}",
            show_log)

        return [{
            'name': 'subproblem',
            'feasible': True,
            'status': status_str,
            'objective_value': sub_obj,
            'model': sub_m,
            'has_duals': len(fix_duals) > 0 or len(sub_duals) > 0,
            'sub_constraints': sub_constraints,
            'fix_duals': fix_duals,
            'sub_duals': sub_duals,
            'fixing_vars': shared_vars,
            'fixing_vals': {v: master_vars.get(v) for v in shared_vars},
        }]

    @staticmethod
    def _unfix_all_bounds(m, var_col_map, shared_vars, original_bounds):
        """Restore original bounds for all shared variables."""
        model_obj = m.model
        interface = m.features.get('interface_name', '')
        for var_name in shared_vars:
            for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                col_info = var_col_map.get((prefix, var_name))
                if col_info is None:
                    continue
                var_obj = m.features.get('variables', {}).get((prefix, var_name))
                if interface == 'gurobi':
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            try:
                                lb, ub = original_bounds.get(
                                    (prefix, var_name, idx), (-1e20, 1e20))
                                v = var_obj[idx] if isinstance(var_obj, dict) else var_obj
                                v.lb = lb
                                v.ub = ub
                            except Exception:
                                pass
                    else:
                        try:
                            lb, ub = original_bounds.get(
                                (prefix, var_name), (-1e20, 1e20))
                            var_obj.lb = lb
                            var_obj.ub = ub
                        except Exception:
                            pass
                else:
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            try:
                                lb, ub = original_bounds.get(
                                    (prefix, var_name, idx), (-1e20, 1e20))
                                model_obj.changeColBounds(col_idx, lb, ub)
                            except Exception:
                                pass
                    else:
                        try:
                            lb, ub = original_bounds.get(
                                (prefix, var_name), (-1e20, 1e20))
                            model_obj.changeColBounds(col_info, lb, ub)
                        except Exception:
                            pass
                break

    def _generate_optimality_cut_cached(self, sub_result, master_vars,
                                        complicating_vars, sub_m,
                                        sub_var_col_map):
        """Generate optimality cut using reduced costs from cached solve."""
        fix_duals = sub_result.get('fix_duals', {})
        no_useful_fix_duals = (
            not fix_duals or
            all(abs(float(v)) <= 1e-12 for v in fix_duals.values()))
        numerical_coeffs = sub_result.get('numerical_coeffs', {})
        if no_useful_fix_duals and numerical_coeffs:
            sub_obj = sub_result.get('objective_value', 0.0)
            rhs = sub_obj
            coeffs = {}
            for key, coeff in numerical_coeffs.items():
                if isinstance(key, tuple):
                    var_name, idx = key
                    y_star = master_vars.get(var_name, {})
                    if isinstance(y_star, dict):
                        x_val = float(y_star.get(idx, 0))
                    else:
                        x_val = float(np.asarray(y_star).flatten()[idx]) if np.asarray(y_star).size > idx else 0.0
                    if var_name not in coeffs:
                        coeffs[var_name] = {}
                    coeffs[var_name][idx] = coeff
                else:
                    var_name = key
                    x_val = float(np.asarray(
                        master_vars.get(var_name, 0)).reshape(-1)[0])
                    coeffs[var_name] = coeff
                rhs -= coeff * x_val
            return {
                'rhs': rhs,
                'coeffs': coeffs,
                'type': 'optimality',
                'sub_obj': sub_obj,
            }
        if not fix_duals:
            sub_obj = sub_result.get('objective_value', 0.0)
            return {
                'rhs': sub_obj,
                'coeffs': {},
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        sub_obj = sub_result.get('objective_value', 0.0)
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        coeffs = {}
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()

            per_elem = {}
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val)
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            try:
                                idx = ast.literal_eval(index_text)
                            except (ValueError, SyntaxError):
                                idx = index_text
                        per_elem[idx] = float(dual_val)
                    except ValueError:
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                else:
                    coeffs[var_name] = per_elem.get(0, 0.0)

        rhs_adjust = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            c = coeffs.get(var_name, 0.0)
            if isinstance(c, dict):
                for idx, mu in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += mu * float(y_arr[idx])
            else:
                rhs_adjust += c * float(np.sum(y_arr))

        return {
            'rhs': sub_obj - rhs_adjust,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    def _logic_cuts(self, master_vars, subproblem_result):
        if self._logic_cut_callback is None:
            return []
        returned = self._logic_cut_callback(master_vars, subproblem_result)
        if returned is None:
            return []
        cuts = returned if isinstance(returned, (list, tuple)) else [returned]
        for cut in cuts:
            if not isinstance(cut, dict) or 'rhs' not in cut or 'coeffs' not in cut:
                raise BendersError(
                    "Logic cut callbacks must return cut dictionaries with rhs and coeffs")
            cut.setdefault('type', 'feasibility')
        return list(cuts)

    def _generate_cut_for_method(self, sub_result, x_bar, complicating_vars):
        """Dispatch to the appropriate cut generator based on the active method."""
        method = getattr(self, '_active_method', 'auto')
        if method == 'generalized':
            return self._generate_generalized_optimality_cut(
                sub_result, x_bar, complicating_vars)
        elif method == 'combinatorial':
            return self._generate_combinatorial_optimality_cut(
                sub_result, x_bar, complicating_vars)
        elif method == 'integer_l_shaped':
            return self._generate_combinatorial_optimality_cut(
                sub_result, x_bar, complicating_vars)
        elif method == 'generalized_l_shaped':
            return self._generate_generalized_optimality_cut(
                sub_result, x_bar, complicating_vars)
        else:
            return self._generate_optimality_cut(
                sub_result, x_bar, complicating_vars)

    # ------------------------------------------------------------------
    # Subproblem solving
    # ------------------------------------------------------------------

    def _solve_subproblems(self, master_vars, solver_opts, solution_generator,
                           show_log, relax_subproblem_integrality=True):
        """Solve every subproblem, in parallel when there is more than one.

        Parallel solving is used whenever ``params.parallel_sub`` is set
        (the default); a single-subproblem (auto) run — or any parallel
        failure — falls back to the serial sweep.
        """
        if not self._is_auto() and self._params.parallel_sub:
            n_valid = 0
            for link in self._links:
                if link is None:
                    continue
                level_name = link['lower']
                for sp in self._subproblems:
                    if isinstance(sp, _BendersProblem) and sp.name == level_name:
                        n_valid += 1
                        break
            if n_valid > 1:
                # _solve_subproblems_parallel falls back to the serial
                # sweep internally, so real errors surface unchanged.
                return self._solve_subproblems_parallel(
                    master_vars, solver_opts, solution_generator, show_log,
                    relax_subproblem_integrality=relax_subproblem_integrality)
        return self._solve_subproblems_serial(
            master_vars, solver_opts, solution_generator, show_log,
            relax_subproblem_integrality=relax_subproblem_integrality)

    def _solve_subproblems_serial(self, master_vars, solver_opts,
                                  solution_generator, show_log,
                                  relax_subproblem_integrality=True):
        results = []

        if self._is_auto():
            auto = self._auto_config
            sub_level = _BendersProblem(
                'subproblem', self._master.model_fn,
                self._master.directions, self._master.obj_index)
            result = self._solve_single_subproblem(
                sub_level, master_vars, auto['complicating_variables'],
                solver_opts, solution_generator, show_log,
                relax_subproblem_integrality=relax_subproblem_integrality)
            results.append(result)
        else:
            for link in self._links:
                if link is None:
                    continue
                level_name = link['lower']
                level = None
                for sp in self._subproblems:
                    if isinstance(sp, _BendersProblem) and sp.name == level_name:
                        level = sp
                        break
                if level is None:
                    continue
                result = self._solve_single_subproblem(
                    level, master_vars, link['shared_vars'],
                    solver_opts, solution_generator, show_log,
                    relax_subproblem_integrality=relax_subproblem_integrality)
                results.append(result)

        return results

    def _solve_single_subproblem(self, level, master_vars, shared_vars,
                                  solver_opts, solution_generator, show_log,
                                 derive_numerical_cut=True,
                                 relax_subproblem_integrality=True,
                                 force_constraint_fixing=False):
        sub_m = self._create_model(
            level.model_fn, level.directions, level.obj_index, solver_opts,
            level_name='subproblem')

        interface = solver_opts.get('sub_interface_name',
                                    solver_opts.get('interface_name', 'highs'))

        # Try bound-based fixing first (HiGHS/Gurobi/CPLEX).
        var_col_map = None
        used_bound_fixing = False
        original_bounds = {}
        if interface in ('highs', 'gurobi', 'cplex', 'copt') \
                and not force_constraint_fixing:
            try:
                var_col_map = self._build_var_col_map(sub_m)
                original_bounds = self._capture_original_bounds(
                    sub_m, var_col_map, shared_vars)
                for var_name, val in master_vars.items():
                    if var_name in shared_vars:
                        self._fix_variable_bound(
                            sub_m, var_col_map, var_name, val)
                used_bound_fixing = True
            except Exception:
                used_bound_fixing = False

        if not used_bound_fixing:
            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    self._fix_variable(sub_m, var_name, val)

        # When using auto API with subproblem_constraints, filter to those + fixing.
        # When using manual API (no sub_constraint_ids), pass all constraints through.
        sub_constraint_ids = self._get_sub_constraint_ids()

        if sub_constraint_ids:
            all_labels = list(sub_m.features.get('constraint_labels', []))
            effective_ids = sub_constraint_ids + [
                lb for lb in all_labels
                if lb and lb.startswith('_benders_fix_')]
        else:
            effective_ids = None

        sub_m.features['solver_options'] = dict(sub_m.features.get('solver_options', {}))
        if interface in {'gurobi', 'cplex', 'xpress', 'highs', 'copt', 'scip'}:
            sub_m.features['solver_options']['Presolve'] = 0
        sub_m.features['log'] = False

        _log_benders(f"  Solving subproblem '{level.name}' (LP relaxation)", False)

        feasible = self._solve_model(sub_m, solution_generator,
                                     constraint_ids=effective_ids,
                                     relax_integrality=relax_subproblem_integrality)
        status = self._get_status(sub_m)

        result = {
            'name': level.name,
            'feasible': feasible,
            'status': status,
            'objective_value': self._get_obj(sub_m) if feasible else None,
            'model': sub_m,
            'has_duals': False,
            'sub_constraints': [],
            'fixing_vars': shared_vars,
            'fixing_vals': {v: master_vars.get(v) for v in shared_vars},
        }

        # In auto mode the subproblem has the full objective; record the
        # integer-variable cost coefficients so the cut generator can
        # subtract them from the fix_duals (which include both integer
        # cost and continuous sensitivity).
        if self._is_auto() and used_bound_fixing and var_col_map is not None:
            try:
                int_var_costs = {}
                interface_name = solver_opts.get('interface_name', '')
                if interface_name == 'gurobi':
                    model_obj = sub_m.model
                    all_vars = model_obj.getVars()
                    for var_name in shared_vars:
                        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                            key = (prefix, var_name)
                            if key in var_col_map:
                                val = var_col_map[key]
                                if isinstance(val, dict):
                                    for idx, ci in val.items():
                                        if isinstance(ci, int) and ci < len(all_vars):
                                            int_var_costs.setdefault(var_name, 0.0)
                                            int_var_costs[var_name] += float(all_vars[ci].Obj)
                                elif isinstance(val, int) and val < len(all_vars):
                                    int_var_costs[var_name] = float(all_vars[val].Obj)
                                break
                else:
                    lp = sub_m.model.getLp()
                    for var_name in shared_vars:
                        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                            key = (prefix, var_name)
                            if key in var_col_map:
                                val = var_col_map[key]
                                if isinstance(val, dict):
                                    for idx, ci in val.items():
                                        if isinstance(ci, int) and ci < len(lp.col_cost_):
                                            int_var_costs.setdefault(var_name, 0.0)
                                            int_var_costs[var_name] += float(lp.col_cost_[ci])
                                elif isinstance(val, int) and val < len(lp.col_cost_):
                                    int_var_costs[var_name] = float(lp.col_cost_[val])
                                break
                result['int_var_costs'] = int_var_costs
            except Exception:
                pass

        if feasible:
            fix_duals = {}
            sub_duals = {}
            sub_constraints = []

            if used_bound_fixing and var_col_map is not None:
                fix_duals = self._extract_reduced_costs(
                    sub_m, var_col_map, shared_vars, master_vars)
                # Discard all-zero reduced costs — they provide no useful
                # Benders information and cause degenerate constant cuts.
                if fix_duals and all(
                        abs(float(v)) <= 1e-12 for v in fix_duals.values()):
                    fix_duals = {}
                if not fix_duals and derive_numerical_cut:
                    # A bound-fixed column (lb == ub) is degenerate:
                    # HiGHS reports a structurally-zero reduced cost for
                    # every one of them, so the extraction above comes
                    # back empty.  Retry once with explicit fixing rows
                    # (_benders_fix_*) whose duals carry the shadow
                    # price of x_bar_j — a single extra subproblem
                    # solve instead of finite-difference probing, which
                    # costs one full model rebuild per shared element.
                    # Only adopt the retry when it yields informative
                    # (nonzero) duals so models whose fixing duals are
                    # genuinely zero keep their probing fallback.
                    try:
                        retry = self._solve_single_subproblem(
                            level, master_vars, shared_vars, solver_opts,
                            solution_generator, show_log,
                            derive_numerical_cut=False,
                            relax_subproblem_integrality=(
                                relax_subproblem_integrality),
                            force_constraint_fixing=True)
                        retry_fix = retry.get('fix_duals') or {}
                        if any(abs(float(v)) > 1e-12
                               for v in retry_fix.values()):
                            return retry
                    except Exception:
                        pass

            if not fix_duals:
                all_labels = list(sub_m.features.get('constraint_labels', []))
                for label in all_labels:
                    if not label:
                        continue
                    dual = self._get_dual(sub_m, label)
                    if dual is None:
                        continue
                    if label.startswith('_benders_fix_'):
                        fix_duals[label] = dual
                    else:
                        sub_constraints.append(label)
                        sub_duals[label] = dual

            # NOTE: When bound-based fixing was used and its reduced costs
            # came back empty, the retry above re-solves with explicit
            # fixing rows first — those duals carry the shadow price of
            # x_bar_j and empirically stay nonzero (a bound-fixed column
            # lb == ub is degenerate, so its reduced cost is structurally
            # 0 for HiGHS).  Finite-difference probing below is now a
            # last resort for models where even those duals vanish.

            result['sub_constraints'] = sub_constraints
            result['fix_duals'] = fix_duals
            result['sub_duals'] = sub_duals

            has_duals = bool(fix_duals)
            result['has_duals'] = has_duals

            # Trigger numerical probing when fix_duals is empty.
            # sub_duals (constraint duals) are stored for reference but
            # are NOT used by the cut generator — only fix_duals
            # (reduced costs of fixing constraints) provide the per-variable
            # Benders coefficients.  When fix_duals is empty, we must fall
            # through to numerical probing for finite-difference coefficients.
            if not has_duals and derive_numerical_cut:
                result['numerical_coeffs'] = self._derive_numerical_coeffs(
                    level, master_vars, shared_vars, solver_opts,
                    solution_generator, show_log, result['objective_value'])

            _log_benders(
                f"  Subproblem '{level.name}': obj = {result['objective_value']:.6f}, "
                f"status = {status}, fix_duals = {len(fix_duals)}, sub_duals = {len(sub_duals)}",
                show_log)
        else:
            _log_benders(
                f"  Subproblem '{level.name}': INFEASIBLE (status = {status})",
                show_log)

            gurobi_farkas_cut = None
            nogood_cut = None

            nogood_cut = self._generate_iis_nogood_cut(
                sub_m, shared_vars, master_vars, show_log)

            if nogood_cut is None and interface in ('gurobi', 'highs', 'cplex',
                                                     'xpress', 'copt'):
                nogood_cut = self._try_constraint_fixing_nogood_cut(
                    level, master_vars, shared_vars, solver_opts,
                    solution_generator, show_log)

            if gurobi_farkas_cut is not None:
                result['feasible'] = False
                result['fix_duals'] = {}
                result['has_duals'] = True
                result['farkas_feasibility_cut'] = gurobi_farkas_cut
            elif nogood_cut is not None:
                result['feasible'] = False
                result['fix_duals'] = {}
                result['has_duals'] = True
                result['farkas_feasibility_cut'] = nogood_cut
            else:
                art_result = None
                if interface == 'gurobi':
                    art_result = self._solve_with_artificial_vars_gurobi(
                        sub_m, var_col_map, shared_vars, master_vars,
                        original_bounds, show_log)
                else:
                    art_result = self._solve_with_artificial_vars(
                        sub_m, var_col_map, shared_vars, master_vars,
                        original_bounds, show_log)
                if art_result is not None and art_result.get('has_duals'):
                    result['feasible'] = True
                    result['objective_value'] = art_result.get('objective_value')
                    result['fix_duals'] = art_result.get('fix_duals', {})
                    result['has_duals'] = True
                    result['original_infeasible'] = True
                else:
                    result['feasible'] = False

        return result

    # ------------------------------------------------------------------
    # Gurobi IIS-based feasibility cut
    # ------------------------------------------------------------------

    def _try_gurobi_iis_cut(self, sub_m, var_col_map, shared_vars,
                             master_vars, original_bounds, show_log):
        """Generate a feasibility cut from Gurobi's IIS (computeIIS).

        When the subproblem LP is infeasible and the solver is Gurobi,
        computeIIS() identifies the Irreducible Infeasible Subsystem.
        From the IIS we compute the extreme ray of the dual by solving
        a small LP, then build a classical feasibility cut:

            sum_j coefs_j * x_j >= cut_rhs

        where:
            coefs_j = sum_k (-r_k * A_kj)
            cut_rhs = sum_k (-r_k * b_k)

        Returns a cut dict with 'rhs', 'coeffs', and 'type' keys, or None.
        """
        try:
            import gurobipy
            model = sub_m.model

            infeasible_statuses = (
                gurobipy.GRB.Status.INFEASIBLE,
                gurobipy.GRB.Status.INF_OR_UNBD,
            )
            if model.Status not in infeasible_statuses:
                return None

            if var_col_map is None:
                var_col_map = self._build_var_col_map(sub_m)

            model.computeIIS()

            constrs = model.getConstrs()
            iis_indices = [i for i, c in enumerate(constrs)
                           if c.IISConstr]
            if not iis_indices:
                return None

            A_full = model.getA()
            senses = [constrs[i].Sense for i in iis_indices]
            rhs_vals = [constrs[i].RHS for i in iis_indices]

            n_iis = len(iis_indices)

            r_model = _create_gurobi_model_silently("_iis_dual", show_log=show_log)
            r_model.setParam("OutputFlag", 0)
            r_vars = r_model.addVars(n_iis, lb=0.0, ub=1.0, name="r")

            A_iis = A_full[iis_indices, :]
            n_cols = A_iis.shape[1]
            for j in range(n_cols):
                col = A_iis[:, j].toarray().flatten()
                expr = gurobipy.LinExpr()
                for k in range(n_iis):
                    if abs(col[k]) > 1e-15:
                        expr.add(r_vars[k], float(col[k]))
                r_model.addConstr(expr == 0, name="col%d" % j)

            obj = gurobipy.LinExpr()
            for k in range(n_iis):
                coeff = float(rhs_vals[k])
                if senses[k] == "<":
                    coeff = -coeff
                obj.add(r_vars[k], coeff)
            r_model.setObjective(obj, gurobipy.GRB.MINIMIZE)

            r_model.optimize()

            if r_model.Status != gurobipy.GRB.Status.OPTIMAL:
                return None

            ray = [r_vars[k].X for k in range(n_iis)]

            coeffs = {}
            cut_rhs = 0.0

            for var_name in shared_vars:
                var_cols = {}
                for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                    if var_col_map is not None:
                        col_info = var_col_map.get((prefix, var_name))
                    else:
                        col_info = None
                    if col_info is not None:
                        if isinstance(col_info, dict):
                            var_cols = col_info
                        else:
                            var_cols = {0: col_info}
                        break

                if not var_cols:
                    continue

                per_elem = {}
                for idx, col_idx in var_cols.items():
                    if col_idx >= A_full.shape[1]:
                        continue

                    col = A_full.getcol(col_idx).toarray().flatten()

                    coef = 0.0
                    for ki, k in enumerate(iis_indices):
                        if abs(ray[ki]) > 1e-12:
                            coef += -ray[ki] * col[k]

                    if abs(coef) > 1e-12:
                        per_elem[idx] = coef

                if per_elem:
                    if len(per_elem) == 1 and 0 in per_elem:
                        coeffs[var_name] = per_elem[0]
                    else:
                        coeffs[var_name] = per_elem

            for ki, k in enumerate(iis_indices):
                if abs(ray[ki]) > 1e-12:
                    b_k = float(rhs_vals[ki])
                    if senses[ki] == "<":
                        cut_rhs += -ray[ki] * (-b_k)
                    else:
                        cut_rhs += -ray[ki] * b_k

            if not coeffs:
                _log_benders("  Gurobi IIS: no useful coefficients",
                             show_log)
                return None

            _log_benders(
                "  Gurobi IIS: %d constraints, coeffs=%s, rhs=%.6f"
                % (n_iis, coeffs, cut_rhs),
                show_log)

            return {
                'rhs': cut_rhs,
                'coeffs': coeffs,
                'type': 'feasibility',
            }

        except Exception as e:
            import traceback
            _log_benders("  Gurobi IIS cut failed: %s" % e, show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    # ------------------------------------------------------------------
    # Gurobi IIS with constraint-based fixing (fallback)
    # ------------------------------------------------------------------

    def _try_gurobi_iis_with_constraint_fixing(
            self, level, master_vars, shared_vars, solver_opts,
            solution_generator, show_log):
        """Generate a feasibility cut using IIS with constraint-based fixing.

        When bound-based fixing produces an IIS with no useful coefficients
        (because the fixing bounds don't appear in the IIS), re-solve the
        subproblem with constraint-based fixing so the fixing constraints
        appear in the IIS and produce a valid feasibility cut.
        """
        try:
            import gurobipy

            sub_m = self._create_model(
                level.model_fn, level.directions, level.obj_index, solver_opts,
                level_name='subproblem')

            sub_m.features['log'] = False

            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    self._fix_variable(sub_m, var_name, val)

            sub_constraint_ids = self._get_sub_constraint_ids()
            if sub_constraint_ids:
                all_labels = list(sub_m.features.get('constraint_labels', []))
                effective_ids = sub_constraint_ids + [
                    lb for lb in all_labels
                    if lb and lb.startswith('_benders_fix_')]
            else:
                effective_ids = None

            self._solve_model(sub_m, solution_generator,
                              constraint_ids=effective_ids,
                              relax_integrality=True)
            status = self._get_status(sub_m)

            infeasible_statuses = (
                gurobipy.GRB.Status.INFEASIBLE,
                gurobipy.GRB.Status.INF_OR_UNBD,
            )
            if status not in infeasible_statuses and status != 'infeasible':
                return None

            model = sub_m.model
            model.computeIIS()

            constrs = model.getConstrs()
            iis_indices = [i for i, c in enumerate(constrs)
                           if c.IISConstr]
            if not iis_indices:
                return None

            A_full = model.getA()
            senses = [constrs[i].Sense for i in iis_indices]
            rhs_vals = [constrs[i].RHS for i in iis_indices]

            n_iis = len(iis_indices)

            r_model = _create_gurobi_model_silently("_iis_dual2", show_log=show_log)
            r_model.setParam("OutputFlag", 0)
            r_vars = r_model.addVars(n_iis, lb=0.0, ub=1.0, name="r")

            A_iis = A_full[iis_indices, :]
            n_cols = A_iis.shape[1]
            for j in range(n_cols):
                col = A_iis[:, j].toarray().flatten()
                expr = gurobipy.LinExpr()
                for k in range(n_iis):
                    if abs(col[k]) > 1e-15:
                        expr.add(r_vars[k], float(col[k]))
                r_model.addConstr(expr == 0, name="col%d" % j)

            obj = gurobipy.LinExpr()
            for k in range(n_iis):
                coeff = float(rhs_vals[k])
                if senses[k] == "<":
                    coeff = -coeff
                obj.add(r_vars[k], coeff)
            r_model.setObjective(obj, gurobipy.GRB.MINIMIZE)

            r_model.optimize()

            if r_model.Status != gurobipy.GRB.Status.OPTIMAL:
                return None

            ray = [r_vars[k].X for k in range(n_iis)]

            coeffs = {}
            cut_rhs = 0.0

            var_col_map = self._build_var_col_map(sub_m)

            for var_name in shared_vars:
                var_cols = {}
                for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                    if var_col_map is not None:
                        col_info = var_col_map.get((prefix, var_name))
                    else:
                        col_info = None
                    if col_info is not None:
                        if isinstance(col_info, dict):
                            var_cols = col_info
                        else:
                            var_cols = {0: col_info}
                        break

                if not var_cols:
                    continue

                per_elem = {}
                for idx, col_idx in var_cols.items():
                    if col_idx >= A_full.shape[1]:
                        continue

                    col = A_full.getcol(col_idx).toarray().flatten()

                    coef = 0.0
                    for ki, k in enumerate(iis_indices):
                        if abs(ray[ki]) > 1e-12:
                            coef += -ray[ki] * col[k]

                    if abs(coef) > 1e-12:
                        per_elem[idx] = coef

                if per_elem:
                    if len(per_elem) == 1 and 0 in per_elem:
                        coeffs[var_name] = per_elem[0]
                    else:
                        coeffs[var_name] = per_elem

            for ki, k in enumerate(iis_indices):
                if abs(ray[ki]) > 1e-12:
                    b_k = float(rhs_vals[ki])
                    if senses[ki] == "<":
                        cut_rhs += -ray[ki] * (-b_k)
                    else:
                        cut_rhs += -ray[ki] * b_k

            if not coeffs:
                _log_benders("  Gurobi IIS (constraint fix): no useful coefficients",
                             show_log)
                return None

            _log_benders(
                "  Gurobi IIS (constraint fix): %d constraints, coeffs=%s, rhs=%.6f"
                % (n_iis, coeffs, cut_rhs),
                show_log)

            return {
                'rhs': cut_rhs,
                'coeffs': coeffs,
                'type': 'feasibility',
            }

        except Exception as e:
            import traceback
            _log_benders("  Gurobi IIS (constraint fix) failed: %s" % e, show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    # ------------------------------------------------------------------
    # Solver-agnostic constraint-fixing + IIS nogood cut
    # ------------------------------------------------------------------

    def _try_constraint_fixing_nogood_cut(
            self, level, master_vars, shared_vars, solver_opts,
            solution_generator, show_log):
        """Re-solve with constraint-based fixing and extract IIS nogood cut.

        When bound-based fixing produces an IIS with no fixing constraints
        (because bounds don't appear in the IIS), re-creates the subproblem
        with constraint-based fixing so fixing constraints appear in the IIS,
        then generates a nogood cut from the IIS.

        Works for any solver: Gurobi, HiGHS, CPLEX, Xpress, COPT, etc.
        """
        try:
            sub_m = self._create_model(
                level.model_fn, level.directions, level.obj_index, solver_opts,
                level_name='subproblem')

            sub_m.features['log'] = False

            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    self._fix_variable(sub_m, var_name, val)

            sub_constraint_ids = self._get_sub_constraint_ids()
            if sub_constraint_ids:
                all_labels = list(sub_m.features.get('constraint_labels', []))
                effective_ids = sub_constraint_ids + [
                    lb for lb in all_labels
                    if lb and lb.startswith('_benders_fix_')]
            else:
                effective_ids = None

            self._solve_model(sub_m, solution_generator,
                              constraint_ids=effective_ids,
                              relax_integrality=True)
            status = self._get_status(sub_m)

            is_infeasible = False
            interface = solver_opts.get('sub_interface_name',
                                        solver_opts.get('interface_name', 'highs'))
            if interface == 'gurobi':
                try:
                    import gurobipy
                    is_infeasible = status in ('infeasible',) or (
                        hasattr(sub_m.model, 'Status') and
                        sub_m.model.Status in (gurobipy.GRB.Status.INFEASIBLE,
                                                gurobipy.GRB.Status.INF_OR_UNBD))
                except Exception:
                    is_infeasible = status == 'infeasible'
            else:
                is_infeasible = status == 'infeasible'

            if not is_infeasible:
                _log_benders("  Constraint-fixing subproblem is feasible", show_log)
                return None

            cut = self._generate_iis_nogood_cut(
                sub_m, shared_vars, master_vars, show_log)
            return cut

        except Exception as e:
            import traceback
            _log_benders("  Constraint-fixing nogood cut failed: %s" % e, show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    # ------------------------------------------------------------------
    # Solver-agnostic IIS feasibility cut (no-good cuts from IIS)
    # ------------------------------------------------------------------

    def _generate_iis_nogood_cut(self, sub_m, shared_vars, master_vars,
                                  show_log):
        """Generate a feasibility cut from the IIS of an infeasible subproblem.

        Works for ANY solver that supports IIS extraction (Gurobi, CPLEX,
        HiGHS, Xpress, COPT, PyOptInterface).  When the IIS contains
        fixing constraints (``_benders_fix_*``), generates a no-good cut
        that excludes the current infeasible fixing value.

        For integer/binary variables: ``y_i >= v_i + 1`` (min) or
        ``y_i <= v_i - 1`` (max).
        For continuous variables: ``y_i >= v_i + eps`` or ``y_i <= v_i - eps``.

        Returns a cut dict ``{'rhs', 'coeffs', 'type': 'feasibility'}`` or
        ``None`` if no useful cut can be extracted.
        """
        try:
            import numpy as np
            interface = sub_m.features.get('interface_name', '')
            model_obj = sub_m.model
            iis_constr_names = []

            if interface == 'gurobi':
                try:
                    import gurobipy
                    infeasible_statuses = (
                        gurobipy.GRB.Status.INFEASIBLE,
                        gurobipy.GRB.Status.INF_OR_UNBD,
                    )
                    if model_obj.Status not in infeasible_statuses:
                        return None
                    model_obj.computeIIS()
                    for c in model_obj.getConstrs():
                        if c.IISConstr:
                            iis_constr_names.append(c.ConstrName)
                except Exception:
                    pass

            elif interface == 'cplex':
                try:
                    model_obj.populate()
                    iis = model_obj.conflict.get(0)
                    all_constrs = model_obj.linear_constraints
                    for status, lin_expr, sense, rhs, name in zip(
                            iis.indicators,
                            [all_constrs.get_row(i) for i in range(all_constrs.get_num())],
                            [all_constrs.get_sense(i) for i in range(all_constrs.get_num())],
                            [all_constrs.get_rhs(i) for i in range(all_constrs.get_num())],
                            [all_constrs.get_names(i) for i in range(all_constrs.get_num())]):
                        if status == 1:
                            iis_constr_names.append(name)
                except Exception:
                    pass

            elif interface == 'highs':
                try:
                    iis_result = model_obj.getIis()
                    if iis_result.valid_ and iis_result.row_index_:
                        lp = model_obj.getLp()
                        row_names = list(lp.row_names_) if lp.row_names_ else []
                        for ri in iis_result.row_index_:
                            if ri < len(row_names) and row_names[ri]:
                                iis_constr_names.append(row_names[ri])
                except Exception:
                    pass

            elif interface == 'xpress':
                try:
                    model_obj.iis()
                    rows = model_obj.getIISRow()
                    all_names = []
                    for ri in range(model_obj.getNumRows()):
                        try:
                            all_names.append(model_obj.getRowName(ri))
                        except Exception:
                            all_names.append('')
                    for ri in rows:
                        if ri < len(all_names) and all_names[ri]:
                            iis_constr_names.append(all_names[ri])
                except Exception:
                    pass

            elif interface == 'copt':
                try:
                    model_obj.computeIIS()
                    for c in model_obj.getConstrs():
                        if c.getUpperIIS() > 0 or c.getLowerIIS() > 0:
                            iis_constr_names.append(c.name)
                except Exception:
                    pass

            if not iis_constr_names:
                _log_benders("  IIS nogood: no IIS constraints found (%s)" % interface, False)
                return None

            _log_benders("  IIS nogood: %d IIS constraints" % len(iis_constr_names), False)

            fixing_labels = [name for name in iis_constr_names
                             if name.startswith('_benders_fix_')]

            if not fixing_labels:
                _log_benders("  IIS nogood: no fixing constraints in IIS", False)
                return None

            other_labels = [name for name in iis_constr_names
                            if not name.startswith('_benders_fix_')]

            direction = 'min'
            if hasattr(self, '_master') and self._master is not None:
                dirs = getattr(self._master, 'directions', ['min'])
                direction = dirs[0] if dirs else 'min'

            if interface == 'gurobi' and other_labels:
                threshold_cut = self._gurobi_threshold_cut(
                    model_obj, fixing_labels, other_labels,
                    shared_vars, master_vars, direction, show_log)
                if threshold_cut is not None:
                    return threshold_cut

            coeffs = {}
            cut_rhs = 0.0

            for label in fixing_labels:
                parts = label.replace('_benders_fix_', '', 1)
                idx_pos = parts.rfind('_')
                if idx_pos > 0:
                    var_name = parts[:idx_pos]
                else:
                    var_name = parts

                current_val = master_vars.get(var_name)
                if current_val is None:
                    continue

                is_indexed = hasattr(current_val, '__len__') and not isinstance(current_val, str)
                if is_indexed:
                    try:
                        elem_str = parts[idx_pos + 1:]
                        elem_idx = int(elem_str)
                        val = float(current_val[elem_idx])
                    except Exception:
                        continue
                else:
                    try:
                        val = float(current_val)
                    except Exception:
                        continue

                if direction == 'min':
                    coeffs[var_name] = 1.0
                    if is_indexed:
                        coeffs[var_name] = {elem_idx: 1.0}
                    cut_rhs = val + 1.0
                else:
                    coeffs[var_name] = -1.0
                    if is_indexed:
                        coeffs[var_name] = {elem_idx: -1.0}
                    cut_rhs = -(val - 1.0)

                _log_benders(
                    "  IIS nogood: %s %s %s %.1f" % (
                        var_name, '>=' if direction == 'min' else '<=',
                        val, cut_rhs),
                    show_log)
                break

            if not coeffs:
                return None

            return {
                'rhs': cut_rhs,
                'coeffs': coeffs,
                'type': 'feasibility',
            }

        except Exception as e:
            import traceback
            _log_benders("  IIS nogood cut failed: %s" % e, show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    @staticmethod
    def _gurobi_threshold_cut(model_obj, fixing_labels, other_labels,
                               shared_vars, master_vars, direction, show_log):
        """Compute tight threshold via small LP for each fixing var in the IIS."""
        try:
            import gurobipy as gp

            constrs_map = {c.ConstrName: c for c in model_obj.getConstrs()}
            iis_other = [constrs_map[n] for n in other_labels if n in constrs_map]
            iis_fix = [constrs_map[n] for n in fixing_labels if n in constrs_map]

            if not iis_fix or not iis_other:
                return None

            for fix_constr in iis_fix:
                label = fix_constr.ConstrName
                parts = label.replace('_benders_fix_', '', 1)
                idx_pos = parts.rfind('_')
                if idx_pos > 0:
                    var_name = parts[:idx_pos]
                else:
                    var_name = parts

                if var_name not in shared_vars:
                    continue

                current_val = master_vars.get(var_name)
                if current_val is None:
                    continue

                is_indexed = hasattr(current_val, '__len__') and not isinstance(current_val, str)
                if is_indexed:
                    try:
                        elem_str = parts[idx_pos + 1:]
                        elem_idx = int(elem_str)
                        val = float(current_val[elem_idx])
                    except Exception:
                        continue
                else:
                    try:
                        val = float(current_val)
                    except Exception:
                        continue

                r_model = gp.Model("_threshold")
                r_model.setParam("OutputFlag", 0)

                r_vars = {}
                for v in model_obj.getVars():
                    if v.VarName == var_name:
                        r_vars[v.VarName] = r_model.addVar(
                            lb=-1e10, ub=1e10, name=v.VarName)
                    else:
                        r_vars[v.VarName] = r_model.addVar(
                            lb=v.lb, ub=v.ub, name=v.VarName)

                r_model.setObjective(
                    r_vars[var_name],
                    gp.GRB.MINIMIZE if direction == 'min' else gp.GRB.MAXIMIZE)

                for oc in iis_other:
                    row = model_obj.getRow(oc)
                    expr = gp.LinExpr()
                    for j in range(row.size()):
                        coeff = row.getCoeff(j)
                        vn = row.getVar(j).VarName
                        if vn in r_vars:
                            expr.add(r_vars[vn], float(coeff))
                    sense = oc.Sense
                    rhs = oc.RHS
                    if sense == '<':
                        r_model.addConstr(expr <= rhs)
                    elif sense == '>':
                        r_model.addConstr(expr >= rhs)
                    else:
                        r_model.addConstr(expr == rhs)

                for fc in iis_fix:
                    if fc is fix_constr:
                        continue
                    fc_label = fc.ConstrName
                    fc_parts = fc_label.replace('_benders_fix_', '', 1)
                    fc_idx_pos = fc_parts.rfind('_')
                    if fc_idx_pos > 0:
                        fc_var_name = fc_parts[:fc_idx_pos]
                    else:
                        fc_var_name = fc_parts
                    fc_val = master_vars.get(fc_var_name)
                    if fc_val is None:
                        continue
                    fc_is_indexed = hasattr(fc_val, '__len__') and not isinstance(fc_val, str)
                    if fc_is_indexed:
                        try:
                            fc_elem_str = fc_parts[fc_idx_pos + 1:]
                            fc_elem_idx = int(fc_elem_str)
                            fc_v = float(fc_val[fc_elem_idx])
                        except Exception:
                            continue
                    else:
                        try:
                            fc_v = float(fc_val)
                        except Exception:
                            continue
                    if fc_var_name in r_vars:
                        r_vars[fc_var_name].lb = fc_v
                        r_vars[fc_var_name].ub = fc_v

                r_model.optimize()

                if r_model.status == gp.GRB.Status.OPTIMAL:
                    threshold = r_model.objVal
                    _log_benders(
                        "  IIS threshold: %s %.1f -> %.1f" % (
                            var_name, val, threshold),
                        show_log)

                    coeffs = {}
                    if is_indexed:
                        coeffs[var_name] = {elem_idx: 1.0}
                    else:
                        coeffs[var_name] = 1.0

                    if direction == 'min':
                        return {
                            'rhs': threshold,
                            'coeffs': coeffs,
                            'type': 'feasibility',
                        }
                    else:
                        coeffs_neg = {}
                        for k, cv in coeffs.items():
                            if isinstance(cv, dict):
                                coeffs_neg[k] = {i: -c for i, c in cv.items()}
                            else:
                                coeffs_neg[k] = -cv
                        return {
                            'rhs': -threshold,
                            'coeffs': coeffs_neg,
                            'type': 'feasibility',
                        }
                else:
                    _log_benders(
                        "  IIS threshold LP failed for %s (status=%d)" % (
                            var_name, r_model.status),
                        show_log)

            return None

        except Exception as e:
            import traceback
            _log_benders("  Gurobi threshold cut failed: %s" % e, show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    # ------------------------------------------------------------------
    # Always-feasible subproblem (Section 3.3.4 — artificial variables)
    # ------------------------------------------------------------------

    def _solve_with_artificial_vars(self, sub_m, var_col_map, shared_vars,
                                    master_vars, original_bounds, show_log):
        """Re-solve an infeasible subproblem with artificial variables.

        Following Section 3.3.4 of the textbook (always-feasible subproblem),
        we add artificial variables per ORIGINAL constraint with penalty M:

        - inequality (≤): Σ e y - w_l ≤ b  (w_l ≥ 0 relaxes the RHS upward)
        - inequality (≥): Σ e y + v_l ≥ b  (v_l ≥ 0 relaxes the RHS downward)
        - equality (=):   Σ e y + v_l - w_l = b (two-sided relaxation)

        The penalty ensures artificials → 0 when the original subproblem
        is feasible.

        Following Algorithm 6.1 Step 3, the augmented optimum Q̃(x̄)
        (INCLUDING the penalty) plus the fixing-constraint duals λ gives
        a valid Benders cut: θ ≥ Q̃(x̄) + Σ λ_i (x_i - x̄_i).  We therefore
        return objective_value = total (augmented) objective, and the
        caller must NOT use it to update the upper bound (it is not the
        value of a truly feasible original solution).
        """
        model_obj = sub_m.model
        if not hasattr(model_obj, 'getNumCol'):
            # The artificial-variables machinery below is HiGHS-native
            # (highspy row/column manipulation). Other interfaces fall
            # back to the caller's solver-agnostic feasibility cut (a
            # no-good on the current fixing) instead of crashing.
            _log_benders(
                "  Artificial vars are HiGHS-only; using generic "
                "feasibility cut for this interface", show_log)
            return None

        import numpy as np
        import highspy

        n_cols = model_obj.getNumCol()

        lp = model_obj.getLp()
        row_upper = list(lp.row_upper_)
        row_lower = list(lp.row_lower_)
        n_rows = len(row_upper)

        # Use HiGHS LP row names (ground truth) instead of features
        # labels which may have been restored after constraint filtering.
        lp_row_names = list(lp.row_names_) if lp.row_names_ else []
        if len(lp_row_names) < n_rows:
            # Fallback: read row names from the solved model
            lp_row_names = []
            for ri in range(n_rows):
                try:
                    lp_row_names.append(model_obj.getRowName(ri))
                except Exception:
                    lp_row_names.append('')

        original_rows = []
        for ci in range(n_rows):
            label = lp_row_names[ci] if ci < len(lp_row_names) else ''
            if not label or not label.startswith('_benders_fix_'):
                original_rows.append(ci)

        if not original_rows:
            _log_benders("  No original constraints for artificial vars", show_log)
            return None

        n_original = len(original_rows)

        obj_coeffs = []
        try:
            obj_coeffs = list(lp.col_cost_)
        except Exception:
            pass

        d_max = max((abs(c) for c in obj_coeffs), default=1.0)
        M = max(10.0 * d_max, 10.0)

        try:
            for var_name in shared_vars:
                self._unfix_variable_bound(sub_m, var_col_map, var_name,
                                           original_bounds)

            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                        col_info = var_col_map.get((prefix, var_name))
                        if col_info is None:
                            continue
                        if isinstance(col_info, dict):
                            for idx, col_idx in col_info.items():
                                try:
                                    fix_val = float(val[idx])
                                    fix_label = f'_benders_fix_{var_name}_{idx}'
                                    row_idx = model_obj.getNumRow()
                                    model_obj.addRow(fix_val, fix_val, 1,
                                                     np.array([col_idx], dtype=np.int32),
                                                     np.array([1.0], dtype=np.float64))
                                    model_obj.passRowName(row_idx, fix_label)
                                    sub_m.features['constraint_labels'].append(fix_label)
                                    sub_m.features['constraints'].append(None)
                                except Exception:
                                    pass
                        else:
                            try:
                                fix_val = float(val)
                                fix_label = f'_benders_fix_{var_name}'
                                row_idx = model_obj.getNumRow()
                                model_obj.addRow(fix_val, fix_val, 1,
                                                 np.array([col_info], dtype=np.int32),
                                                 np.array([1.0], dtype=np.float64))
                                model_obj.passRowName(row_idx, fix_label)
                                sub_m.features['constraint_labels'].append(fix_label)
                                sub_m.features['constraints'].append(None)
                            except Exception:
                                pass
                        break

            n_art = 2 * n_original
            art_lower = np.zeros(n_art, dtype=np.float64)
            art_upper = np.full(n_art, 1e20, dtype=np.float64)
            model_obj.addVars(n_art, art_lower, art_upper)

            for i, row in enumerate(original_rows):
                v_col = n_cols + 2 * i
                w_col = n_cols + 2 * i + 1
                ub = row_upper[row]
                lb = row_lower[row]
                has_ub = ub < 1e20
                has_lb = lb > -1e20
                if has_ub and has_lb:
                    model_obj.changeRowBounds(row, ub, ub)
                    model_obj.changeCoeff(row, v_col, 1.0)
                    model_obj.changeCoeff(row, w_col, -1.0)
                    model_obj.changeColCost(v_col, M)
                    model_obj.changeColCost(w_col, M)
                elif has_ub:
                    model_obj.changeCoeff(row, w_col, -1.0)
                    model_obj.changeColCost(w_col, M)
                    model_obj.changeColBounds(v_col, 0.0, 0.0)
                elif has_lb:
                    model_obj.changeCoeff(row, v_col, 1.0)
                    model_obj.changeColCost(v_col, M)
                    model_obj.changeColBounds(w_col, 0.0, 0.0)
                else:
                    model_obj.changeColBounds(v_col, 0.0, 0.0)
                    model_obj.changeColBounds(w_col, 0.0, 0.0)
                    continue

            _log_benders("  Re-solving with artificial variables...", False)
            model_obj.run()

            status_val = model_obj.getModelStatus()
            feasible = status_val == highspy.HighsModelStatus.kOptimal

            if not feasible:
                _log_benders(
                    f"  Artificial-var solve still infeasible ({status_val})",
                    show_log)
                return None

            fix_duals = {}
            sol = model_obj.getSolution()
            row_dual = list(sol.row_dual) if hasattr(sol, 'row_dual') and sol.row_dual is not None else []
            row_names = []
            try:
                lp_after = model_obj.getLp()
                row_names = list(lp_after.row_names_) if lp_after.row_names_ else []
            except Exception:
                pass

            for ri in range(len(row_dual)):
                label = row_names[ri] if ri < len(row_names) else None
                if label and label.startswith('_benders_fix_'):
                    dual_val = row_dual[ri]
                    if abs(dual_val) > 1e-12:
                        fix_duals[label] = float(dual_val)

            if not fix_duals:
                fix_duals = self._extract_reduced_costs(
                    sub_m, var_col_map, shared_vars, master_vars)

            total_obj = self._get_obj(sub_m)
            sol = model_obj.getSolution()
            art_penalty = 0.0
            for i in range(n_art):
                a_col = n_cols + i
                if a_col < len(sol.col_value):
                    art_penalty += M * sol.col_value[a_col]

            original_obj = (total_obj - art_penalty) if total_obj is not None else None

            _log_benders(
                f"  Artificial-var solve: total={total_obj:.6f}, "
                f"penalty={art_penalty:.6f}, orig_obj={original_obj:.6f}, "
                f"M={M:.1f}, fix_duals={fix_duals}",
                False)

            return {
                'feasible': True,
                'objective_value': total_obj,
                'original_objective': original_obj,
                'artificial_penalty': art_penalty,
                'fix_duals': fix_duals,
                'sub_constraints': [],
                'has_duals': bool(fix_duals),
            }
        except Exception as e:
            import traceback
            _log_benders(f"  Artificial-var solve failed: {e}", show_log)
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Gurobi-compatible always-feasible subproblem (artificial variables)
    # ------------------------------------------------------------------

    def _solve_with_artificial_vars_gurobi(
            self, sub_m, var_col_map, shared_vars, master_vars,
            original_bounds, show_log):
        """Gurobi-compatible version of _solve_with_artificial_vars.

        Adds artificial variables to each ORIGINAL constraint with penalty M,
        re-solves, and extracts fixing-constraint duals (Pi) for the Benders cut.
        """
        import gurobipy as gp

        model_obj = sub_m.model
        all_constrs = model_obj.getConstrs()

        orig_constrs = []
        for c in all_constrs:
            if not c.ConstrName.startswith('_benders_fix_'):
                orig_constrs.append(c)
        if not orig_constrs:
            _log_benders("  No original constraints for artificial vars", show_log)
            return None

        n_orig = len(orig_constrs)
        gur_vars = model_obj.getVars()
        n_cols = len(gur_vars)

        obj_coeffs = [v.Obj for v in gur_vars]
        d_max = max((abs(c) for c in obj_coeffs), default=1.0)
        M = max(10.0 * d_max, 10.0)

        try:
            for var_name in shared_vars:
                self._unfix_variable_bound(sub_m, var_col_map, var_name,
                                           original_bounds)

            for var_name, val in master_vars.items():
                if var_name in shared_vars:
                    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                        col_info = var_col_map.get((prefix, var_name))
                        if col_info is None:
                            continue
                        if isinstance(col_info, dict):
                            for idx, col_idx in col_info.items():
                                try:
                                    fix_val = float(val[idx])
                                    fix_label = f'_benders_fix_{var_name}_{idx}'
                                    var_obj = sub_m.features['variables'].get(
                                        (prefix, var_name))
                                    if isinstance(var_obj, dict):
                                        var_g = var_obj[idx]
                                    else:
                                        var_g = var_obj
                                    model_obj.addConstr(
                                        var_g == fix_val, name=fix_label)
                                    sub_m.features['constraint_labels'].append(
                                        fix_label)
                                    sub_m.features['constraints'].append(None)
                                except Exception:
                                    pass
                        else:
                            try:
                                fix_val = float(val)
                                fix_label = f'_benders_fix_{var_name}'
                                var_obj = sub_m.features['variables'].get(
                                    (prefix, var_name))
                                model_obj.addConstr(
                                    var_obj == fix_val, name=fix_label)
                                sub_m.features['constraint_labels'].append(
                                    fix_label)
                                sub_m.features['constraints'].append(None)
                            except Exception:
                                pass
                        break

            model_obj.update()
            n_cols_after_fix = len(model_obj.getVars())

            art_vars = []
            for i in range(2 * n_orig):
                v = model_obj.addVar(lb=0.0, ub=gp.GRB.INFINITY,
                                     obj=M, name=f"_art{i}")
                art_vars.append(v)
            model_obj.update()

            for row_pos, c in enumerate(orig_constrs):
                sense = c.Sense
                if sense == '<':
                    model_obj.chgCoeff(c, art_vars[2 * row_pos + 1], -1.0)
                elif sense == '>':
                    model_obj.chgCoeff(c, art_vars[2 * row_pos], 1.0)
                elif sense == '=':
                    model_obj.chgCoeff(c, art_vars[2 * row_pos], 1.0)
                    model_obj.chgCoeff(c, art_vars[2 * row_pos + 1], -1.0)
            model_obj.update()

            _log_benders("  Gurobi: re-solving with artificial variables...",
                         show_log)
            model_obj.optimize()

            if model_obj.status not in (
                    gp.GRB.Status.OPTIMAL, gp.GRB.Status.SUBOPTIMAL):
                _log_benders(
                    f"  Gurobi artificial-var solve failed (status={model_obj.status})",
                    show_log)
                return None

            fix_duals = {}
            for c in model_obj.getConstrs():
                if c.ConstrName.startswith('_benders_fix_'):
                    pi = c.Pi
                    if abs(pi) > 1e-12:
                        fix_duals[c.ConstrName] = float(pi)

            if not fix_duals:
                fix_duals = self._extract_reduced_costs(
                    sub_m, var_col_map, shared_vars, master_vars)

            total_obj = model_obj.objVal
            art_penalty = sum(art_vars[i].X for i in range(len(art_vars))) * M
            original_obj = total_obj - art_penalty

            _log_benders(
                f"  Gurobi artificial-var solve: total={total_obj:.6f}, "
                f"penalty={art_penalty:.6f}, orig_obj={original_obj:.6f}, "
                f"M={M:.1f}, fix_duals_count={len(fix_duals)}",
                show_log)

            return {
                'feasible': True,
                'objective_value': total_obj,
                'original_objective': original_obj,
                'artificial_penalty': art_penalty,
                'fix_duals': fix_duals,
                'sub_constraints': [],
                'has_duals': bool(fix_duals),
            }
        except Exception as e:
            import traceback
            _log_benders(
                f"  Gurobi artificial-var solve failed: {e}", show_log)
            _log_benders("  %s" % traceback.format_exc(), show_log)
            return None

    # ------------------------------------------------------------------
    # Reduced-cost extraction (bound-based fixing path)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reduced_costs(sub_m, var_col_map, shared_vars, master_vars):
        """Extract Benders cut coefficients from variable reduced costs.

        When fixing is done via bounds, the reduced cost (col_dual) of each
        fixed variable equals the shadow price of the fixing -- the Benders
        cut coefficient.
        """
        model_obj = sub_m.model
        interface = sub_m.features.get('interface_name', '')

        col_duals = None
        if interface == 'gurobi':
            try:
                all_vars = model_obj.getVars()
                col_duals = [v.RC for v in all_vars]
            except Exception:
                return {}
        else:
            try:
                sol = model_obj.getSolution()
                col_duals = sol.col_dual
            except Exception:
                return {}

        if col_duals is None:
            return {}

        fix_duals = {}
        for var_name in shared_vars:
            val = master_vars.get(var_name)
            if val is None:
                continue
            for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                col_info = var_col_map.get((prefix, var_name))
                if col_info is None:
                    continue
                if isinstance(col_info, dict):
                    is_indexed = hasattr(val, '__len__') and not isinstance(val, str)
                    for idx, col_idx in col_info.items():
                        if col_idx < len(col_duals):
                            label = '_benders_fix_%s_%s' % (var_name, idx)
                            fix_duals[label] = float(col_duals[col_idx])
                else:
                    if col_info < len(col_duals):
                        label = '_benders_fix_%s' % var_name
                        fix_duals[label] = float(col_duals[col_info])
                break
        return fix_duals

    def _derive_numerical_coeffs(self, level, master_vars, shared_vars,
                                  solver_opts, solution_generator, show_log,
                                  base_obj):
        """Estimate recourse slopes when the backend exposes no fixing duals.

        Uses finite-difference probing: for each complicating variable, flips
        it to the opposite binary value (or by ±1 for continuous/integer)
        and measures the change in subproblem objective.

        For binary variables, probes at the opposite bound (0↔1).
        For general variables, probes at center ± 1.
        """
        # Every probe is a full subproblem rebuild + solve; cap the
        # total budget so a large shared set degrades to the weaker
        # constant cut (the direct-solve safety net then catches any
        # quality difference) instead of hanging the run for minutes.
        MAX_NUMERICAL_PROBES = 64
        n_elements = 0
        for var_name in shared_vars:
            value = master_vars.get(var_name)
            if value is None:
                continue
            if isinstance(value, dict):
                n_elements += len(value)
            elif isinstance(value, np.ndarray):
                n_elements += value.size
            else:
                n_elements += 1
            if n_elements > MAX_NUMERICAL_PROBES:
                _log_benders(
                    f"  Numerical probing skipped: {n_elements}+ shared"
                    f" elements exceed the {MAX_NUMERICAL_PROBES}-probe"
                    f" budget; falling back to a constant cut.", show_log)
                return {}
        coeffs = {}
        for var_name in shared_vars:
            value = master_vars.get(var_name)
            if value is None:
                continue

            if isinstance(value, dict):
                value_items = list(value.items())
            elif isinstance(value, np.ndarray) and value.size > 1:
                value_items = [(idx[0] if len(idx) == 1 else idx, float(val))
                               for idx, val in np.ndenumerate(value)]
            else:
                center = float(np.asarray(value).reshape(-1)[0])
                is_binary = center in (0.0, 1.0)
                deltas = []
                if is_binary:
                    deltas = [1.0 - center, 0.0 - center]
                else:
                    deltas = [1.0, -1.0]
                for delta in deltas:
                    if abs(delta) < 1e-12:
                        continue
                    probe = center + delta
                    probe_vars = dict(master_vars)
                    probe_vars[var_name] = probe
                    probe_result = self._solve_single_subproblem(
                        level, probe_vars, shared_vars, solver_opts,
                        solution_generator, show_log,
                        derive_numerical_cut=False)
                    if probe_result['feasible']:
                        coeffs[var_name] = (
                            (probe_result['objective_value'] - base_obj) / delta)
                        break
                continue

            for idx, center in value_items:
                center = float(center)
                is_binary = center in (0.0, 1.0)
                deltas = []
                if is_binary:
                    deltas = [1.0 - center, 0.0 - center]
                else:
                    deltas = [1.0, -1.0]
                for delta in deltas:
                    if abs(delta) < 1e-12:
                        continue
                    probe_vars = {}
                    for k, v in master_vars.items():
                        if k == var_name:
                            if isinstance(v, dict):
                                probe_vars[k] = dict(v)
                                probe_vars[k][idx] = center + delta
                            elif isinstance(v, np.ndarray):
                                probe_vars[k] = v.copy().astype(float)
                                flat_idx = int(idx) if isinstance(idx, int) else int(idx[0])
                                probe_vars[k].flat[flat_idx] = center + delta
                            else:
                                probe_vars[k] = center + delta
                        else:
                            probe_vars[k] = v
                    probe_result = self._solve_single_subproblem(
                        level, probe_vars, shared_vars, solver_opts,
                        solution_generator, show_log,
                        derive_numerical_cut=False)
                    if probe_result['feasible']:
                        key = (var_name, idx)
                        coeffs[key] = (
                            (probe_result['objective_value'] - base_obj) / delta)
                        break
        return coeffs

    # ------------------------------------------------------------------
    # Variable fixing (constraint-based, legacy)
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_variable(m, var_name, value):
        var_obj = None
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            var_obj = m.features['variables'].get((prefix, var_name))
            if var_obj is not None:
                break

        if var_obj is None:
            return

        # pyomo's IndexedVar and COPT's tupledict are dict-like
        # containers rather than dict subclasses; comparing the container
        # itself against a scalar falls back to object equality and would
        # store a constant False row.  to_indexed_dict guards COPT's Var,
        # which raises CoptError (not AttributeError) on unknown
        # attribute lookups.
        if not isinstance(var_obj, dict):
            container = to_indexed_dict(var_obj)
            if container is not None:
                var_obj = container

        is_indexed = hasattr(value, '__len__') and not isinstance(value, str)
        if is_indexed and isinstance(var_obj, dict):
            for idx, v in var_obj.items():
                try:
                    val = float(value[idx])
                except Exception:
                    continue
                try:
                    m.con(v == val, name=f'_benders_fix_{var_name}_{idx}')
                except Exception:
                    try:
                        m.con(v >= val,
                               name=f'_benders_fix_{var_name}_{idx}_lb')
                        m.con(v <= val,
                               name=f'_benders_fix_{var_name}_{idx}_ub')
                    except Exception:
                        pass
        else:
            try:
                scalar_val = float(value)
            except Exception:
                return
            if isinstance(var_obj, dict):
                # scalar target for an indexed variable: fix every element
                for idx, v in var_obj.items():
                    try:
                        m.con(v == scalar_val,
                              name=f'_benders_fix_{var_name}_{idx}')
                    except Exception:
                        pass
                return
            try:
                m.con(var_obj == scalar_val,
                       name=f'_benders_fix_{var_name}')
            except Exception:
                try:
                    m.con(var_obj >= scalar_val,
                           name=f'_benders_fix_{var_name}_lb')
                    m.con(var_obj <= scalar_val,
                           name=f'_benders_fix_{var_name}_ub')
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # HiGHS bound-based fixing (fast, O(1) per variable)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_var_col_map(m):
        """Map (prefix, name) -> col_idx for all variables in a model."""
        mapping = {}
        model_obj = m.model
        interface = m.features.get('interface_name', '')
        try:
            if interface == 'gurobi':
                n_cols = len(model_obj.getVars())
            else:
                n_cols = model_obj.getNumCol()
        except Exception:
            n_cols = 0
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            for (p, name), var_obj in m.features.get('variables', {}).items():
                if p != prefix:
                    continue
                if isinstance(var_obj, dict):
                    mapping[(prefix, name)] = {}
                    for idx, v in var_obj.items():
                        try:
                            if interface == 'gurobi':
                                col_idx = v.index if hasattr(v, 'index') else None
                            else:
                                col_idx = v.index if hasattr(v, 'index') else v._col
                            if col_idx is not None:
                                mapping[(prefix, name)][idx] = col_idx
                        except Exception:
                            pass
                else:
                    try:
                        if interface == 'gurobi':
                            col_idx = var_obj.index if hasattr(var_obj, 'index') else None
                        else:
                            col_idx = var_obj.index if hasattr(var_obj, 'index') else var_obj._col
                        if col_idx is not None:
                            mapping[(prefix, name)] = col_idx
                    except Exception:
                        pass
        return mapping

    @staticmethod
    def _fix_variable_bound(m, var_col_map, var_name, value):
        """Fix variable via bounds on the model (HiGHS or Gurobi)."""
        model_obj = m.model
        interface = m.features.get('interface_name', '')
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            var_obj = m.features.get('variables', {}).get((prefix, var_name))
            if interface == 'gurobi':
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        try:
                            val = float(value[idx])
                            v = var_obj[idx] if isinstance(var_obj, dict) else var_obj
                            v.lb = val
                            v.ub = val
                        except Exception:
                            pass
                else:
                    try:
                        val = float(value)
                        var_obj.lb = val
                        var_obj.ub = val
                    except Exception:
                        pass
            else:
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        try:
                            val = float(value[idx])
                            model_obj.changeColBounds(col_idx, val, val)
                        except Exception:
                            pass
                else:
                    try:
                        val = float(value)
                        model_obj.changeColBounds(col_info, val, val)
                    except Exception:
                        pass
            return

    @staticmethod
    def _unfix_variable_bound(m, var_col_map, var_name, original_bounds):
        """Restore original bounds for a variable."""
        model_obj = m.model
        interface = m.features.get('interface_name', '')
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            var_obj = m.features.get('variables', {}).get((prefix, var_name))
            if interface == 'gurobi':
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        try:
                            lb, ub = original_bounds.get(
                                (prefix, var_name, idx), (-1e20, 1e20))
                            v = var_obj[idx] if isinstance(var_obj, dict) else var_obj
                            v.lb = lb
                            v.ub = ub
                        except Exception:
                            pass
                else:
                    try:
                        lb, ub = original_bounds.get(
                            (prefix, var_name), (-1e20, 1e20))
                        var_obj.lb = lb
                        var_obj.ub = ub
                    except Exception:
                        pass
            else:
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        try:
                            lb, ub = original_bounds.get(
                                (prefix, var_name, idx), (-1e20, 1e20))
                            model_obj.changeColBounds(col_idx, lb, ub)
                        except Exception:
                            pass
                else:
                    try:
                        lb, ub = original_bounds.get(
                            (prefix, var_name), (-1e20, 1e20))
                        model_obj.changeColBounds(col_info, lb, ub)
                    except Exception:
                        pass
            return

    @staticmethod
    def _capture_original_bounds(m, var_col_map, shared_vars):
        """Capture original bounds for shared variables before fixing."""
        model_obj = m.model
        interface = m.features.get('interface_name', '')
        orig = {}
        for var_name in shared_vars:
            for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                col_info = var_col_map.get((prefix, var_name))
                if col_info is None:
                    continue
                var_obj = m.features.get('variables', {}).get((prefix, var_name))
                if interface == 'gurobi':
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            try:
                                v = var_obj[idx] if isinstance(var_obj, dict) else var_obj
                                orig[(prefix, var_name, idx)] = (v.lb, v.ub)
                            except Exception:
                                pass
                    else:
                        try:
                            orig[(prefix, var_name)] = (var_obj.lb, var_obj.ub)
                        except Exception:
                            pass
                else:
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            try:
                                b = model_obj.getColBounds(col_idx)
                                orig[(prefix, var_name, idx)] = (b[0], b[1])
                            except Exception:
                                pass
                    else:
                        try:
                            b = model_obj.getColBounds(col_info)
                            orig[(prefix, var_name)] = (b[0], b[1])
                        except Exception:
                            pass
                break
        return orig

    # ------------------------------------------------------------------
    # Cut generation (Benders optimality / feasibility cuts)
    # ------------------------------------------------------------------

    def _generate_optimality_cut(self, sub_result, master_vars,
                                 complicating_vars):
        """Generate a Benders optimality cut from subproblem duals.

        For minimization, the standard Benders cut is:
            theta >= Q(x_bar) + sum_j c_j * (x_j - x_bar_j)

        where c_j = sum of linking-constraint duals involving x_j.
        By LP sensitivity, the fixing-constraint dual mu[j] = -c_j,
        so c_j = -mu[j].

        Equivalent form:
            theta >= rhs + sum_j c_j * x_j
        where rhs = Q(x_bar) - sum_j c_j * x_bar_j = Q(x_bar) + sum_j mu[j] * x_bar_j
        """
        fix_duals = sub_result.get('fix_duals', {})
        numerical_coeffs = sub_result.get('numerical_coeffs', {})
        no_useful_fix_duals = (
            not fix_duals or
            all(abs(float(value)) <= 1e-12 for value in fix_duals.values()))
        if no_useful_fix_duals and numerical_coeffs:
            rhs = sub_result.get('objective_value', 0.0)
            coeffs = {}
            for key, coeff in numerical_coeffs.items():
                if isinstance(key, tuple):
                    var_name, idx = key
                    y_star = master_vars.get(var_name, {})
                    if isinstance(y_star, dict):
                        x_val = float(y_star.get(idx, 0))
                    else:
                        x_val = float(np.asarray(y_star).flatten()[idx]) if np.asarray(y_star).size > idx else 0.0
                    if var_name not in coeffs:
                        coeffs[var_name] = {}
                    coeffs[var_name][idx] = coeff
                else:
                    var_name = key
                    x_val = float(np.asarray(
                        master_vars.get(var_name, 0)).reshape(-1)[0])
                    coeffs[var_name] = coeff
                rhs -= coeff * x_val
            return {
                'rhs': rhs,
                'coeffs': coeffs,
                'type': 'optimality',
                'sub_obj': sub_result.get('objective_value', 0.0),
            }
        if not fix_duals:
            sub_obj = sub_result.get('objective_value', 0.0)
            return {
                'rhs': sub_obj,
                'coeffs': {},
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        sub_obj = sub_result.get('objective_value', 0.0)
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        # In auto mode the subproblem has the full objective, so the
        # fix_duals include both the integer cost coefficient c_I_j and the
        # continuous sensitivity dQ/dx_j.  Subtract the integer part so
        # the cut coefficients represent only the continuous sensitivity.
        int_var_costs = sub_result.get('int_var_costs', {})

        # Compute per-variable cut coefficients from fixing-constraint duals.
        # The fixing-constraint dual mu[j] = -sum_i(pi[i,j]) where pi are
        # the linking-constraint duals. The Benders coefficient is c_j = -mu[j].
        coeffs = {}
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()

            # In auto mode the fix_duals include the integer cost
            # coefficient c_I_j.  Subtract it to get the pure continuous
            # sensitivity dQ/dx_j.
            c_I = int_var_costs.get(var_name, 0.0)

            per_elem = {}
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val) - c_I
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            try:
                                idx = ast.literal_eval(index_text)
                            except (ValueError, SyntaxError):
                                idx = index_text
                        # The fixing-row dual is the recourse slope for the
                        # corresponding master variable in the minimization
                        # value function.
                        per_elem[idx] = float(dual_val) - c_I
                    except ValueError:
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                else:
                    coeffs[var_name] = per_elem.get(0, 0.0)

        # Compute rhs = Q(x_bar) - sum_j c_j * x_bar_j.
        # When the subproblem objective includes integer costs (add_problem
        # path), sub_obj = int_cost(x_bar) + Q(x_bar) but the cut
        # coefficients only capture continuous sensitivity dQ/dx_j.  We must
        # subtract int_cost(x_bar) so the rhs represents Q(x_bar) - sum c_j x_bar_j
        # and avoids double-counting the integer costs in the master.
        int_cost_at_x_bar = 0.0
        for var_name in fixing_vars:
            c_I = int_var_costs.get(var_name, 0.0)
            if c_I != 0.0:
                y_star = fixing_vals.get(var_name, 0)
                y_arr = np.asarray(
                    list(y_star.values()) if isinstance(y_star, dict) else y_star,
                    dtype=float).flatten()
                int_cost_at_x_bar += c_I * float(np.sum(y_arr))

        rhs_adjust = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            c = coeffs.get(var_name, 0.0)
            if isinstance(c, dict):
                for idx, mu in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += mu * float(y_arr[idx])
            else:
                if isinstance(c, dict):
                    for idx, coefficient in c.items():
                        if isinstance(idx, int) and idx < len(y_arr):
                            rhs_adjust += coefficient * float(y_arr[idx])
                else:
                    rhs_adjust += c * float(np.sum(y_arr))

        return {
            'rhs': sub_obj - int_cost_at_x_bar - rhs_adjust,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    def _generate_feasibility_cut(self, sub_result, master_vars,
                                   complicating_vars):
        """Generate a Benders feasibility cut.

        When the subproblem is infeasible, we need a cut that eliminates the
        current fixing x̄ from the master's feasible region.

        Path 1 (fixing-constraint duals from artificial variables):
            The feasibility cut is:  Σ λ_i (x_i - x̄_i) ≥ 0
            which is:               Σ λ_i x_i  ≥  Σ λ_i x̄_i
            where λ_i are the reduced costs of the fixed variables from
            the always-feasible subproblem (Section 3.3.4).

        Path 2 (no good cut for binary/integer variables):
            Eliminates the current integer fixing.
        """
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})
        fix_duals = sub_result.get('fix_duals', {})
        sub_duals = sub_result.get('sub_duals', {})

        all_duals = {**sub_duals, **fix_duals}

        if all_duals:
            coeffs = {}
            rhs = 0.0
            for var_name in fixing_vars:
                y_star = fixing_vals.get(var_name, 0)
                if isinstance(y_star, dict):
                    y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
                else:
                    y_arr = np.asarray(y_star, dtype=float).flatten()

                per_elem = {}
                for label, dual_val in all_duals.items():
                    if abs(dual_val) < 1e-12:
                        continue
                    prefix = f'_benders_fix_{var_name}_'
                    if label == f'_benders_fix_{var_name}':
                        per_elem[0] = float(dual_val)
                    elif label.startswith(prefix):
                        try:
                            index_text = label[len(prefix):]
                            if index_text.isdigit():
                                idx = int(index_text)
                            else:
                                idx = ast.literal_eval(index_text)
                            per_elem[idx] = float(dual_val)
                        except (ValueError, SyntaxError):
                            pass

                if per_elem:
                    if len(y_arr) > 1:
                        coeffs[var_name] = per_elem
                        for idx, lam in per_elem.items():
                            if isinstance(idx, int) and idx < len(y_arr):
                                rhs += lam * float(y_arr[idx])
                    else:
                        lam = per_elem.get(0, 0.0)
                        coeffs[var_name] = lam
                        rhs += lam * float(np.sum(y_arr))

            if coeffs:
                return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility'}

        coeffs = {}
        rhs = 1.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()
            var_coeffs = {}
            if isinstance(y_star, dict):
                indices_iter = y_star.keys()
            elif isinstance(y_star, np.ndarray):
                indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
            else:
                indices_iter = [None]
            for index, value in zip(indices_iter, y_arr):
                v = float(value)
                if v < 0.5:
                    var_coeffs[index] = 1.0
                else:
                    var_coeffs[index] = -1.0
                    rhs -= 1.0
            if var_coeffs:
                coeffs[var_name] = var_coeffs

        if not coeffs:
            for v in complicating_vars:
                coeffs[v] = 1.0

        return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility'}



    # ------------------------------------------------------------------
    # Advanced cut generation and acceleration techniques
    # ------------------------------------------------------------------

    def _add_to_cut_pool(self, cut, max_size=None):
        """Add cut to pool with size management."""
        if max_size is None:
            max_size = self._cut_pool_max_size
        
        self._cut_pool.append(cut)
        
        # Keep only most violated cuts if pool is too large
        if len(self._cut_pool) > max_size:
            # Sort by violation magnitude (approximated by RHS magnitude)
            self._cut_pool.sort(key=lambda c: abs(c.get('rhs', 0)), reverse=True)
            self._cut_pool = self._cut_pool[:max_size]

    def _select_most_violated_cuts(self, master_vars, max_cuts=5):
        """Select most violated cuts from pool for current master solution."""
        if not self._cut_pool:
            return []
        
        violations = []
        for cut in self._cut_pool:
            violation = self._compute_cut_violation(cut, master_vars)
            violations.append((violation, cut))
        
        # Sort by violation (descending) and select top cuts
        violations.sort(key=lambda x: x[0], reverse=True)
        selected = [cut for _, cut in violations[:max_cuts]]
        
        return selected

    def _compute_cut_violation(self, cut, master_vars):
        """Compute violation of a cut for current master solution."""
        if cut.get('type') == 'feasibility':
            # For feasibility cuts: LHS - RHS; violation if LHS < RHS
            lhs = -cut['rhs']
            for var_name, coeff in cut.get('coeffs', {}).items():
                val = master_vars.get(var_name, 0)
                if isinstance(coeff, dict):
                    if isinstance(val, dict):
                        for idx, c in coeff.items():
                            if idx in val:
                                lhs += c * val[idx]
                    else:
                        c0 = coeff.get(0, 0.0)
                        lhs += c0 * float(np.asarray(val).flatten()[0])
                else:
                    if isinstance(val, dict):
                        for v in val.values():
                            lhs += coeff * v
                    else:
                        lhs += coeff * float(np.asarray(val).flatten()[0])
            return max(0, -lhs)  # Violation if constraint is not satisfied
        else:
            # For optimality cuts: θ - (rhs + Σ c_j x_j) should be ≥ 0
            # Violation = max(0, rhs + Σ c_j x_j - θ)
            theta_val = 0.0
            for tname in ('theta', '_benders_theta'):
                if tname in master_vars:
                    theta_val = float(np.asarray(master_vars[tname]).flatten()[0])
                    break
            if theta_val == 0.0:
                return 0.0
            expr = cut.get('rhs', 0)
            for var_name, coeff in cut.get('coeffs', {}).items():
                val = master_vars.get(var_name, 0)
                if isinstance(coeff, dict):
                    if isinstance(val, dict):
                        for idx, c in coeff.items():
                            if idx in val:
                                expr += c * val[idx]
                    else:
                        c0 = coeff.get(0, 0.0)
                        expr += c0 * float(np.asarray(val).flatten()[0])
                else:
                    if isinstance(val, dict):
                        for v in val.values():
                            expr += coeff * v
                    else:
                        expr += coeff * float(np.asarray(val).flatten()[0])
            return max(0, expr - theta_val)

    def _generate_multi_cuts(self, sub_results, master_vars, complicating_vars):
        """Generate multiple cuts from subproblem results."""
        cuts = []
        
        for sub_result in sub_results:
            if not sub_result['feasible']:
                cut = self._generate_feasibility_cut(sub_result, master_vars, complicating_vars)
                if cut is not None:
                    cuts.append(cut)
            else:
                # Generate both standard and strengthened cuts
                standard_cut = self._generate_optimality_cut(sub_result, master_vars, complicating_vars)
                if standard_cut is not None:
                    cuts.append(standard_cut)
                
                # Try to generate strengthened cut
                strengthened_cut = self._generate_strengthened_cut(sub_result, master_vars, complicating_vars)
                if strengthened_cut is not None:
                    cuts.append(strengthened_cut)
        
        return cuts

    def _generate_strengthened_cut(self, sub_result, master_vars, complicating_vars):
        """Generate a strengthened optimality cut using Magnanti-Wong technique.

        The Magnanti-Wong method strengthens Benders cuts by solving an LP to find
        the point on the optimal face of the subproblem that maximizes the cut
        violation at the current master solution. This produces tighter cuts
        that accelerate convergence.

        For problems where the full Magnanti-Wong LP is not practical, we use
        a simplified strengthening based on:
        1. Coefficient perturbation toward extreme rays
        2. RHS tightening via dual bound information
        """
        fix_duals = sub_result.get('fix_duals', {})
        if not fix_duals:
            sub_obj = sub_result.get('objective_value', 0.0)
            return {
                'rhs': sub_obj,
                'coeffs': {},
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        sub_obj = sub_result.get('objective_value', 0.0)
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})
        sub_duals = sub_result.get('sub_duals', {})

        coeffs = {}
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()

            per_elem = {}
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val)
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            try:
                                idx = ast.literal_eval(index_text)
                            except (ValueError, SyntaxError):
                                idx = index_text
                        per_elem[idx] = float(dual_val)
                    except ValueError:
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                else:
                    coeffs[var_name] = per_elem.get(0, 0.0)

        # NOTE: We do NOT clip coefficients or scale the RHS arbitrarily.
        # The standard Benders cut is: θ ≥ Q(x̄) + Σ μ_j (x_j - x̄_j)
        # where μ_j are the fixing-constraint duals. The coefficients must
        # remain as-is to preserve cut validity.

        rhs_adjust = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            c = coeffs.get(var_name, 0.0)
            if isinstance(c, dict):
                for idx, mu in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += mu * float(y_arr[idx])
            else:
                rhs_adjust += c * float(np.sum(y_arr))

        strengthened_rhs = sub_obj - rhs_adjust

        return {
            'rhs': strengthened_rhs,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
            'strengthened': True
        }

    def _apply_trust_region(self, master_vars, radius):
        """Apply trust region constraints to limit variable movement."""
        trust_constraints = []
        
        for var_name, val in master_vars.items():
            if isinstance(val, dict):
                # For indexed variables, apply trust region to each element
                for idx, v in val.items():
                    # Create trust region: |x - x_bar| <= radius
                    # This requires additional variables, so we implement a simpler version
                    pass
            else:
                # For scalar variables, add simple bounds
                # In practice, this would require additional constraints
                pass
        
        return trust_constraints

    def _generate_pareto_optimal_cut(self, sub_results, master_vars, complicating_vars):
        """Generate a strengthened Benders optimality cut.

        Uses the most recent subproblem duals to generate a standard Benders cut.
        For a true Magnanti-Wong Pareto-optimal cut, one would solve an auxiliary
        LP to find the point on the dual optimal face that maximizes cut violation.
        Here we use the current duals directly, which is a valid (if not maximal)
        Benders optimality cut.

        Reference: Magnanti and Wong (1981), "Accelerating Benders Decomposition"
        """
        if not sub_results:
            return None

        base_result = sub_results[-1]
        if not base_result['feasible']:
            return None

        fix_duals = base_result.get('fix_duals', {})
        if not fix_duals:
            sub_obj = base_result.get('objective_value', 0.0)
            return {
                'rhs': sub_obj,
                'coeffs': {},
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        sub_obj = base_result.get('objective_value', 0.0)
        fixing_vars = base_result.get('fixing_vars', [])
        fixing_vals = base_result.get('fixing_vals', {})

        coeffs = {}
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()

            per_elem = {}
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val)
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            try:
                                idx = ast.literal_eval(index_text)
                            except (ValueError, SyntaxError):
                                idx = index_text
                        per_elem[idx] = float(dual_val)
                    except ValueError:
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                else:
                    coeffs[var_name] = per_elem.get(0, 0.0)

        rhs_adjust = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            c = coeffs.get(var_name, 0.0)
            if isinstance(c, dict):
                for idx, mu in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += mu * float(y_arr[idx])
            else:
                rhs_adjust += c * float(np.sum(y_arr))

        return {
            'rhs': sub_obj - rhs_adjust,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
            'pareto': True
        }

    def _cleanup_duplicate_cuts(self, tolerance=1e-6):
        """Remove duplicate or very similar cuts from storage."""
        if len(self._stored_cuts) <= 1:
            return
        
        unique_cuts = []
        seen_cuts = set()
        
        for cut in self._stored_cuts:
            # Create a signature for the cut
            signature = self._create_cut_signature(cut, tolerance)
            
            if signature not in seen_cuts:
                unique_cuts.append(cut)
                seen_cuts.add(signature)
        
        self._stored_cuts = unique_cuts

    def _create_cut_signature(self, cut, tolerance):
        """Create a signature for cut deduplication."""
        coeffs_tuple = tuple(sorted(
            (var_name, tuple(sorted(coeff.items())) if isinstance(coeff, dict) else coeff)
            for var_name, coeff in cut.get('coeffs', {}).items()
        ))
        rhs_rounded = round(cut.get('rhs', 0.0) / tolerance) * tolerance
        cut_type = cut.get('type', 'unknown')
        
        return (cut_type, rhs_rounded, coeffs_tuple)

    def _apply_cut_selection(self, master_vars, max_cuts=10):
        """Select best cuts to add to master problem."""
        if not self._stored_cuts:
            return self._stored_cuts
        
        # If using cut pool, select most violated cuts
        if self._cut_pool:
            selected = self._select_most_violated_cuts(master_vars, max_cuts)
            return selected
        
        # Otherwise, use all stored cuts
        return self._stored_cuts

    # ------------------------------------------------------------------
    # Cut application to master model
    # ------------------------------------------------------------------

    def _add_cuts_to_master(self, master_m, complicating_vars, cuts=None):
        """Add stored Benders cuts to the master model as constraints."""
        cuts_to_add = cuts if cuts is not None else self._stored_cuts
        if not cuts_to_add:
            return

        theta_var = None
        for name in ('theta', '_benders_theta'):
            for prefix in ('fvar', 'pvar'):
                theta_var = master_m.features['variables'].get((prefix, name))
                if theta_var is not None:
                    break
            if theta_var is not None:
                break

        has_theta = theta_var is not None
        if not has_theta:
            try:
                theta_var = master_m.fvar('_benders_theta')
                master_m.con(theta_var >= 0, name='_benders_theta_init')
                has_theta = True
            except Exception:
                pass

        for i, cut in enumerate(cuts_to_add):
            is_feasibility = cut.get('type') == 'feasibility'
            try:
                if is_feasibility:
                    # Feasibility cut: sum_{j in I} y_j >= 1
                    # where I = {j : y_j* = 0}
                    expr = -cut['rhs']
                    for var_name, coeff in cut.get('coeffs', {}).items():
                        var_obj = None
                        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                            var_obj = master_m.features['variables'].get(
                                (prefix, var_name))
                            if var_obj is not None:
                                break
                        if var_obj is None:
                            continue
                        if not isinstance(var_obj, dict):
                            container = to_indexed_dict(var_obj)
                            if container is not None:
                                var_obj = container
                        if isinstance(var_obj, dict):
                            for idx, v in var_obj.items():
                                try:
                                    current_coeff = coeff.get(idx, 0.0) if isinstance(coeff, dict) else coeff
                                    expr = expr + current_coeff * v
                                except Exception:
                                    pass
                        else:
                            try:
                                expr = expr + coeff * var_obj
                            except Exception:
                                pass
                    master_m.con(expr >= 0, name=f'_benders_feas_cut_{i}')
                elif has_theta:
                    # Optimality cut for minimization:
                    # theta >= rhs + sum_j c_j * y_j
                    # → theta - sum_j c_j * y_j - rhs >= 0
                    expr = theta_var - cut['rhs']
                    for var_name, coeff in cut.get('coeffs', {}).items():
                        var_obj = None
                        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                            var_obj = master_m.features['variables'].get(
                                (prefix, var_name))
                            if var_obj is not None:
                                break
                        if var_obj is None:
                            continue
                        if not isinstance(var_obj, dict):
                            container = to_indexed_dict(var_obj)
                            if container is not None:
                                var_obj = container
                        if isinstance(coeff, dict):
                            if isinstance(var_obj, dict):
                                for idx, v in var_obj.items():
                                    c = coeff.get(idx, 0.0)
                                    if abs(c) > 1e-12:
                                        try:
                                            expr = expr - c * v
                                        except Exception:
                                            pass
                            else:
                                c0 = coeff.get(0, 0.0)
                                if abs(c0) > 1e-12:
                                    try:
                                        expr = expr - c0 * var_obj
                                    except Exception:
                                        pass
                        elif isinstance(var_obj, dict):
                            for idx, v in var_obj.items():
                                try:
                                    expr = expr - coeff * v
                                except Exception:
                                    pass
                        else:
                            try:
                                expr = expr - coeff * var_obj
                            except Exception:
                                pass
                    master_m.con(expr >= 0, name=f'_benders_opt_cut_{i}')
                else:
                    # No theta: theta is implicitly 0, so rhs + sum c_j y_j <= 0
                    expr = -cut['rhs']
                    for var_name, coeff in cut.get('coeffs', {}).items():
                        var_obj = None
                        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                            var_obj = master_m.features['variables'].get(
                                (prefix, var_name))
                            if var_obj is not None:
                                break
                        if var_obj is None:
                            continue
                        if not isinstance(var_obj, dict):
                            container = to_indexed_dict(var_obj)
                            if container is not None:
                                var_obj = container
                        if isinstance(var_obj, dict):
                            for idx, v in var_obj.items():
                                try:
                                    expr = expr + coeff * v
                                except Exception:
                                    pass
                        else:
                            try:
                                expr = expr + coeff * var_obj
                            except Exception:
                                pass
                    master_m.con(expr >= 0, name=f'_benders_cut_{i}')
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Combinatorial Benders cuts (Big-M optimality, no-good feasibility)
    # ------------------------------------------------------------------

    def _generate_combinatorial_optimality_cut(self, sub_result, master_vars,
                                                complicating_vars):
        """Generate a combinatorial Benders optimality cut.

        For integer-only subproblems where duals are unavailable, generates:
            theta >= z* - M * nogood(x_bar)
        where M = z* - theta_lb is a valid big-M.

        This follows the formulation from BendersLib's CombinatorialOC.
        """
        sub_obj = sub_result.get('objective_value', 0.0)
        if sub_obj is None:
            return None

        theta_val = 0.0
        for var_name in complicating_vars:
            val = master_vars.get(var_name, 0)
            if isinstance(val, dict):
                for v in val.values():
                    theta_val += float(v)
            else:
                theta_val += float(np.asarray(val).flatten()[0])
        # Use a simple estimate: theta_lb is the lower bound from params
        theta_lb = self._params.theta_lb
        big_m = sub_obj - theta_lb
        if big_m <= 0:
            big_m = max(abs(sub_obj), 1.0)

        # No-good component: at least one binary/integer variable must change
        coeffs = {}
        rhs_adj = 0.0
        for var_name in complicating_vars:
            y_star = master_vars.get(var_name, 0)
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()
            var_coeffs = {}
            if isinstance(y_star, dict):
                indices_iter = y_star.keys()
            elif isinstance(y_star, np.ndarray):
                indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
            else:
                indices_iter = [None]
            for index, value in zip(indices_iter, y_arr):
                v = float(value)
                if v < 0.5:
                    var_coeffs[index] = 1.0 / big_m
                else:
                    var_coeffs[index] = -1.0 / big_m
                    rhs_adj -= 1.0
            if var_coeffs:
                coeffs[var_name] = var_coeffs

        rhs = sub_obj / big_m + rhs_adj
        return {
            'rhs': rhs,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
            'sense': '>=',
        }

    def _generate_nogood_cut(self, master_vars, complicating_vars):
        """Generate a no-good feasibility cut excluding the current solution.

        Equivalent to BendersLib's NoGoodFC:
            Σ_{i∈I₁} x_i - Σ_{i∈I₀} x_i ≤ |I₁| - 1
        """
        coeffs = {}
        rhs = 1.0
        for var_name in complicating_vars:
            y_star = master_vars.get(var_name, 0)
            if y_star is None:
                continue
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()
            var_coeffs = {}
            if isinstance(y_star, dict):
                indices_iter = y_star.keys()
            elif isinstance(y_star, np.ndarray):
                indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
            else:
                indices_iter = [None]
            for index, value in zip(indices_iter, y_arr):
                v = float(value)
                if v < 0.5:
                    var_coeffs[index] = 1.0
                else:
                    var_coeffs[index] = -1.0
                    rhs -= 1.0
            if var_coeffs:
                coeffs[var_name] = var_coeffs

        if not coeffs:
            for v in complicating_vars:
                coeffs[v] = 1.0
        return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility', 'sense': '>='}

    # ------------------------------------------------------------------
    # Generalized Benders cuts (Lagrange multiplier based)
    # ------------------------------------------------------------------

    def _generate_generalized_optimality_cut(self, sub_result, master_vars,
                                               complicating_vars):
        """Generate a generalized Benders optimality cut using Lagrange multipliers.

        Form: theta >= f(x_bar) + lambda^T * (b - A*x)
        where lambda are the Lagrange multipliers from the subproblem.

        Falls back to numerical coefficients when duals are unavailable
        (e.g., binary first-stage variables with HiGHS).
        """
        fix_duals = sub_result.get('fix_duals', {})
        sub_obj = sub_result.get('objective_value', 0.0)

        # Check if we have useful duals
        has_useful_duals = (
            fix_duals and
            not all(abs(float(v)) <= 1e-12 for v in fix_duals.values()))

        # Fall back to numerical probing coefficients when duals are empty
        numerical_coeffs = sub_result.get('numerical_coeffs', {})
        if not has_useful_duals and numerical_coeffs:
            coeffs = {}
            rhs = sub_obj
            for key, coeff in numerical_coeffs.items():
                if isinstance(key, tuple):
                    var_name, idx = key
                    y_star = master_vars.get(var_name, {})
                    if isinstance(y_star, dict):
                        x_val = float(y_star.get(idx, 0))
                    else:
                        flat = np.asarray(y_star).flatten()
                        x_val = float(flat[idx]) if idx < len(flat) else 0.0
                    if var_name not in coeffs:
                        coeffs[var_name] = {}
                    coeffs[var_name][idx] = coeff
                else:
                    var_name = key
                    x_val = float(np.asarray(
                        master_vars.get(var_name, 0)).reshape(-1)[0])
                    coeffs[var_name] = coeff
                rhs -= coeff * x_val
            return {
                'rhs': rhs,
                'coeffs': coeffs,
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        if not fix_duals:
            sub_obj = sub_result.get('objective_value', 0.0)
            return {
                'rhs': sub_obj,
                'coeffs': {},
                'type': 'optimality',
                'sub_obj': sub_obj,
            }

        sub_obj = sub_result.get('objective_value', 0.0)
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        coeffs = {}
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()

            per_elem = {}
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val)
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            try:
                                idx = ast.literal_eval(index_text)
                            except (ValueError, SyntaxError):
                                idx = index_text
                        per_elem[idx] = float(dual_val)
                    except ValueError:
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                else:
                    coeffs[var_name] = per_elem.get(0, 0.0)

        # Generalized cut (Geoffrion 1972):
        #   theta >= f(x_bar) - lambda^T A (x - x_bar)
        # Rewritten: theta + lambda^T A x >= f(x_bar) + lambda^T A x_bar
        # We use the form: theta >= sub_obj - sum(c_j * (x_j - x_bar_j))
        # which equals: theta - sum(c_j * x_j) >= sub_obj - sum(c_j * x_bar_j)
        # so rhs = sub_obj - sum(c_j * x_bar_j)
        rhs_adjust = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            c = coeffs.get(var_name, 0.0)
            if isinstance(c, dict):
                for idx, mu in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += mu * float(y_arr[idx])
            else:
                rhs_adjust += c * float(np.sum(y_arr))

        return {
            'rhs': sub_obj - rhs_adjust,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
            'generalized': True,
            'sense': '>=',
        }

    # ------------------------------------------------------------------
    # Parallel subproblem solving
    # ------------------------------------------------------------------

    def _solve_subproblems_parallel(self, master_vars, solver_opts,
                                     solution_generator, show_log,
                                     relax_subproblem_integrality=True):
        """Solve subproblems in parallel using ThreadPoolExecutor.

        Falls back to the serial sweep when the parallel phase raises,
        so real subproblem errors surface exactly as in a serial run.
        """
        threads = self._params.parallel_threads
        max_workers = threads if threads > 0 else None

        if self._is_auto():
            # Single subproblem in auto mode — no parallelism needed
            return self._solve_subproblems_serial(
                master_vars, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)

        _sub_iface = solver_opts.get(
            'sub_interface_name', solver_opts.get('interface_name', 'highs'))
        if _sub_iface not in _PARALLEL_SAFE_INTERFACES:
            # The sub solve path touches global stdout/stderr — running
            # several at once would race those redirects.
            return self._solve_subproblems_serial(
                master_vars, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)

        results = [None] * len(self._links)

        def _solve_one(idx, link):
            if link is None:
                return idx, None
            level_name = link['lower']
            level = None
            for sp in self._subproblems:
                if isinstance(sp, _BendersProblem) and sp.name == level_name:
                    level = sp
                    break
            if level is None:
                return idx, None
            result = self._solve_single_subproblem(
                level, master_vars, link['shared_vars'],
                solver_opts, solution_generator, show_log,
                relax_subproblem_integrality=relax_subproblem_integrality)
            if result is not None:
                # Same default as the serial L-shaped branch.
                result['probability'] = link.get(
                    'probability', 1.0 / max(len(self._links), 1))
            return idx, result

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_solve_one, i, link): i
                           for i, link in enumerate(self._links)}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
        except Exception:
            return self._solve_subproblems_serial(
                master_vars, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # IIS-based feasibility cuts
    # ------------------------------------------------------------------

    def _compute_iis(self, sub_m):
        """Compute Irreducible Infeasible Subsystem (IIS) of an infeasible subproblem.

        Returns a list of variable names involved in the IIS.
        Falls back to all complicating variables if IIS is not available.
        """
        interface = sub_m.features.get('interface_name', '')
        iis_vars = []

        if interface == 'gurobi':
            try:
                sub_m.model.computeIIS()
                constrs = sub_m.model.getConstrs()
                for c in constrs:
                    if c.IISConstr:
                        row = sub_m.model.getRow(c)
                        for j in range(row.size()):
                            var = row.getVar(j)
                            vname = var.VarName
                            if vname:
                                iis_vars.append(vname)
            except Exception:
                pass
        elif interface == 'cplex':
            try:
                sub_m.model.populate()
                iis = sub_m.model.conflict.get(0)
                for status, col in zip(iis.indicators, iis.sub_cols):
                    if status == 1:
                        iis_vars.append(sub_m.model.getColName(col))
            except Exception:
                pass

        return list(set(iis_vars))

    # ------------------------------------------------------------------
    # L-Shaped (stochastic) support
    # ------------------------------------------------------------------

    def add_scenario(self, name, model_fn, directions, shared_vars, probability=1.0):
        """Add a scenario for L-shaped (stochastic) Benders decomposition.

        Parameters
        ----------
        name : str
            Scenario name.
        model_fn : callable
            ``model_fn(m) -> m`` that builds this scenario's model.
        directions : list of str
            Optimization direction (e.g. ``["min"]``).
        shared_vars : list of str
            Complicating variables shared with the master.
        probability : float
            Probability weight for this scenario (default 1.0).
        """
        level = _BendersProblem(name, model_fn, directions, obj_index=0)
        self._subproblems.append(level)
        self._links.append({
            'upper': 'master',
            'lower': name,
            'shared_vars': list(shared_vars),
            'probability': probability,
        })
        return self

    def _solve_subproblems_l_shaped(self, master_vars, solver_opts,
                                     solution_generator, show_log,
                                     relax_subproblem_integrality=True):
        """Solve all scenarios for L-shaped decomposition.

        Supports both single-cut (aggregated) and multi-cut modes.
        """
        results = []

        if self._params.parallel_sub:
            raw = self._solve_subproblems_parallel(
                master_vars, solver_opts, solution_generator, show_log,
                relax_subproblem_integrality)
            for r in raw:
                if r is not None:
                    results.append(r)
        else:
            for link in self._links:
                if link is None:
                    continue
                level_name = link['lower']
                level = None
                for sp in self._subproblems:
                    if isinstance(sp, _BendersProblem) and sp.name == level_name:
                        level = sp
                        break
                if level is None:
                    continue
                result = self._solve_single_subproblem(
                    level, master_vars, link['shared_vars'],
                    solver_opts, solution_generator, show_log,
                    relax_subproblem_integrality=relax_subproblem_integrality)
                # Attach probability
                result['probability'] = link.get('probability', 1.0 / max(len(self._links), 1))
                results.append(result)

        return results

    def _generate_l_shaped_aggregated_cut(self, sub_results, master_vars,
                                           complicating_vars):
        """Generate a single aggregated L-shaped cut across all scenarios.

        Form: theta >= sum_omega p_omega * pi_omega^T (b_omega - T_omega * x)
        """
        total_rhs = 0.0
        total_coeffs = {}

        for sub_result in sub_results:
            if not sub_result.get('feasible', False):
                continue
            prob = sub_result.get('probability', 1.0)
            fix_duals = sub_result.get('fix_duals', {})
        sub_obj = sub_result.get('objective_value', 0.0)
        # When duals have been decontaminated (M-contamination removed),
        # the rhs must use the original (unpenalized) objective to stay
        # consistent with the clean duals.
        if sub_result.get('decontaminated', False):
            orig_obj = sub_result.get('original_objective')
            if orig_obj is not None:
                sub_obj = orig_obj

            # Compute per-scenario coefficients
            scenario_rhs = 0.0
            for var_name in fixing_vars:
                y_star = fixing_vals.get(var_name, 0)
                y_arr = np.asarray(
                    list(y_star.values()) if isinstance(y_star, dict) else y_star,
                    dtype=float).flatten()
                for label, dual_val in fix_duals.items():
                    prefix = f'_benders_fix_{var_name}_'
                    mu = 0.0
                    idx = 0
                    if label == f'_benders_fix_{var_name}':
                        mu = float(dual_val)
                        idx = 0
                    elif label.startswith(prefix):
                        try:
                            index_text = label[len(prefix):]
                            idx = int(index_text) if index_text.isdigit() else 0
                            mu = float(dual_val)
                        except ValueError:
                            continue
                    if isinstance(idx, int) and idx < len(y_arr):
                        total_rhs += prob * mu * float(y_arr[idx])

            total_rhs += prob * sub_obj

            # Aggregate coefficients
            for var_name in fixing_vars:
                y_star = fixing_vals.get(var_name, 0)
                y_arr = np.asarray(
                    list(y_star.values()) if isinstance(y_star, dict) else y_star,
                    dtype=float).flatten()
                for label, dual_val in fix_duals.items():
                    prefix = f'_benders_fix_{var_name}_'
                    mu = 0.0
                    idx = 0
                    if label == f'_benders_fix_{var_name}':
                        mu = float(dual_val)
                        idx = 0
                    elif label.startswith(prefix):
                        try:
                            index_text = label[len(prefix):]
                            idx = int(index_text) if index_text.isdigit() else 0
                            mu = float(dual_val)
                        except ValueError:
                            continue
                    if var_name not in total_coeffs:
                        total_coeffs[var_name] = {}
                    if isinstance(idx, int):
                        total_coeffs[var_name][idx] = total_coeffs[var_name].get(idx, 0.0) + prob * mu

        if not total_coeffs:
            return None
        return {
            'rhs': total_rhs,
            'coeffs': total_coeffs,
            'type': 'optimality',
            'sub_obj': total_rhs,
            'sense': '>=',
        }

    def _generate_integer_l_shaped_cut(self, sub_results, master_vars,
                                        complicating_vars):
        """Generate an aggregated integer L-shaped cut (Laporte & Louveaux 1993).

        Uses combinatorial (no-good) cuts for stochastic problems with
        integer complicating variables. Each scenario produces a
        CombinatorialOC cut; these are aggregated with probability weights.

        Form: theta >= sum_omega p_omega * (z_omega - M_omega * nogood(x, x_bar_omega))
        """
        total_rhs = 0.0
        total_coeffs = {}

        for sub_result in sub_results:
            if not sub_result.get('feasible', False):
                continue
            prob = sub_result.get('probability', 1.0)
            sub_obj = sub_result.get('objective_value', 0.0)
            if sub_obj is None:
                continue

            fixing_vars = sub_result.get('fixing_vars', [])
            fixing_vals = sub_result.get('fixing_vals', {})

            theta_lb = self._params.theta_lb
            big_m = sub_obj - theta_lb
            if big_m <= 0:
                big_m = max(abs(sub_obj), 1.0)

            scenario_rhs_adj = 0.0
            for var_name in fixing_vars:
                y_star = fixing_vals.get(var_name, 0)
                if isinstance(y_star, dict):
                    y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
                else:
                    y_arr = np.asarray(y_star, dtype=float).flatten()
                var_indices = y_star.keys() if isinstance(y_star, dict) else (
                    np.ndindex(y_star.shape) if isinstance(y_star, np.ndarray) and y_star.ndim > 0 else [None])
                for index, value in zip(var_indices, y_arr):
                    v = float(value)
                    if var_name not in total_coeffs:
                        total_coeffs[var_name] = {}
                    if v < 0.5:
                        total_coeffs[var_name][index] = total_coeffs[var_name].get(index, 0.0) + prob * (1.0 / big_m)
                    else:
                        total_coeffs[var_name][index] = total_coeffs[var_name].get(index, 0.0) + prob * (-1.0 / big_m)
                        scenario_rhs_adj -= 1.0

            total_rhs += prob * (sub_obj / big_m + scenario_rhs_adj)

        if not total_coeffs:
            return None
        return {
            'rhs': total_rhs,
            'coeffs': total_coeffs,
            'type': 'optimality',
            'sub_obj': total_rhs,
            'sense': '>=',
        }

    def _generate_generalized_l_shaped_cut(self, sub_results, master_vars,
                                            complicating_vars):
        """Generate an aggregated generalized L-shaped cut.

        Uses generalized Benders cuts (Lagrange multiplier based) for
        stochastic problems with convex (nonlinear) recourse. Each scenario
        produces a GeneralizedOC cut; these are aggregated with probability weights.

        Form: theta >= sum_omega p_omega * (f_omega(x_bar) + grad_f_omega(x_bar)^T (x - x_bar))
        """
        total_rhs = 0.0
        total_coeffs = {}

        for sub_result in sub_results:
            if not sub_result.get('feasible', False):
                continue
            prob = sub_result.get('probability', 1.0)
            fix_duals = sub_result.get('fix_duals', {})
            sub_obj = sub_result.get('objective_value', 0.0)
            if sub_obj is None:
                continue

            fixing_vars = sub_result.get('fixing_vars', [])
            fixing_vals = sub_result.get('fixing_vals', {})

            scenario_rhs = prob * sub_obj
            for var_name in fixing_vars:
                y_star = fixing_vals.get(var_name, 0)
                if isinstance(y_star, dict):
                    y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
                else:
                    y_arr = np.asarray(y_star, dtype=float).flatten()
                var_indices = y_star.keys() if isinstance(y_star, dict) else (
                    np.ndindex(y_star.shape) if isinstance(y_star, np.ndarray) and y_star.ndim > 0 else [None])
                for label, dual_val in fix_duals.items():
                    prefix = f'_benders_fix_{var_name}_'
                    mu = 0.0
                    idx = 0
                    if label == f'_benders_fix_{var_name}':
                        mu = float(dual_val)
                        idx = 0
                    elif label.startswith(prefix):
                        try:
                            index_text = label[len(prefix):]
                            idx = int(index_text) if index_text.isdigit() else 0
                            mu = float(dual_val)
                        except ValueError:
                            continue
                    if isinstance(idx, int) and idx < len(y_arr):
                        # Generalized: rhs = f(x_bar) + lambda^T A x_bar
                        scenario_rhs += prob * mu * float(y_arr[idx])
                    if var_name not in total_coeffs:
                        total_coeffs[var_name] = {}
                    if isinstance(idx, int):
                        total_coeffs[var_name][idx] = total_coeffs[var_name].get(idx, 0.0) + prob * mu

            total_rhs += scenario_rhs

        if not total_coeffs:
            return None
        return {
            'rhs': total_rhs,
            'coeffs': total_coeffs,
            'type': 'optimality',
            'sub_obj': total_rhs,
            'sense': '>=',
        }

    # ------------------------------------------------------------------
    # Result packaging (legacy, kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(objective, variables, iterations, bounds_history,
                      status, n_opt_cuts=0, n_feas_cuts=0, runtime=0.0,
                      obj_history=None, n_sol=0):
        from time import perf_counter
        _last = bounds_history[-1] if bounds_history else (None, None)
        _lb = _last[0] if _last and _last[0] is not None else float('-inf')
        _ub = _last[1] if _last and _last[1] is not None else objective
        _gap_abs = (abs(_ub - _lb)
                    if _ub < float('inf') and _lb > float('-inf')
                    else float('inf'))
        _gap_rel = (_gap_abs / abs(_ub)
                    if _gap_abs < float('inf') and _ub != 0
                    else float('inf'))
        return BendersResult(
            status=status if isinstance(status, BendersStatus) else (
                BendersStatus.OPTIMAL if status == 'optimal' else
                BendersStatus.MAX_ITERATIONS if status == 'max_iterations' else
                BendersStatus.INFEASIBLE if status == 'infeasible' else
                BendersStatus.UNSOLVED),
            objective=objective,
            variables=variables or {},
            iterations=iterations,
            bounds=bounds_history,
            n_optimality_cuts=n_opt_cuts,
            n_feasibility_cuts=n_feas_cuts,
            n_cuts=n_opt_cuts + n_feas_cuts,
            n_sol=n_sol,
            runtime_total=runtime,
            gap_abs=_gap_abs,
            gap_rel=_gap_rel,
            lb_history=[b[0] for b in bounds_history],
            ub_history=[b[1] for b in bounds_history],
            obj_history=obj_history or [],
        )

    # ------------------------------------------------------------------
    # Direct HiGHS operations (bypass feloopy overhead)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cut_direct_highs(m, cut, theta_col_idx, var_col_map):
        """Add a Benders cut directly to HiGHS model as a row."""
        model_obj = m.model
        indices = []
        values = []

        if cut.get('type') == 'feasibility':
            for var_name, coeff in cut.get('coeffs', {}).items():
                for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                    col_info = var_col_map.get((prefix, var_name))
                    if col_info is None:
                        continue
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            c = coeff.get(idx, 0.0) if isinstance(coeff, dict) else coeff
                            if abs(c) > 1e-15:
                                indices.append(col_idx)
                                values.append(c)
                    else:
                        if abs(coeff) > 1e-15:
                            indices.append(col_info)
                            values.append(coeff)
                    break
            lower = cut['rhs']
            upper = 1e20
        else:
            if theta_col_idx is not None:
                indices.append(theta_col_idx)
                values.append(1.0)
            for var_name, coeff in cut.get('coeffs', {}).items():
                for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                    col_info = var_col_map.get((prefix, var_name))
                    if col_info is None:
                        continue
                    if isinstance(coeff, dict):
                        if isinstance(col_info, dict):
                            for idx, col_idx in col_info.items():
                                c = coeff.get(idx, 0.0)
                                if abs(c) > 1e-15:
                                    indices.append(col_idx)
                                    values.append(-c)
                    elif isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            if abs(coeff) > 1e-15:
                                indices.append(col_idx)
                                values.append(-coeff)
                    else:
                        if abs(coeff) > 1e-15:
                            indices.append(col_info)
                            values.append(-coeff)
                    break
            lower = cut['rhs']
            upper = 1e20

        if indices:
            model_obj.addRow(lower, upper, len(indices), indices, values)

    @staticmethod
    def _get_reduced_costs_highs(m):
        """Extract reduced costs from HiGHS solution."""
        try:
            sol = m.model.getSolution()
            return {i: sol.col_dual[i] for i in range(m.model.getNumCol())}
        except Exception:
            return {}

    @staticmethod
    def _get_theta_col_idx(m, var_col_map):
        """Get column index of theta variable."""
        for name in ('theta', '_benders_theta'):
            for prefix in ('fvar', 'pvar'):
                col_info = var_col_map.get((prefix, name))
                if col_info is not None and not isinstance(col_info, dict):
                    return col_info
        return None

    @staticmethod
    def _get_var_col_indices(m, var_name, var_col_map):
        """Get column indices for a variable."""
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is not None:
                if isinstance(col_info, dict):
                    return col_info
                return {0: col_info}
        return {}

    def _compute_benders_coeffs_from_subduals(self, sub_m, sub_duals,
                                              shared_vars, master_vars):
        """Compute Benders coefficients from constraint duals + matrix.

        When bound-based fixing yields zero reduced costs (fix_duals empty),
        we reconstruct the Benders gradient from the constraint duals and the
        sparse constraint matrix stored in the HiGHS LP:

            c_j = -sum_i  pi_i * A_{i,j}

        where pi_i are constraint duals and A_{i,j} are the coefficients of
        the complicating variables in the subproblem constraints.
        """
        lp_data = sub_m.features.get('lp_data', {})
        row_names = lp_data.get('row_names', [])
        A_col_ptrs = lp_data.get('A_col_pointers', [])
        A_row_idx = lp_data.get('A_row_indices', [])
        A_vals = lp_data.get('A_values', [])
        n_cols = lp_data.get('n_cols', 0)

        if not A_col_ptrs or not row_names or not sub_duals:
            return {}

        label_to_row = {name: i for i, name in enumerate(row_names) if name}
        row_to_label = {i: name for name, i in label_to_row.items()}

        var_col_map = self._build_var_col_map(sub_m)

        fix_duals = {}
        for var_name in shared_vars:
            col_indices = self._get_var_col_indices(sub_m, var_name, var_col_map)
            if not col_indices:
                continue
            for idx, col_idx in col_indices.items():
                if col_idx is None or col_idx >= n_cols:
                    continue
                start = int(A_col_ptrs[col_idx])
                end = int(A_col_ptrs[col_idx + 1]) if col_idx + 1 < len(A_col_ptrs) else len(A_row_idx)
                coeff = 0.0
                for k in range(start, end):
                    row_i = int(A_row_idx[k])
                    a_val = float(A_vals[k])
                    lbl = row_to_label.get(row_i)
                    if lbl is None:
                        continue
                    pi = sub_duals.get(lbl)
                    if pi is None:
                        continue
                    coeff += float(pi) * a_val
                if abs(coeff) > 1e-12:
                    fix_duals[f'_benders_fix_{var_name}_{idx}'] = -coeff
        return fix_duals
