# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Automatic Benders decomposition orchestration.

Analyses model structure, selects the best Benders variant, identifies
complicating variables, and configures acceleration/cut strategy.
Called from ``feloopy.py`` when ``boost='benders'``.
"""

from ..base.logging import make_logger
from .enums import BendersStatus
from .result import BendersResult


# ------------------------------------------------------------------
# Solver pair utility
# ------------------------------------------------------------------

def split_solver_pair(value, default='highs'):
    """Split interface/solver into (master, sub) pair.

    Supports:
      - 'highs' → ('highs', 'highs')
      - ['gurobi', 'cplex'] → ('gurobi', 'cplex')
      - ('highs',) → ('highs', 'highs')
    """
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return value[0], value[1]
        if len(value) == 1:
            return value[0], value[0]
    return value, value


# ------------------------------------------------------------------
# Variant selection
# ------------------------------------------------------------------

def select_variant(captured, integer_vars, continuous_vars,
                   n_constraints, options, all_var_kinds):
    """Auto-detect the best Benders variant for the problem.

    Decision tree:
    1. All integer/binary, no continuous → 'logic' (if callback) or direct
    2. Linear subproblem with integer complicating → 'classical'
    3. Nonlinear subproblem separable by integer vars → 'generalized'
    4. Pure integer with no meaningful LP relaxation → 'combinatorial'
    5. Multiple scenarios / stochastic structure → 'l_shaped'

    Returns the variant name string.
    """
    n_integer = len(integer_vars)
    n_continuous = len(continuous_vars)

    has_nonlinear = False
    for constraint, _ in captured.constraints:
        if hasattr(constraint, 'expression'):
            expr = constraint.expression
            if hasattr(expr, 'terms'):
                for key in expr.terms:
                    if isinstance(key, tuple) and len(key) > 3:
                        has_nonlinear = True
                        break

    method_override = options.get('benders_method', None)
    if method_override in ('classical', 'combinatorial', 'generalized',
                            'logic', 'l_shaped'):
        return method_override

    if has_nonlinear:
        return 'generalized'

    if n_integer == 0 and n_continuous > 0:
        return 'classical'

    all_binary = all(
        kind == 'bvar' for (kind, _) in all_var_kinds
        if kind in ('ivar', 'bvar'))
    if all_binary and n_integer > 0:
        return 'generalized'

    if options.get('logic_cut_callback') is not None:
        return 'logic'

    return 'classical'


# ------------------------------------------------------------------
# Complicating variable selection
# ------------------------------------------------------------------

def select_complicating_variables(captured, integer_vars, continuous_vars,
                                  benders_method, options, all_var_kinds):
    """Select the best complicating variables for Benders decomposition.

    Core principle: after fixing the complicating variables, the
    subproblem must be a *nontrivial* optimisation problem with
    meaningful dual information.

    Scoring per candidate variable *v*:
      linking_score   = number of constraints where *v* appears
                        alongside at least one other variable
      objective_bonus = 1 if *v* is NOT the only objective variable
      type_bonus      = 10 if *v* is integer/binary, 0 if continuous

      score = linking_score + 5 * objective_bonus + type_bonus

    Returns a list of variable names.
    """
    all_vars = integer_vars + continuous_vars
    n_total = len(all_vars)
    n_integer = len(integer_vars)

    user_vars = options.get('complicating_variables', None)
    if user_vars:
        return user_vars

    if n_total == 0:
        return []

    constr_int_vars = {}
    constr_cont_vars = {}
    var_constr_count = {}
    var_in_obj = set()

    for ci, constraint in enumerate(
            captured.constraints if hasattr(captured, 'constraints') else []):
        cobj = constraint[0] if isinstance(constraint, tuple) else constraint
        int_in_constr = set()
        cont_in_constr = set()

        if hasattr(cobj, 'expression') and hasattr(cobj.expression, 'terms'):
            for key in cobj.expression.terms:
                if isinstance(key, tuple) and len(key) >= 2:
                    kind, name = key[0], key[1]
                    if kind in ('ivar', 'bvar'):
                        int_in_constr.add(name)
                    elif kind in ('fvar', 'pvar', 'rvar'):
                        cont_in_constr.add(name)

        constr_int_vars[ci] = int_in_constr
        constr_cont_vars[ci] = cont_in_constr

        for name in int_in_constr | cont_in_constr:
            var_constr_count[name] = var_constr_count.get(name, 0) + 1

    if hasattr(captured, 'objectives') and captured.objectives:
        obj_expr = captured.objectives[0][0]
        if hasattr(obj_expr, 'terms'):
            for key in obj_expr.terms:
                if isinstance(key, tuple) and len(key) >= 2:
                    var_in_obj.add(key[1])

    def _score_variable(name):
        linking = 0
        obj_coupling = 0
        n_constrs = len(captured.constraints) if hasattr(captured, 'constraints') else 0
        for ci in range(n_constrs):
            all_in = constr_int_vars.get(ci, set()) | \
                     constr_cont_vars.get(ci, set())
            if name in all_in:
                others = all_in - {name}
                if others:
                    linking += 1
                if others & var_in_obj:
                    obj_coupling += 1
        type_bonus = 10 if name in integer_vars else 0
        return linking + 5 * obj_coupling + type_bonus

    scored = [(name, _score_variable(name)) for name in all_vars]
    scored.sort(key=lambda x: -x[1])

    if benders_method == 'combinatorial':
        return _select_combinatorial(
            integer_vars, constr_int_vars, constr_cont_vars,
            var_constr_count)

    if benders_method == 'logic':
        return list(integer_vars) if integer_vars else list(continuous_vars)

    if benders_method == 'classical':
        if n_integer == 0:
            max_vars = n_total - 1
            selected = [name for name, _ in scored if _ > 0][:max_vars]
            if not selected and scored:
                selected = [scored[0][0]]
            return selected
        else:
            int_scored = [(n, s) for n, s in scored if n in integer_vars]
            # For small/medium MILPs use all integers —
            # subset selection leaves linking constraints like 2*y+z>=i
            # with fixed small y (0) infeasible beyond z cap (5), causing
            # artificial-var fallback LB 84 UB 0 and objective 0.0.
            if n_integer <= 20:
                return [name for name, _ in int_scored]
            max_vars = min(n_integer, max(1, n_total // 3))
            selected = [name for name, _ in int_scored[:max_vars]]
            if not selected:
                selected = [scored[0][0]] if scored else []
            return selected

    if benders_method == 'generalized':
        max_vars = min(n_total - 1, max(3, n_total // 2))
        int_scored = [(n, s) for n, s in scored if n in integer_vars]
        cont_scored = [(n, s) for n, s in scored if n in continuous_vars]
        selected = [name for name, _ in int_scored[:max_vars]]
        remaining = max_vars - len(selected)
        if remaining > 0:
            selected += [name for name, _ in cont_scored[:remaining]]
        return selected if selected else all_vars[:min(3, n_total)]

    return all_vars[:min(3, n_total)]


def _select_combinatorial(integer_vars, constr_int_vars,
                           constr_cont_vars, var_constr_count):
    """Greedy set-cover selection for combinatorial Benders."""
    critical = [
        ci for ci in constr_int_vars
        if constr_int_vars.get(ci) and constr_cont_vars.get(ci)
    ]
    if not critical:
        ranked = sorted(integer_vars,
                        key=lambda v: -var_constr_count.get(v, 0))
        return ranked[:min(5, len(integer_vars))]

    uncovered = set(critical)
    selected = []
    var_coverage = {}
    for name in integer_vars:
        covers = {ci for ci in critical
                  if name in constr_int_vars.get(ci, set())}
        var_coverage[name] = covers

    for _ in range(min(len(integer_vars), 20)):
        if not uncovered:
            break
        best_var = max(
            integer_vars,
            key=lambda v: len(var_coverage.get(v, set()) & uncovered))
        if not (var_coverage.get(best_var, set()) & uncovered):
            break
        selected.append(best_var)
        uncovered -= var_coverage.get(best_var, set())

    return selected if selected else integer_vars[:min(5, len(integer_vars))]


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_automatic_benders(search, show_log):
    """Solve a linear mixed-integer model through automatic Benders.

    This is the main orchestration function called from ``feloopy.py``
    when ``boost='benders'``.  It:

    1. Validates the problem is a linear MILP.
    2. Captures the model (linearized or direct).
    3. Selects the best Benders variant.
    4. Identifies optimal complicating variables.
    5. Builds master/subproblem factories.
    6. Configures acceleration, cut strategy, theta bounds.
    7. Solves and returns a ``BendersResult``.

    Parameters
    ----------
    search : _Search
        The search object from ``feloopy.py``.
    show_log : bool
        Whether to print progress.
    """
    log = make_logger('Benders', show_log)

    if (search.method not in ('exact', 'uncertain')
            or not search.directions
            or len(search.directions) != 1):
        raise ValueError("boost='benders' requires one exact MILP objective")

    decomposed_type = search.em.features.get('al_problem_type')
    if decomposed_type is None:
        decomposed_type = search.em.features.get('problem_type')
    if decomposed_type not in (None, 'MILP', 'LP', 'IP'):
        raise ValueError("boost='benders' supports linear MILP models only")

    all_var_kinds = list(search.em.features.get('variables', {}))
    integer_vars = [name for (kind, name) in all_var_kinds
                    if kind in ('ivar', 'bvar')]
    continuous_vars = [name for (kind, name) in all_var_kinds
                       if kind in ('fvar', 'pvar', 'rvar')]

    if not integer_vars and not continuous_vars:
        raise ValueError("boost='benders' requires at least one variable")

    if not continuous_vars:
        log("All variables are integer/binary — "
            "using exact MILP fallback (no LP subproblem possible)")
        search._benders_result = BendersResult(status=BendersStatus.DIRECT)
        return

    from ..automatic_capture import _AutomaticCaptureModel, _materialize_automatic_model
    from .core import BendersDecomposition
    from .result import BendersParams

    direction = search.directions[0]
    sign = -1 if direction == 'max' else 1

    # --- Capture the model ---
    if search.auto_linearize:
        captured = search._build_linearized_capture()
        obj_expr = captured.objectives[0][0] if captured.objectives else None
        if obj_expr is not None:
            has_int_obj = any(k[0] in ('ivar', 'bvar')
                              for k in obj_expr.terms)
            if not has_int_obj:
                search._benders_result = {
                    'objective': None, 'variables': {}, 'iterations': 0,
                    'bounds': [], 'status': 'logic_based_direct',
                    'mode': 'logic_based_direct',
                }
                return
    else:
        captured = _AutomaticCaptureModel()
        try:
            search.environment(captured, *search.args, **search.kwargs)
        except (TypeError, ValueError) as error:
            from ....helpers.error import ConstantConstraintError
            if isinstance(error, ConstantConstraintError):
                raise  # infeasible model, not an unsupported expression
            raise ValueError(
                "boost='benders' supports linear MILP expressions "
                "and constraints") from error

    n_integer = len(integer_vars)
    n_constraints = len(search.em.features.get('constraints', []))

    # --- Step 1: Select variant ---
    benders_method = search.options.get('benders_method', None)
    if benders_method is None:
        benders_method = select_variant(
            captured, integer_vars, continuous_vars,
            n_constraints, search.options, all_var_kinds)
    log(f"Auto-selected variant: {benders_method}")

    # --- Step 2: Select complicating variables ---
    complicating_vars = search.options.get('complicating_variables', None)
    if complicating_vars is None:
        complicating_vars = select_complicating_variables(
            captured, integer_vars, continuous_vars,
            benders_method, search.options, all_var_kinds)
    # The automatic master drops every constraint containing a
    # continuous variable (see _materialize_automatic_model), so only
    # first-stage (integer) vars may be pinned in the subproblem:
    # fixing a continuous var there too would leave those constraints
    # enforced nowhere, the fixed sub then reports infeasible for
    # every master solution, UB never moves off inf and the master
    # eventually dies.  Continuous complicating vars stay free in the
    # subproblem, which is where their constraints live.
    _first_stage = set(integer_vars)
    _shared_fix = [name for name in complicating_vars
                   if name in _first_stage]
    if _shared_fix:
        complicating_vars = _shared_fix
    log(f"Complicating variables: {complicating_vars}")

    # --- Step 3: Build master / subproblem factories ---
    def build_subproblem(m):
        _materialize_automatic_model(
            captured, m, objective_scope='subproblem')
        if sign == -1:
            objective = m.features['objectives'][0]
            m.features['objectives'][0] = -objective
        return m

    def build_master(m):
        _materialize_automatic_model(
            captured, m, objective_scope='master')
        if n_integer >= 50:
            theta_lower, theta_upper = -1e12, 1e12
        elif n_integer >= 20:
            theta_lower, theta_upper = -1e9, 1e9
        elif n_integer >= 5:
            theta_lower, theta_upper = -1e6, 1e6
        else:
            theta_lower, theta_upper = -1e4, 1e4
        theta = m.fvar('_benders_theta', bound=[theta_lower, theta_upper])
        objective = m.features['objectives'][0]
        m.features['objectives'][0] = sign * objective + theta
        return m

    master_if, sub_if = split_solver_pair(search.interface, 'highs')
    master_sv, sub_sv = split_solver_pair(search.solver, 'highs')

    decomposition = BendersDecomposition()
    decomposition.add_level('master', build_master, ['min'])
    decomposition.add_level('subproblem', build_subproblem, ['min'])
    decomposition.add_link('master', 'subproblem',
                           shared_vars=complicating_vars)

    # --- Step 4: Select acceleration and cut strategy ---
    acceleration_method = search.options.get('acceleration', None)
    if acceleration_method is None:
        if n_integer >= 30 and n_constraints >= 20:
            acceleration_method = 'hybrid'
        elif n_integer >= 15:
            acceleration_method = 'pareto'
        elif n_integer >= 8:
            acceleration_method = 'multi_cut'
        else:
            acceleration_method = 'pareto'

    cut_strategy = search.options.get('cut_strategy', None)
    if cut_strategy is None:
        if n_integer >= 15:
            cut_strategy = 'multi'
        else:
            cut_strategy = 'single'

    # --- Step 5: Configure params ---
    params = BendersParams(
        tol_abs=search.options.get('tolerance', 1e-6),
        tol_rel=search.options.get('relative_gap', 1e-4),
        time_limit=search.time_limit or float('inf'),
        iter_limit=search.options.get('max_iterations', 100),
    )
    decomposition.set_params(
        tol_abs=params.tol_abs,
        tol_rel=params.tol_rel,
        time_limit=params.time_limit,
        iter_limit=params.iter_limit,
    )

    # --- Step 6: Solve ---
    _benders_opts = {
        'max_iterations', 'tolerance', 'cut_strategy', 'acceleration',
        'benders_method', 'logic_cut_callback',
        'relax_subproblem_integrality', 'complicating_variables',
        'linking_constraints', 'cg_stabilization',
    }
    solver_opts = {k: v for k, v in search.options.items()
                   if k not in _benders_opts}

    search._benders_result = decomposition.solve(
        interface=master_if, solver=master_sv,
        sub_interface=sub_if, sub_solver=sub_sv,
        max_iterations=search.options.get('max_iterations', 100),
        tolerance=search.options.get('tolerance', 1e-6),
        cut_strategy=cut_strategy,
        acceleration=acceleration_method,
        show_log=bool(show_log),
        method=benders_method,
        logic_cut_callback=search.options.get('logic_cut_callback', None),
        relax_subproblem_integrality=search.options.get(
            'relax_subproblem_integrality', None),
        solver_log=False, options=solver_opts,
        time_limit=search.time_limit, cpu_threads=search.cpu_threads,
        absolute_gap=search.absolute_gap, relative_gap=search.relative_gap,
    )
