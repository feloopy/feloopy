# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import math
import pulp as pulp_interface
import timeit

from ..pulp_compat import PULP_V4, problem_variables


_pulp_solver_map = {
    'cbc': getattr(pulp_interface, 'PULP_CBC_CMD', None) or pulp_interface.COIN_CMD,
    'choco': pulp_interface.CHOCO_CMD,
    'coin': pulp_interface.COIN_CMD,
    'coinmp-dll': pulp_interface.COINMP_DLL,
    'coinmp_dll': pulp_interface.COINMP_DLL,
    'cplex-py': pulp_interface.CPLEX_PY,
    'cplex_py': pulp_interface.CPLEX_PY,
    'cplex': pulp_interface.CPLEX_CMD,
    'glpk': pulp_interface.GLPK_CMD,
    'gurobi-cmd': pulp_interface.GUROBI_CMD,
    'gurobi_cmd': pulp_interface.GUROBI_CMD,
    'gurobi': pulp_interface.GUROBI,
    'highs': pulp_interface.HiGHS_CMD,
    'mipcl': pulp_interface.MIPCL_CMD,
    'mosek': pulp_interface.MOSEK,
    'pyglpk': pulp_interface.PYGLPK,
    'scip': pulp_interface.SCIP_CMD,
    'xpress-py': pulp_interface.XPRESS_PY,
    'xpress_py': pulp_interface.XPRESS_PY,
    'xpress': pulp_interface.XPRESS,
    'cuopt': pulp_interface.CUOPT,
}


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    solver_options = dict(features['solver_options'])

    if time_limit is not None:
        solver_options['timeLimit'] = time_limit
    if thread_count is not None:
        solver_options['threads'] = thread_count
    if relative_gap is not None:
        solver_options['gapRel'] = relative_gap
    if absolute_gap is not None:
        solver_options['gapAbs'] = absolute_gap
    solver_options['msg'] = 1 if log else 0

    if solver_name not in _pulp_solver_map:
        raise RuntimeError(
            "Using solver '%s' is not supported by 'pulp'! \n"
            "Possible fixes: \n1) Check the solver name. \n"
            "2) Use another interface. \n" % (solver_name))

    # PuLP resolves executables on PATH itself, so make feloopy's
    # provisioned solvers visible — and fetch a free binary on first
    # explicit use (FELOOPY_NO_AUTO_DOWNLOAD=1 opts out).  Names that
    # are Python bindings (gurobipy, cplex, ...) only get the lookup.
    if solver_name == 'cbc' and getattr(pulp_interface, 'PULP_CBC_CMD', None):
        pass  # PuLP ships its own CBC — nothing to install
    else:
        from ...helpers.solver_executables import (
            ensure_solver_on_path,
            inject_solver_paths,
            missing_solver_message,
        )
        from ...helpers.solver_provisioning import ensure_solver, is_provisionable
        target = 'cbc' if solver_name in ('coin', 'coin_cmd') else solver_name
        executable = ensure_solver(target)
        if executable:
            ensure_solver_on_path(executable)
        elif is_provisionable(target):
            raise RuntimeError(missing_solver_message(target))
        else:
            inject_solver_paths((target,))

    if PULP_V4:
        return _solve_v4(features, model_object, model_objectives,
                         model_constraints, directions, constraint_labels,
                         solver_name, objective_id, solver_options)

    model_object.constraints.clear()

    match directions[objective_id]:
        case "min":
            model_object += model_objectives[objective_id]
        case "max":
            model_object += -model_objectives[objective_id]

    for counter, constraint in enumerate(model_constraints):
        model_object += (constraint, constraint_labels[counter])

    pulp_solver = _pulp_solver_map[solver_name](**solver_options)

    time_solve_begin = timeit.default_timer()
    result = model_object.solve(solver=pulp_solver)
    time_solve_end = timeit.default_timer()

    return [result, [time_solve_begin, time_solve_end]]


def _solve_v4(features, model_object, model_objectives, model_constraints,
              directions, constraint_labels, solver_name, objective_id,
              solver_options):
    """Rebuild the model on a fresh problem and solve it (PuLP 4).

    PuLP 4 cannot remove a constraint from a problem: there is no
    ``constraints.clear()`` equivalent, and re-adding a label raises
    ``PulpError: Repeated constraint names`` on solve.  The in-place
    clear-and-re-add used on 3.x therefore has no spelling here, so each
    solve materializes the current features (variables, constraints,
    objective) on a new problem and solves that.  The primal values are
    copied back onto the stored variable wrappers that feloopy readers
    hold, and the solved problem is published as
    ``model_object_before_solve`` so status/objective/duals/slacks are
    read from the problem that actually solved.
    """

    problem = pulp_interface.LpProblem(model_object.name, model_object.sense)

    var_map = {}
    for var in problem_variables(model_object):
        # PuLP 4 spells an open bound as None on the way in, but its
        # LpVariable properties report the stored inf back out, and
        # add_variable rejects explicit inf.  Translate non-finite
        # bounds back to None before re-adding.
        low = var.lowBound
        up = var.upBound
        new_var = problem.add_variable(
            var.name,
            lowBound=low if low is not None and math.isfinite(low) else None,
            upBound=up if up is not None and math.isfinite(up) else None,
            cat=var.cat)
        var_map[var.name] = new_var
        # carry warm-start values across the rebuild (fixValue() pins
        # lowBound/upBound and is therefore already carried above)
        try:
            if var.varValue is not None:
                new_var.setInitialValue(var.varValue)
        except Exception:
            pass

    for counter, constraint in enumerate(model_constraints):
        label = constraint_labels[counter]
        # PuLP 4's rust core reserves leading '_' names (auto-generated
        # constraints are _C1, _C2, ...); feloopy's multi-objective
        # scaffolding uses labels like '_payoff_0_0'.  Rename them the
        # way PuLP's own MPS reader does (see lp_problem.addConstraint's
        # MPS path: imp_ prefix); get_constraint mirrors the rename when
        # reading duals/slacks back.
        if isinstance(label, str) and label.startswith("_"):
            label = "imp_" + label
        problem += (_rebind_constraint(constraint, var_map), label)

    match directions[objective_id]:
        case "min":
            problem += _rebind_expression(
                model_objectives[objective_id], var_map)
        case "max":
            problem += -_rebind_expression(
                model_objectives[objective_id], var_map)

    pulp_solver = _pulp_solver_map[solver_name](**solver_options)

    time_solve_begin = timeit.default_timer()
    result = problem.solve(solver=pulp_solver)
    time_solve_end = timeit.default_timer()

    _copy_values_back(features, var_map)
    features['model_object_before_solve'] = problem

    return result


def _rebind_expression(expression, var_map):
    """Rebuild a stored expression on the fresh problem's variables."""

    if not hasattr(expression, 'items'):
        return expression  # constant objective: nothing to rebind

    terms = {}
    for var, coeff in expression.items():
        source = var_map.get(var.name)
        if source is None:
            raise RuntimeError(
                "PuLP 4 could not rebuild variable '%s' while re-solving: "
                "it is not part of the model problem." % var.name)
        terms[source] = coeff

    return pulp_interface.LpAffineExpression.from_dict(
        terms, constant=getattr(expression, 'constant', 0.0))


def _rebind_constraint(constraint, var_map):
    """Rebuild a stored constraint on the fresh problem's variables."""

    expression = _rebind_expression(constraint, var_map)

    if constraint.sense == pulp_interface.LpConstraintLE:
        return expression <= 0
    if constraint.sense == pulp_interface.LpConstraintGE:
        return expression >= 0
    return expression == 0


def _copy_values_back(features, var_map):
    """Write solved values back onto the wrappers feloopy readers hold.

    PuLP 4 wrappers belong to the problem that created them, so the
    rebuild above yields new wrapper objects; the originals stored in
    ``features['variables']`` (and handed back to user code) would keep
    reading ``None`` without this copy-back.
    """

    def _wrappers(stored):
        if isinstance(stored, dict):
            for item in stored.values():
                yield from _wrappers(item)
        elif hasattr(stored, 'name'):
            yield stored

    for stored in features.get('variables', {}).values():
        for wrapper in _wrappers(stored):
            source = var_map.get(wrapper.name)
            if source is None:
                continue
            try:
                wrapper.varValue = source.varValue
            except Exception:
                pass
            try:
                wrapper.dj = source.dj
            except Exception:
                pass
