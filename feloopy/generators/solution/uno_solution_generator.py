# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
import numpy as np

from ..variable.uno_variable_generator import (
    UnoVar, UnoExpr, evaluate_expr, compute_gradient,
    expr_to_gradient_func
)

uno_solver_selector = {'uno': 'uno'}

INF = float('inf')
UNO_BIG = 1e20
STRICT_EPS = 1e-6


def _parse_constraint(constraint):
    """Parse a constraint expression and return (lower_bound, upper_bound, function_expr).

    Uno expects constraints in the form: c_lb <= g(x) <= c_ub.
    """
    if isinstance(constraint, tuple) and len(constraint) == 3:
        sense = constraint[0]
        lhs = constraint[1]
        rhs = constraint[2]

        if isinstance(rhs, (int, float)):
            rhs_val = float(rhs)
        else:
            rhs_val = None

        if isinstance(lhs, (int, float)):
            lhs_val = float(lhs)
        else:
            lhs_val = None

        if rhs_val is not None and lhs_val is not None:
            return None, None, None

        if sense == '<=':
            if rhs_val is not None:
                return None, rhs_val, lhs
            elif lhs_val is not None:
                # lhs <= rhs with constant lhs means rhs >= lhs
                return lhs_val, None, rhs
            return None, 0.0, lhs - rhs

        elif sense == '>=':
            if lhs_val is not None:
                # lhs >= rhs with constant lhs means rhs <= lhs
                return None, lhs_val, rhs
            elif rhs_val is not None:
                return rhs_val, None, lhs
            return 0.0, None, lhs - rhs

        elif sense == '==':
            if rhs_val is not None:
                return rhs_val, rhs_val, lhs
            elif lhs_val is not None:
                return lhs_val, lhs_val, rhs
            return 0.0, 0.0, lhs - rhs

        elif sense == '<':
            if rhs_val is not None:
                return None, rhs_val - STRICT_EPS, lhs
            elif lhs_val is not None:
                return lhs_val + STRICT_EPS, None, rhs
            return None, -STRICT_EPS, lhs - rhs

        elif sense == '>':
            if lhs_val is not None:
                return None, lhs_val - STRICT_EPS, rhs
            elif rhs_val is not None:
                return rhs_val + STRICT_EPS, None, lhs
            return STRICT_EPS, None, lhs - rhs

    elif isinstance(constraint, (UnoExpr, UnoVar)):
        return 0.0, 0.0, constraint

    return None, None, None


def _build_constraint_callback(constraint_expr):
    """Build a callback function for a constraint expression."""
    def callback(x, constraint_values):
        constraint_values[0] = evaluate_expr(constraint_expr, np.array(x, dtype=float))
    return callback


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    solver_options = features['solver_options']

    if solver_name not in uno_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'uno'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    try:
        import unopy
    except ImportError:
        raise ImportError(
            "The 'unopy' package is required for Uno interface. "
            "Install it with: pip install unopy"
        )

    n = model_object.number_variables
    n_constraints = len(model_constraints)

    # Variable bounds so finite-difference gradients never probe outside the
    # feasible domain (e.g. negative base for a fractional power -> NaN).
    var_bounds = [
        (-INF if model_object.variables_lower_bounds[i] is None
         else float(model_object.variables_lower_bounds[i]),
         INF if model_object.variables_upper_bounds[i] is None
         else float(model_object.variables_upper_bounds[i]))
        for i in range(n)
    ]

    objective_expr = model_objectives[objective_id]
    if isinstance(objective_expr, (int, float)):
        obj_val = float(objective_expr)
        obj_func = lambda x: obj_val
        obj_grad = lambda x, g: None
    else:
        obj_func = lambda x: evaluate_expr(objective_expr, np.array(x, dtype=float))
        obj_grad = expr_to_gradient_func(objective_expr, n, bounds=var_bounds)

    constraint_exprs = []
    constraint_lower = []
    constraint_upper = []
    constraint_funcs = []

    for constraint in model_constraints:
        lb, ub, expr = _parse_constraint(constraint)
        if expr is not None:
            constraint_exprs.append(expr)
            constraint_funcs.append(_build_constraint_callback(expr))

            if lb is not None:
                constraint_lower.append(float(lb))
            else:
                constraint_lower.append(-UNO_BIG)

            if ub is not None:
                constraint_upper.append(float(ub))
            else:
                constraint_upper.append(UNO_BIG)

    n_cons = len(constraint_exprs)

    if n_cons > 0:
        jacobian_nnz = n_cons * n
        jacobian_row_indices = []
        jacobian_column_indices = []
        for i in range(n_cons):
            for j in range(n):
                jacobian_row_indices.append(i)
                jacobian_column_indices.append(j)

        def constraints_callback(x, constraint_values):
            for i, func in enumerate(constraint_funcs):
                func(x, constraint_values[i:i+1])

        def jacobian_callback(x, jacobian_values):
            x_arr = np.array(x, dtype=float)
            for i, expr in enumerate(constraint_exprs):
                offset = i * n
                for j in range(n):
                    jacobian_values[offset + j] = compute_gradient(
                        expr, x_arr, j, bounds=var_bounds[j])
    else:
        jacobian_nnz = 0
        jacobian_row_indices = []
        jacobian_column_indices = []
        constraints_callback = None
        jacobian_callback = None

    uno_problem_type = unopy.PROBLEM_NONLINEAR
    uno_base_indexing = unopy.ZERO_BASED_INDEXING

    uno_model = unopy.Model(uno_problem_type, n, uno_base_indexing)
    uno_model.set_variables_lower_bounds(model_object.variables_lower_bounds[:n])
    uno_model.set_variables_upper_bounds(model_object.variables_upper_bounds[:n])

    match directions[objective_id]:
        case 'min':
            uno_optimization_sense = unopy.MINIMIZE
        case 'max':
            uno_optimization_sense = unopy.MAXIMIZE

    uno_model.set_objective(uno_optimization_sense, obj_func, obj_grad)

    if n_cons > 0:
        uno_model.set_constraints(
            n_cons,
            constraints_callback,
            constraint_lower,
            constraint_upper,
            jacobian_nnz,
            jacobian_row_indices,
            jacobian_column_indices,
            jacobian_callback
        )

    x0 = np.zeros(n)
    for i in range(n):
        lb = model_object.variables_lower_bounds[i]
        ub = model_object.variables_upper_bounds[i]
        lb_big = abs(lb) >= UNO_BIG
        ub_big = abs(ub) >= UNO_BIG
        if lb_big and ub_big:
            x0[i] = 0.0
        elif lb_big:
            x0[i] = min(ub, 0.0) if ub > 0 else ub
        elif ub_big:
            x0[i] = max(lb, 0.0) if lb < 0 else lb
        else:
            x0[i] = (lb + ub) / 2.0

    # overlay the user supplied warm start on top of the default point
    for i, value in (features.get('_uno_start') or {}).items():
        if 0 <= int(i) < n:
            x0[int(i)] = value

    uno_model.set_initial_primal_iterate(x0)

    uno_solver = unopy.UnoSolver()

    if not log:
        import io
        uno_solver.set_logger_stream(io.StringIO())

    preset = solver_options.get('preset', 'ipopt')
    uno_solver.set_preset(preset)

    for key, value in solver_options.items():
        if key.startswith("---"):
            continue
        if value is None:
            continue
        if key != 'preset':
            uno_solver.set_option(key, value)



    time_solve_begin = timeit.default_timer()
    result = uno_solver.optimize(uno_model)
    time_solve_end = timeit.default_timer()

    return result, [time_solve_begin, time_solve_end], features['variables']
