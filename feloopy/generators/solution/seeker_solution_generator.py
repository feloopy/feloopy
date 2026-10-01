# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit

seeker_solver_selector = {'seeker': 'seeker'}


def _enforce_constraint(model_object, constraint):
    if not isinstance(constraint, list) or len(constraint) < 3:
        return

    lhs = constraint[0]
    sense = constraint[1]
    rhs = constraint[2]

    if sense in ['<=', 'le', 'leq', '=l=']:
        model_object.enforce_leq(lhs, rhs)
    elif sense in ['>=', 'ge', 'geq', '=g=']:
        model_object.enforce_geq(lhs, rhs)
    elif sense in ['==', 'eq', '=e=']:
        model_object.enforce_eq(lhs, rhs)
    elif sense in ['<', 'lt']:
        model_object.enforce_lt(lhs, rhs)
    elif sense in ['>', 'gt']:
        model_object.enforce_gt(lhs, rhs)
    elif sense in ['!=', 'neq']:
        model_object.enforce_neq(lhs, rhs)


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    max_iterations = features['max_iterations']
    solver_options = features['solver_options']

    if solver_name not in seeker_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'seeker'! "
            "\nPossible fixes: \n1) Check the solver name. "
            "\n2) Use another interface. " % solver_name)

    for constraint in model_constraints:
        _enforce_constraint(model_object, constraint)

    time_limit_val = time_limit if time_limit is not None else 1e9
    lower_bound = solver_options.get('lb', -1e20)
    upper_bound = solver_options.get('ub', 1e20)

    if thread_count is not None:
        try:
            model_object.set_parameters({"NumberOfThreads": thread_count})
        except Exception:
            pass

    if max_iterations is not None:
        try:
            model_object.set_parameters({"MaxIterations": max_iterations})
        except Exception:
            pass

    user_params = {
        k: v for k, v in solver_options.items()
        if k not in ('lb', 'ub', 'license') and not k.startswith("---") and v is not None
    }
    if user_params:
        try:
            model_object.set_parameters(user_params)
        except Exception:
            pass

    direction = directions[objective_id]
    objective = model_objectives[objective_id]

    time_solve_begin = timeit.default_timer()
    if direction == 'min':
        model_object.minimize(objective, lowerBound=lower_bound, timeLimit=time_limit_val)
    else:
        model_object.maximize(objective, upperBound=upper_bound, timeLimit=time_limit_val)
    time_solve_end = timeit.default_timer()

    return [objective, [time_solve_begin, time_solve_end]]
