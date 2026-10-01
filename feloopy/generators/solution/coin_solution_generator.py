# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
import logging

logging.getLogger('pyomo').setLevel(logging.CRITICAL)

bonmin_solver_selector = {'bonmin': 'bonmin'}
couenne_solver_selector = {'couenne': 'couenne'}

coin_solver_map = {}
coin_solver_map.update(bonmin_solver_selector)
coin_solver_map.update(couenne_solver_selector)


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

    if solver_name not in coin_solver_map.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'bonmin' or 'couenne'! "
            "Possible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    try:
        import pyomo.environ as pyo
        from pyomo.environ import Objective, Constraint, ConstraintList, minimize, maximize
        from pyomo.opt import SolverFactory
    except ImportError:
        raise ImportError(
            "The 'pyomo' package is required for bonmin/couenne interface. "
            "Install it with: flp install pyomo  (or: pip install pyomo)"
        )

    objective_expr = model_objectives[objective_id]
    direction = directions[objective_id]

    if direction == 'min':
        obj = Objective(expr=objective_expr, sense=minimize)
    else:
        obj = Objective(expr=objective_expr, sense=maximize)

    if model_object.component('_felooopy_obj') is not None:
        model_object.del_component('_felooopy_obj')
    model_object.add_component('_felooopy_obj', obj)

    if len(model_constraints) != 0:
        if model_object.component('_felooopy_constraint_list') is not None:
            model_object.del_component('_felooopy_constraint_list')
        if constraint_labels[0] is None:
            model_object._felooopy_constraint_list = ConstraintList()
            for element in model_constraints:
                model_object._felooopy_constraint_list.add(expr=element)
        else:
            counter = 0
            for element in model_constraints:
                label = constraint_labels[counter] if counter < len(constraint_labels) else f'_felooopy_con_{counter}'
                if model_object.component(label) is not None:
                    model_object.del_component(label)
                con = Constraint(expr=element)
                model_object.add_component(label, con)
                counter += 1

    from ...helpers.solver_executables import (
        ensure_solver_on_path,
        missing_solver_message,
    )
    from ...helpers.solver_provisioning import ensure_solver

    solver_path = ensure_solver(solver_name)
    if not solver_path:
        raise RuntimeError(missing_solver_message(solver_name))
    ensure_solver_on_path(solver_path)
    opt = SolverFactory(solver_name, executable=solver_path)

    if time_limit is not None:
        if solver_name == 'bonmin':
            opt.options['bonmin_time_limit'] = time_limit
        elif solver_name == 'couenne':
            opt.options['couenne_time_limit'] = time_limit

    for key, value_opt in solver_options.items():
        if key.startswith("---"):
            continue
        if value_opt is None:
            continue
        opt.options[key] = value_opt

    time_solve_begin = timeit.default_timer()
    result = opt.solve(model_object, tee=log)
    model_object.solutions.load_from(result)
    time_solve_end = timeit.default_timer()

    return result, [time_solve_begin, time_solve_end]
