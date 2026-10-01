# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit

hexaly_solver_selector = {'hexaly': 'hexaly'}

def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    save = features['save_solver_log']
    save_model = features['write_model_file']
    email = features['email_address']
    max_iterations = features['max_iterations']
    solver_options = features['solver_options']

    if solver_name not in hexaly_solver_selector.keys():
        raise RuntimeError("Using solver '%s' is not supported by 'hexaly'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    optimizer = features['hexaly_optimizer']

    if time_limit:
        optimizer.param.set_time_limit(int(time_limit * 1000))
    if thread_count:
        optimizer.param.set_nb_threads(thread_count)
    if relative_gap:
        optimizer.param.set_gap_limit(relative_gap)
    if not log:
        optimizer.param.set_verbosity(0)
    for key, val in solver_options.items():
        if key.startswith("---"):
            continue
        if val is None:
            continue
        optimizer.param.set_advanced_param(key, val)

    for constraint in model_constraints:
        model_object.add_constraint(constraint)

    match directions[objective_id]:
        case "min":
            model_object.minimize(model_objectives[objective_id])
        case "max":
            model_object.maximize(model_objectives[objective_id])

    time_solve_begin = timeit.default_timer()
    optimizer.solve()
    time_solve_end = timeit.default_timer()

    generated_solution = [None, [time_solve_begin, time_solve_end]]
    return generated_solution
