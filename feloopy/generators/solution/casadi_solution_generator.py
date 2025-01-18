# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import casadi as cas
import timeit

casadi_solver_selector = {'ipopt': 'ipopt'}

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
    max_iterations = features['max_iterations']
    solver_options = features['solver_options']
    variables = {k[1]: v for k, v in features["variables"].items()}

    # Validate Solver
    if solver_name not in casadi_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'casadi'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    # Solver options
    solver_opts = {}
    if time_limit is not None:
        solver_opts['time_limit'] = time_limit
    if relative_gap is not None:
        solver_opts['tol'] = relative_gap
    if absolute_gap is not None:
        solver_opts['acceptable_tol'] = absolute_gap
    if thread_count is not None:
        solver_opts['max_cpu_time'] = thread_count
    if log:
        solver_opts['print_level'] = 5
    if max_iterations is not None:
        solver_opts['max_iter'] = max_iterations

    for key, value in solver_options.items():
        solver_opts[key] = value

    match debug:

        case False:

            # Set Objective
            match directions[objective_id]:

                case 'min':
                    model_object.minimize(model_objectives[objective_id])

                case 'max':
                    model_object.minimize(-model_objectives[objective_id])

            # Add Constraints
            for idx, constraint in enumerate(model_constraints):
                model_object.subject_to(constraint)

            # Save model if needed
            if save_model:
                with open(save_model, 'w') as model_file:
                    model_file.write(str(model_object))

            # Solve the problem
            model_object.solver(solver_name, solver_opts)
            time_solve_begin = timeit.default_timer()
            result = model_object.solve()
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end], variables]

    # Save solver logs
    if save:
        with open(save, "w") as log_file:
            log_file.write(str(result))

    return generated_solution
