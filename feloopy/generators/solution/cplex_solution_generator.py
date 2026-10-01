# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit

cplex_solver_selector = {'cplex': 'cplex'}


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
    callback = features.get('callback', None)

    if solver_name not in cplex_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'cplex'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    if time_limit is not None:
        model_object.parameters.timelimit.set(time_limit)

    if thread_count is not None:
        model_object.parameters.threads = thread_count

    if relative_gap is not None:
        model_object.parameters.mip.tolerances.mipgap = relative_gap

    if absolute_gap is not None:
        model_object.parameters.mip.tolerances.absmipgap = absolute_gap

    if max_iterations is not None:
        model_object.parameters.mip.limits.nodes = max_iterations

    if log:
        model_object.context.solver.log_output = True
        model_object.context.solver.verbose = 5
    else:
        model_object.context.solver.log_output = False
        model_object.context.solver.verbose = 0

    if len(solver_options) != 0:
        for key in solver_options:
            if key.startswith("---"):
                continue
            val = solver_options[key]
            if val is None:
                continue
            try:
                model_object.parameters.__setattr__(key, val)
            except Exception:
                try:
                    model_object.context.solver.__setattr__(key, val)
                except Exception:
                    pass

    if save is not False:
        try:
            model_object.set_results_stream(str(save))
        except Exception:
            pass

    match debug:

        case False | True:

            match directions[objective_id]:

                case 'min':
                    model_object.set_objective(
                        'min', model_objectives[objective_id])

                case 'max':
                    model_object.set_objective(
                        'max', model_objectives[objective_id])

            model_object.add_constraints(model_constraints, names=constraint_labels)

            if save_model is not False:
                if '.mps' in str(save_model):
                    model_object.export_as_mps(path=str(save_model))
                else:
                    model_object.export_as_lp(path=str(save_model))

            if callback is not None:
                model_object.set_output_callback(callback)

            from ..init_generator import flush_init
            flush_init(features, force=True)

            time_solve_begin = timeit.default_timer()
            result = model_object.solve()
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end]]

    if save is not False:
        try:
            with open(f"{save}.log", "w") as f:
                f.write(model_object.solution.to_string())
        except Exception:
            pass

    return generated_solution
