# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import xpress as xpress_interface
import timeit

xpress_solver_selector = {'xpress': 'xpress'}


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

    if solver_name not in xpress_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'xpress'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    if log:
        model_object.controls.outputlog = 1
    else:
        model_object.controls.outputlog = 0

    if time_limit is not None:
        model_object.controls.timelimit = time_limit

    if thread_count is not None:
        model_object.controls.threads = thread_count

    if relative_gap is not None:
        model_object.controls.miprelstop = relative_gap

    if absolute_gap is not None:
        model_object.controls.mipabsstop = absolute_gap

    if max_iterations is not None:
        model_object.controls.maxnode = max_iterations

    if save is not False:
        try:
            model_object.controls.savelog = str(save)
        except Exception:
            pass

    if len(solver_options) != 0:
        for key in solver_options:
            if key.startswith("---"):
                continue
            val = solver_options[key]
            if val is None:
                continue
            try:
                model_object.controls.__setattr__(key, val)
            except Exception:
                pass

    match debug:

        case False | True:

            for i, constraint in enumerate(model_constraints):
                label = constraint_labels[i] if i < len(constraint_labels) else None
                if label is not None:
                    constraint.name = str(label)
                try:
                    model_object.addConstraint(constraint)
                except Exception:
                    pass

            match directions[objective_id]:

                case "min":
                    model_object.setObjective(
                        model_objectives[objective_id], sense=xpress_interface.minimize)

                case "max":
                    model_object.setObjective(
                        model_objectives[objective_id], sense=xpress_interface.maximize)

            if callback is not None:
                try:
                    model_object.setCallback(callback)
                except Exception:
                    pass

            if save_model is not False:
                try:
                    model_object.write(str(save_model))
                except Exception:
                    pass

            from ..init_generator import flush_init
            flush_init(features, force=True)

            time_solve_begin = timeit.default_timer()
            result = model_object.solve()
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end]]

    return generated_solution
