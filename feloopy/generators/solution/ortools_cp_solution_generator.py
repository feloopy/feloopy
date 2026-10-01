# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


from ortools.sat.python import cp_model
import timeit


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

    if len(directions) != 0:

        match directions[objective_id]:

            case "min":
                model_object.Minimize(model_objectives[objective_id])

            case "max":
                model_object.Maximize(model_objectives[objective_id])

    for i, constraint in enumerate(model_constraints):
        if constraint is None:
            continue
        label = constraint_labels[i] if i < len(constraint_labels) and constraint_labels[i] is not None else f"con_{i}"
        model_object.Add(constraint)

    solver = cp_model.CpSolver()

    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit

    if thread_count is not None:
        solver.parameters.num_workers = thread_count

    if relative_gap is not None:
        solver.parameters.relative_gap_limit = relative_gap

    if absolute_gap is not None:
        solver.parameters.absolute_gap_limit = absolute_gap

    if max_iterations is not None:
        solver.parameters.max_deterministic_time = max_iterations

    if 'enumerate' in solver_options:
        solver.parameters.enumerate_all_solutions = solver_options['enumerate']

    if 'log' in solver_options:
        solver.parameters.log_search_progress = solver_options['log']
    elif log:
        solver.parameters.log_search_progress = True

    from ..init_generator import flush_init
    flush_init(features, force=True)

    time_solve_begin = timeit.default_timer()
    if callback is not None:
        result = solver.Solve(model_object, callback)
    else:
        result = solver.Solve(model_object)
    time_solve_end = timeit.default_timer()

    generated_solution = [[result, solver],
                          [time_solve_begin, time_solve_end]]

    if log:

        _status_code_to_name = {
            int(cp_model.UNKNOWN): "unknown",
            int(cp_model.MODEL_INVALID): "model_invalid",
            int(cp_model.INFEASIBLE): "infeasible",
            int(cp_model.FEASIBLE): "feasible",
            int(cp_model.OPTIMAL): "optimal",
        }
        _name = _status_code_to_name.get(int(result), f"UNKNOWN({result})")
        _solved = int(result) in (int(cp_model.OPTIMAL), int(cp_model.FEASIBLE))

        print('\nStatistics')
        print(f'  status         : {_name}')
        print(f'  conflicts      : {solver.NumConflicts()}')
        print(f'  branches       : {solver.NumBranches()}')
        print(f'  wall time      : {solver.WallTime()} s')
        if _solved:
            try:
                print(f'  objective      : {solver.ObjectiveValue()}')
            except Exception:
                print(f'  objective      : N/A')

    return generated_solution
