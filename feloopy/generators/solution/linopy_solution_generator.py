# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
import os
import sys

def _import_linopy_quietly():
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    saved_stdout_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        os.dup2(devnull, stdout_fd)
        from linopy import Model as LINOPYMODEL
        from linopy import LinearExpression
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(devnull)
    return LINOPYMODEL, LinearExpression

LINOPYMODEL, LinearExpression = _import_linopy_quietly()

linopy_solver_selector = {'cbc': 'cbc',
                          'glpk': 'glpk',
                          'highs': 'highs',
                          'gurobi': 'gurobi',
                          'xpress': 'xpress',
                          'cplex': 'cplex',
                          'copt': 'copt'}

def _suppress_output(fd_list):
    """Suppress both stdout and stderr at file-descriptor level."""
    saved = []
    devnull = os.open(os.devnull, os.O_WRONLY)
    for fd in fd_list:
        saved.append(os.dup(fd))
        os.dup2(devnull, fd)
    os.close(devnull)
    return saved

def _restore_output(fd_list, saved):
    """Restore previously suppressed file descriptors."""
    for fd, s in zip(fd_list, saved):
        os.dup2(s, fd)
        os.close(s)

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
    
    if solver_name not in linopy_solver_selector.keys():
        raise RuntimeError("Using solver '%s' is not supported by 'linopy'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))
    match debug:
        case False | True:
            match directions[objective_id]:
                case "min":
                    model_object.add_objective(model_objectives[objective_id], overwrite=True)
                case "max":
                    model_object.add_objective(-1*model_objectives[objective_id], overwrite=True)
            for constraint, label in zip(model_constraints, constraint_labels): model_object.add_constraints(constraint, name=label)
            time_solve_begin = timeit.default_timer()
            if log:
                result = model_object.solve(solver_name=solver_name)
            else:
                fds = [sys.stdout.fileno(), sys.stderr.fileno()]
                saved = _suppress_output(fds)
                try:
                    result = model_object.solve(solver_name=solver_name)
                finally:
                    _restore_output(fds, saved)
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end]]
    return generated_solution
