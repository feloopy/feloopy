# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import pymprog as pymprog_interface
import timeit

pymprog_solver_selector = {'glpk': 'glpk'}


import sys, os
import sys
import os
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def suppress_output():
    try:
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
    except io.UnsupportedOperation:
        with open(os.devnull, 'w') as devnull, \
             redirect_stdout(devnull), \
             redirect_stderr(devnull):
            yield
        return
    original_stdout = os.dup(original_stdout_fd)
    original_stderr = os.dup(original_stderr_fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), original_stdout_fd)
            os.dup2(devnull.fileno(), original_stderr_fd)
            yield
    finally:
        os.dup2(original_stdout, original_stdout_fd)
        os.dup2(original_stderr, original_stderr_fd)
        os.close(original_stdout)
        os.close(original_stderr)

            
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

    if log:

        msg_lev = pymprog_interface.glpk.GLP_MSG_OFF

    else:
        pass
        
        

    if time_limit != None:
        tmlim = time_limit
    else:
        tmlim= None

    if solver_name not in pymprog_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'pymprog'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    match debug:

        case False:

            match directions[objective_id]:

                case "min":
                    pymprog_interface.minimize(
                        model_objectives[objective_id], 'objective')
                case "max":
                    pymprog_interface.maximize(
                        model_objectives[objective_id], 'objective')

            for constraint in model_constraints:
                constraint
            time_solve_begin = timeit.default_timer()
            
            if log:
                result = pymprog_interface.solve(msg_lev=msg_lev, tmlim=tmlim)
            
            else:
                with suppress_output():
                    result = pymprog_interface.solve(tmlim=tmlim)
                
            time_solve_end = timeit.default_timer()
            generated_solution = result, [time_solve_begin, time_solve_end]

    return generated_solution
