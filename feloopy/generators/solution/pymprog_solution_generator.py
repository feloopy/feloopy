# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
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
            variables = {k[1]: v for k, v in features["variables"].items()}
            generated_solution = result, [time_solve_begin, time_solve_end], variables

        case True:

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
                result = pymprog_interface.solve(tmlim=tmlim)
            else:
                with suppress_output():
                    result = pymprog_interface.solve(tmlim=tmlim)
                
            time_solve_end = timeit.default_timer()
            variables = {k[1]: v for k, v in features["variables"].items()}
            generated_solution = result, [time_solve_begin, time_solve_end], variables

    try:
        _extract_lp_data(features)
    except Exception:
        pass

    return generated_solution


def _extract_lp_data(features):
    glp_prob = pymprog_interface.model._prob_._glp_
    swiglpk = pymprog_interface.glpk.swiglpk

    n_cols = swiglpk.glp_get_num_cols(glp_prob)
    n_rows = swiglpk.glp_get_num_rows(glp_prob)

    if n_cols == 0 or n_rows == 0:
        return

    constraint_labels = features.get('constraint_labels', [])
    constraint_counter = features.get('constraint_counter', [0, 0])

    glp_row_count = 0
    for label in constraint_labels:
        found_row = False
        for i in range(1, n_rows + 1):
            current_name = swiglpk.glp_get_row_name(glp_prob, i)
            if current_name.startswith('R') and current_name[1:].isdigit():
                swiglpk.glp_set_row_name(glp_prob, i, label)
                glp_row_count += 1
                found_row = True
                break
        if not found_row:
            break

    col_lower = []
    col_upper = []
    col_cost = []
    col_names = []
    integrality = []

    for j in range(1, n_cols + 1):
        lb = swiglpk.glp_get_col_lb(glp_prob, j)
        ub = swiglpk.glp_get_col_ub(glp_prob, j)
        cost = swiglpk.glp_get_obj_coef(glp_prob, j)
        kind = swiglpk.glp_get_col_kind(glp_prob, j)
        name = swiglpk.glp_get_col_name(glp_prob, j)

        if lb <= -1e30:
            lb = -1e20
        if ub >= 1e30:
            ub = 1e20

        col_lower.append(lb)
        col_upper.append(ub)
        col_cost.append(cost)
        col_names.append(name if name else f"x{j-1}")
        integrality.append(kind)

    row_lower = []
    row_upper = []
    row_names = []

    for i in range(1, n_rows + 1):
        lb = swiglpk.glp_get_row_lb(glp_prob, i)
        ub = swiglpk.glp_get_row_ub(glp_prob, i)
        name = swiglpk.glp_get_row_name(glp_prob, i)

        if lb <= -1e30:
            lb = -1e20
        if ub >= 1e30:
            ub = 1e20

        row_lower.append(lb)
        row_upper.append(ub)
        row_names.append(name if name else f"row_{i-1}")

    A_col_pointers = [0]
    A_row_indices = []
    A_values = []

    for j in range(1, n_cols + 1):
        nnz = swiglpk.glp_get_mat_col(glp_prob, j, None, None)
        ind = swiglpk.intArray(nnz + 1)
        val = swiglpk.doubleArray(nnz + 1)
        swiglpk.glp_get_mat_col(glp_prob, j, ind, val)
        for k in range(1, nnz + 1):
            A_row_indices.append(ind[k] - 1)
            A_values.append(val[k])
        A_col_pointers.append(A_col_pointers[-1] + nnz)

    features['lp_data'] = {
        'n_cols': n_cols,
        'n_rows': n_rows,
        'col_lower': col_lower,
        'col_upper': col_upper,
        'col_cost': col_cost,
        'col_names': col_names,
        'integrality': integrality,
        'row_lower': row_lower,
        'row_upper': row_upper,
        'row_names': row_names,
        'A_col_pointers': A_col_pointers,
        'A_row_indices': A_row_indices,
        'A_values': A_values,
    }
