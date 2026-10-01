# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mosek
import timeit

INF = 1e30

mosek_solver_selector = {'mosek': 'mosek'}

def _add_constraint(task, constr):
    """Add a MosekConstr to the task, return constraint index."""
    from ..variable.mosek_expression import MosekConstr, MosekLin, MosekVar

    if isinstance(constr, MosekConstr):
        coeffs = dict(constr.coeffs)
        const = constr.const
        sense = constr.sense
        rhs_val = constr.rhs
    else:
        return None

    if isinstance(rhs_val, (int, float)):
        effective_rhs = float(rhs_val) - const
    else:
        effective_rhs = -const

    if not coeffs:
        return None

    con_idx = task.getnumcon()
    task.appendcons(1)

    var_indices = list(coeffs.keys())
    var_coeffs = [coeffs[k] for k in var_indices]
    task.putarow(con_idx, var_indices, var_coeffs)

    match sense:
        case '<=':
            task.putconbound(con_idx, mosek.boundkey.up, -INF, effective_rhs)
        case '>=':
            task.putconbound(con_idx, mosek.boundkey.lo, effective_rhs, INF)
        case '==':
            task.putconbound(con_idx, mosek.boundkey.fx, effective_rhs, effective_rhs)

    return con_idx


def _add_objective(task, obj_expr, direction):
    """Set objective from MosekLin or MosekVar."""
    from ..variable.mosek_expression import MosekLin, MosekVar

    if isinstance(obj_expr, MosekVar):
        coeffs = {obj_expr.idx: 1.0}
        const = 0.0
    elif isinstance(obj_expr, MosekLin):
        coeffs = dict(obj_expr.coeffs)
        const = obj_expr.const
    elif isinstance(obj_expr, (int, float)):
        return
    else:
        return

    var_indices = list(coeffs.keys())
    var_coeffs = [coeffs[k] for k in var_indices]
    task.putclist(var_indices, var_coeffs)

    if direction == 'min':
        task.putobjsense(mosek.objsense.minimize)
    else:
        task.putobjsense(mosek.objsense.maximize)


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

    if solver_name not in mosek_solver_selector.keys():
        raise RuntimeError("Using solver '%s' is not supported by 'mosek'!" % solver_name)

    task = model_object

    if not log:
        task.set_Stream(mosek.streamtype.log, lambda msg: None)

    if time_limit:
        task.putdouparam(mosek.dparam.optimizer_max_time, time_limit)
    if thread_count:
        task.putintparam(mosek.iparam.num_threads, thread_count)
    if absolute_gap:
        task.putdouparam(mosek.dparam.mio_abs_gap, absolute_gap)
    if relative_gap:
        task.putdouparam(mosek.dparam.mio_rel_gap, relative_gap)
    for key, val in solver_options.items():
        if key.startswith("---"):
            continue
        if val is None:
            continue
        try:
            if isinstance(val, int):
                task.putintparam(getattr(mosek.iparam, key, None) or key, val)
            elif isinstance(val, float):
                task.putdouparam(getattr(mosek.dparam, key, None) or key, val)
            elif isinstance(val, str):
                # String params not directly supported, ignore
                pass
        except Exception:
            pass

    con_label_map = {}
    for constraint, label in zip(model_constraints, constraint_labels):
        con_idx = _add_constraint(task, constraint)
        if con_idx is not None and label:
            task.putconname(con_idx, label)
            con_label_map[label] = con_idx

    features['_mosek_con_label_map'] = con_label_map

    _add_objective(task, model_objectives[objective_id], directions[objective_id])

    time_solve_begin = timeit.default_timer()
    task.optimize()
    time_solve_end = timeit.default_timer()

    generated_solution = [None, [time_solve_begin, time_solve_end]]
    return generated_solution
