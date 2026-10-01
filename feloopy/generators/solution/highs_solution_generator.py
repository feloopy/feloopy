# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import highspy as highs_interface
import timeit
import numpy as np

highs_solver_selector = {'highs': 'highs'}

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

    if solver_name not in highs_solver_selector.keys():
        raise RuntimeError("Using solver '%s' is not supported by 'highs'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    try:
        model_object.setOptionValue('output_flag', log)
    except Exception:
        pass
    if time_limit:
        model_object.setOptionValue('time_limit', time_limit)
    if thread_count:
        model_object.setOptionValue('threads', thread_count)
    if absolute_gap:
        model_object.setOptionValue('mip_abs_gap', absolute_gap)
    if relative_gap:
        model_object.setOptionValue('mip_rel_gap', relative_gap)
    for key in solver_options.keys():
        if key.startswith("---"):
            continue
        val = solver_options[key]
        if val is None:
            continue
        if isinstance(val, (bool, int, float, str)):
            try:
                model_object.setOptionValue(key, val)
            except Exception:
                pass
            
    from highspy.highs import HighsStatus
    counter = 0
    if features.get('_reusable_rows'):
        try:
            _n_rows = model_object.getNumRow()
            if _n_rows:
                model_object.deleteRows(
                    _n_rows, np.arange(_n_rows, dtype=np.int32))
        except Exception:
            pass
    for constraint, label in zip(model_constraints, constraint_labels):
        if constraint is None or isinstance(constraint, (bool, int, float, np.bool_, np.integer, np.floating)):
            counter += 1
            continue
        try:
            model_object.addConstr(constraint, name=label)
        except Exception:
            idxs, vals = constraint.unique_elements()
            lb = constraint.bounds[0]
            ub = constraint.bounds[1]
            row_idx = model_object.numConstrs
            model_object.addRow(lb, ub, len(idxs), list(idxs), list(vals))
            if label:
                model_object.passRowName(row_idx, label)
        counter += 1
    from ..init_generator import flush_init

    direction = directions[objective_id] if directions[objective_id] else "min"

    match direction:
        case "max":
            model_object.setObjective(model_objectives[objective_id],
                                       highs_interface.ObjSense.kMaximize)
        case _:
            model_object.setObjective(model_objectives[objective_id],
                                       highs_interface.ObjSense.kMinimize)

    flush_init(features, force=True)

    time_solve_begin = timeit.default_timer()
    result = model_object.solve()
    time_solve_end = timeit.default_timer()
    generated_solution = result, [time_solve_begin, time_solve_end]

    try:
        lp = model_object.getLp()
        a = lp.a_matrix_
        features['lp_data'] = {
            'n_cols': lp.num_col_,
            'n_rows': lp.num_row_,
            'row_lower': list(lp.row_lower_),
            'row_upper': list(lp.row_upper_),
            'col_lower': list(lp.col_lower_),
            'col_upper': list(lp.col_upper_),
            'col_cost': list(lp.col_cost_),
            'integrality': list(lp.integrality_),
            'col_names': list(lp.col_names_) if lp.col_names_ else [],
            'row_names': list(lp.row_names_) if lp.row_names_ else [],
            'A_col_pointers': list(a.start_),
            'A_row_indices': list(a.index_),
            'A_values': list(a.value_),
        }
    except Exception:
        pass

    return generated_solution

