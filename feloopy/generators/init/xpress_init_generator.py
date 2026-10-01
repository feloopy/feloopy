# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Xpress warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_xpress_mip_start', {})[int(variable.index)] = float(value)


def set_init_value(features, variable, value, fix):
    if fix:
        variable.lb = value
        variable.ub = value
    else:
        _stage(features, variable, value)


def set_init_values(features, assignments, fix=False):
    if fix:
        for variable, value in assignments:
            variable.lb = value
            variable.ub = value
        return
    for variable, value in assignments:
        _stage(features, variable, value)


def finalize(features):
    staged = features.get('_xpress_mip_start')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return

    colind = sorted(staged)
    solval = [staged[i] for i in colind]

    status = None
    try:
        status = model_object.addMipSol(solval, colind)
    except Exception:
        status = None

    if status in (None, -1):
        dense = [0.0] * int(model_object.attributes.cols)
        for i, v in staged.items():
            if 0 <= i < len(dense):
                dense[i] = v
        model_object.loadMipSol(dense)


def set_branch_priority(features, variable, priority):
    try:
        variable.branching_priority = priority
    except Exception:
        pass


def set_var_hint(variable, hint_val, hint_pri=0):
    try:
        variable.sol = hint_val
    except Exception:
        pass
