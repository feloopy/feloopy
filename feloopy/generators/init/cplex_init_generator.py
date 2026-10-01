# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""CPLEX warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_cplex_mip_starts', {})[id(variable)] = (
        variable, float(value))


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
    staged = features.get('_cplex_mip_starts')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return

    from docplex.mp.solution import SolveSolution

    solution = SolveSolution(model_object, name='feloopy_warm_start')
    for variable, value in staged.values():
        solution[variable] = value

    try:
        model_object._mipstarts = []
    except Exception:
        pass

    try:
        model_object.add_mip_start(solution)
    except Exception as exc:
        if 'discrete' in str(exc).lower():
            return
        raise


def set_branch_priority(features, variable, priority):
    try:
        variable.branching_priority = priority
    except Exception:
        pass


def set_var_hint(variable, hint_val, hint_pri=0):
    try:
        variable.start_value = hint_val
    except Exception:
        pass
