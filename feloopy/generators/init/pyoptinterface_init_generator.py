# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""pyoptinterface warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def set_init_value(features, variable, value, fix):
    model_object = _get_model(features)
    if model_object is None:
        return

    if fix:
        model_object.set_variable_bounds(variable, value, value)
        return

    model_object.set_primal_start([variable], [float(value)])


def set_init_values(features, assignments, fix=False):
    if fix:
        for variable, value in assignments:
            set_init_value(features, variable, value, fix=True)
        return
    if not assignments:
        return
    model_object = _get_model(features)
    if model_object is None:
        return
    model_object.set_primal_start(
        [variable for variable, _value in assignments],
        [float(value) for _variable, value in assignments])


def set_branch_priority(features, variable, priority):
    pass


def set_var_hint(variable, hint_val, hint_pri=0):
    try:
        model = variable.model
        model.set_primal_start([variable], [float(hint_val)])
    except Exception:
        pass
