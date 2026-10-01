# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""COPT warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_copt_mip_start', {})[id(variable)] = (
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
            set_init_value(features, variable, value, fix=True)
        return
    for variable, value in assignments:
        _stage(features, variable, value)


def finalize(features):
    staged = features.get('_copt_mip_start')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return

    variables = [variable for variable, _value in staged.values()]
    values = [value for _variable, value in staged.values()]

    model_object.setMipStart(variables, values)
    try:
        model_object.loadMipStart()
    except Exception:
        # already registered by setMipStart on some versions
        pass
