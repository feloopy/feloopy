# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""python-mip warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_mip_start', {})[id(variable)] = (
        variable, float(value))


def set_init_value(features, variable, value, fix):
    if fix:
        variable.lb = value
        variable.ub = value
    else:
        _stage(features, variable, value)


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)


def finalize(features):
    staged = features.get('_mip_start')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return
    model_object.start = [(variable, value)
                          for variable, value in staged.values()]
