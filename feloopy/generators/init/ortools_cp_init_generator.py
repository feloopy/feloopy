# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""OR-Tools CP-SAT warm start (``CpModel.AddHint``)."""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_ortools_cp_start', {})[id(variable)] = (
        variable, float(value))


def set_init_value(features, variable, value, fix):
    if fix:
        model_object = _get_model(features)
        if model_object is not None:
            model_object.Add(variable == int(round(float(value))))
        return
    _stage(features, variable, value)


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)


def finalize(features):
    staged = features.get('_ortools_cp_start')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return
    for variable, value in staged.values():
        model_object.AddHint(variable, int(round(value)))
