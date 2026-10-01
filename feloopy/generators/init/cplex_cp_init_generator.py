# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""CP Optimizer (docplex.cp) warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _stage(features, variable, value):
    features.setdefault('_cplex_cp_start', {})[id(variable)] = (
        variable, float(value))


def set_init_value(features, variable, value, fix):
    if fix:
        # CP Optimizer variable domains are fixed at creation, so a fix has to
        # become a constraint
        model_object = _get_model(features)
        if model_object is not None:
            model_object.add(variable == int(round(float(value))))
        return
    _stage(features, variable, value)


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)


def finalize(features):
    staged = features.get('_cplex_cp_start')
    if not staged:
        return
    model_object = _get_model(features)
    if model_object is None:
        return

    solution = model_object.create_empty_solution()
    for variable, value in staged.values():
        solution[variable] = int(round(value))
    model_object.set_starting_point(solution)
