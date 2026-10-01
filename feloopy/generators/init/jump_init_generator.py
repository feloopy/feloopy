# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""JuMP (Julia) warm start.

The model is emitted as Julia source, so the start is recorded here and the
solution generator writes ``set_start_value(jlmodel, var, value)`` lines into
the generated script.
"""


def _full_name(variable):
    getter = getattr(variable, '_full_name', None)
    if callable(getter):
        return getter()
    name = getattr(variable, 'name', None)
    if callable(name):
        name = name()
    return str(name if name is not None else variable)


def set_init_value(features, variable, value, fix):
    name = _full_name(variable)
    if fix:
        features.setdefault('_jump_fix', {})[name] = float(value)
    else:
        features.setdefault('_jump_start', {})[name] = float(value)


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)
