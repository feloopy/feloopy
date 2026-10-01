# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Uno warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def set_init_value(features, variable, value, fix):
    index = getattr(variable, 'index', None)
    if index is None:
        return
    index = int(index)

    if fix:
        # the bounds the solver actually sees are read off the model object
        model_object = _get_model(features)
        lowered = False
        for attr in ('variables_lower_bounds', 'variables_upper_bounds'):
            bounds = getattr(model_object, attr, None)
            if bounds is not None and 0 <= index < len(bounds):
                bounds[index] = float(value)
                lowered = True
        if not lowered:
            try:
                variable.lb = float(value)
                variable.ub = float(value)
            except Exception:
                pass
        return

    features.setdefault('_uno_start', {})[index] = float(value)
