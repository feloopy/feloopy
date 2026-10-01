# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
from ..picat_expression import PicatVar

sets = it.product

_CP_INT_MIN = -1000000000
_CP_INT_MAX = 1000000000


def _picat_safe_name(name):
    """Map a feloopy variable name to a valid Picat identifier.

    Picat source cannot contain spaces in variable names (they would split
    the token and make the generated program fail), so spaces become '_'.
    """
    if not name:
        return name
    name = name.replace(' ', '_')
    if name[0].isalpha() and name[0].islower():
        return name[0].upper() + name[1:]
    return name


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0] if variable_bound[0] is not None else _CP_INT_MIN
    ub = variable_bound[1] if variable_bound[1] is not None else _CP_INT_MAX

    if isinstance(lb, float):
        lb = int(lb)
    if isinstance(ub, float):
        ub = int(ub)

    if variable_dim == 0:

        safe_name = _picat_safe_name(variable_name)
        var = PicatVar(safe_name, lb, ub, model_object, variable_type)
        model_object.add_variable(var)
        return var

    else:

        if isinstance(variable_dim, set):
            keys = variable_dim
        elif len(variable_dim) == 1:
            keys = variable_dim[0]
        else:
            keys = it.product(*variable_dim)

        result = {}
        for key in keys:
            safe_name = _picat_safe_name(variable_name)
            if isinstance(key, tuple):
                name = safe_name + '_' + '_'.join(str(k) for k in key)
            else:
                name = f"{safe_name}_{key}"
            var = PicatVar(name, lb, ub, model_object, variable_type)
            model_object.add_variable(var)
            result[key] = var

        return result
