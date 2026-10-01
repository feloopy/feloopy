# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product


def _make_scalar(model_object, vtype, lb, ub, name):
    if vtype == 'fvar' or vtype == 'ftvar' or vtype == 'pvar' or vtype == 'ptvar' or vtype == 'rvar':
        return model_object.NumVar(lb, ub, name)
    return model_object.IntVar(lb, ub, name)


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0] if variable_bound[0] is not None else -model_object.infinity()
    ub = variable_bound[1] if variable_bound[1] is not None else model_object.infinity()

    _continuous = {'pvar', 'ptvar', 'fvar', 'ftvar', 'rvar'}

    if variable_dim == 0:
        return _make_scalar(model_object, variable_type, lb, ub, variable_name)

    if isinstance(variable_dim, set):
        return {key: _make_scalar(model_object, variable_type, lb, ub, f"{variable_name}{key}") for key in variable_dim}

    if len(variable_dim) == 1:
        return {key: _make_scalar(model_object, variable_type, lb, ub, f"{variable_name}{key}") for key in variable_dim[0]}

    return {key: _make_scalar(model_object, variable_type, lb, ub, f"{variable_name}{key}") for key in sets(*variable_dim)}
