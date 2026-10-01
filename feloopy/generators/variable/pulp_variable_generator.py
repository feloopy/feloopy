# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pulp as pulp_interface
import itertools as it

from ..pulp_compat import make_variable

sets = it.product

_type_map = {
    'pvar': pulp_interface.LpContinuous,
    'bvar': pulp_interface.LpBinary,
    'ivar': pulp_interface.LpInteger,
    'fvar': pulp_interface.LpContinuous,
}


def _make_var(model_object, name, lb, ub, vtype):
    return make_variable(model_object, name, lb, ub, vtype)


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    vtype = _type_map.get(variable_type, pulp_interface.LpContinuous)
    lb, ub = variable_bound[0], variable_bound[1]

    if variable_dim == 0:
        return _make_var(model_object, variable_name, lb, ub, vtype)

    if isinstance(variable_dim, set):
        return {k: _make_var(model_object, f"{variable_name}{k}", lb, ub, vtype) for k in variable_dim}

    if len(variable_dim) == 1:
        return {k: _make_var(model_object, f"{variable_name}{k}", lb, ub, vtype) for k in variable_dim[0]}

    return {k: _make_var(model_object, f"{variable_name}{k}", lb, ub, vtype) for k in sets(*variable_dim)}
