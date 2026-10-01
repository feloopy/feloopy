# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import xpress as xpress_interface
import itertools as it

sets = it.product

VariableGenerator = xpress_interface.var

INFINITY = xpress_interface.infinity
BINARY = xpress_interface.binary
INTEGER = xpress_interface.integer

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    bound = list(variable_bound)
    lb = bound[0] if bound[0] is not None else -INFINITY
    ub = bound[1] if bound[1] is not None else +INFINITY

    def _get_keys():
        if isinstance(variable_dim, set):
            return list(variable_dim)
        elif len(variable_dim) == 1:
            return list(variable_dim[0])
        else:
            return list(sets(*variable_dim))

    def _make_scalar():
        v = model_object.addVariable(name=variable_name, lb=lb, ub=ub)
        return v

    def _make_array(keys):
        vs = [model_object.addVariable(name=f"{variable_name}{key}", lb=lb, ub=ub) for key in keys]
        return dict(zip(keys, vs))

    def _make_binary_scalar():
        v = model_object.addVariable(name=variable_name, vartype=BINARY)
        return v

    def _make_binary_array(keys):
        vs = [model_object.addVariable(name=f"{variable_name}{key}", vartype=BINARY) for key in keys]
        return dict(zip(keys, vs))

    def _make_integer_scalar():
        v = model_object.addVariable(name=variable_name, lb=lb, ub=ub, vartype=INTEGER)
        return v

    def _make_integer_array(keys):
        vs = [model_object.addVariable(name=f"{variable_name}{key}", lb=lb, ub=ub, vartype=INTEGER) for key in keys]
        return dict(zip(keys, vs))

    match variable_type:

        case 'pvar':
            if variable_dim == 0:
                return _make_scalar()
            return _make_array(_get_keys())

        case 'bvar':
            if variable_dim == 0:
                return _make_binary_scalar()
            return _make_binary_array(_get_keys())

        case 'ivar':
            if variable_dim == 0:
                return _make_integer_scalar()
            return _make_integer_array(_get_keys())

        case 'fvar' | 'rvar' | 'dvar':
            if variable_dim == 0:
                return _make_scalar()
            return _make_array(_get_keys())
