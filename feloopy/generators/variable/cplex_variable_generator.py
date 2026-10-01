# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    bound = list(variable_bound)
    lb = bound[0] if bound[0] is not None else -1e20
    ub = bound[1] if bound[1] is not None else +1e20

    def _make_scalar():
        return model_object.continuous_var(lb=lb, ub=ub, name=variable_name)

    def _make_array(keys):
        return {key: model_object.continuous_var(lb=lb, ub=ub, name=f"{variable_name}{key}") for key in keys}

    def _make_dict(keys):
        return model_object.continuous_var_dict(keys, lb=lb, ub=ub, name=variable_name)

    def _get_keys():
        if isinstance(variable_dim, set):
            return variable_dim
        elif len(variable_dim) == 1:
            return variable_dim[0]
        else:
            return list(sets(*variable_dim))

    match variable_type:

        case 'pvar':
            if variable_dim == 0:
                return model_object.continuous_var(lb=lb, ub=ub, name=variable_name)
            keys = _get_keys()
            if isinstance(variable_dim, set):
                return _make_dict(keys)
            return _make_array(keys)

        case 'bvar':
            if variable_dim == 0:
                return model_object.binary_var(name=variable_name)
            keys = _get_keys()
            if isinstance(variable_dim, set):
                return {key: model_object.binary_var(name=f"{variable_name}{key}") for key in keys}
            return {key: model_object.binary_var(name=f"{variable_name}{key}") for key in keys}

        case 'ivar':
            if variable_dim == 0:
                return model_object.integer_var(lb=lb, ub=ub, name=variable_name)
            keys = _get_keys()
            if isinstance(variable_dim, set):
                return model_object.integer_var_dict(keys, lb=lb, ub=ub, name=variable_name)
            return {key: model_object.integer_var(lb=lb, ub=ub, name=f"{variable_name}{key}") for key in keys}

        case 'fvar' | 'rvar' | 'dvar':
            if variable_dim == 0:
                return model_object.continuous_var(lb=lb, ub=ub, name=variable_name)
            keys = _get_keys()
            if isinstance(variable_dim, set):
                return _make_dict(keys)
            return _make_array(keys)
