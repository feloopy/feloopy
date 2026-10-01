# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyoptinterface as poi
import itertools as it

sets = it.product

POSITIVE = poi.VariableDomain.Continuous
INTEGER = poi.VariableDomain.Integer
BINARY = poi.VariableDomain.Binary
FREE = poi.VariableDomain.Continuous

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    bound = list(variable_bound)
    lb = bound[0] if bound[0] is not None else -float('inf')
    ub = bound[1] if bound[1] is not None else float('inf')

    def _get_keys():
        if isinstance(variable_dim, set):
            return list(variable_dim)
        elif len(variable_dim) == 1:
            return list(variable_dim[0])
        else:
            return list(sets(*variable_dim))

    match variable_type:

        case 'pvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=POSITIVE, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=POSITIVE, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'bvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=BINARY, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=BINARY, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'ivar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=INTEGER, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=INTEGER, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'fvar' | 'rvar' | 'dvar' | 'ftvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=FREE, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=FREE, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'ptvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=POSITIVE, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=POSITIVE, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'itvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=INTEGER, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=INTEGER, name=f"{variable_name}{key}") for key in _get_keys()}

        case 'btvar':
            if variable_dim == 0:
                return model_object.add_variable(lb=lb, ub=ub, domain=BINARY, name=variable_name)
            return {key: model_object.add_variable(lb=lb, ub=ub, domain=BINARY, name=f"{variable_name}{key}") for key in _get_keys()}
