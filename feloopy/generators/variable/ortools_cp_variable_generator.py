# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product

_CP_INT_MIN = -1000000000
_CP_INT_MAX = 1000000000

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = int(variable_bound[0]) if variable_bound[0] is not None else _CP_INT_MIN
    ub = int(variable_bound[1]) if variable_bound[1] is not None else _CP_INT_MAX

    match variable_type:

        case 'pvar':

            if variable_dim == 0:
                generated_variable = model_object.NewIntVar(lb, ub, variable_name)
            else:
                if isinstance(variable_dim,set):
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim}
                elif len(variable_dim) == 1:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in it.product(*variable_dim)}

        case 'bvar':

            if variable_dim == 0:
                generated_variable = model_object.NewIntVar(variable_bound[0], variable_bound[1], variable_name)
            else:
                if isinstance(variable_dim,set):
                    generated_variable = {key: model_object.NewIntVar(variable_bound[0], variable_bound[1], f"{variable_name}{key}") for key in variable_dim}
                elif len(variable_dim) == 1:
                    generated_variable = {key: model_object.NewIntVar(variable_bound[0], variable_bound[1], f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.NewIntVar(variable_bound[0], variable_bound[1], f"{variable_name}{key}") for key in it.product(*variable_dim)}

        case 'ivar':

            if variable_dim == 0:
                generated_variable = model_object.NewIntVar(lb, ub, variable_name)
            else:
                if isinstance(variable_dim,set):
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim}
                elif len(variable_dim) == 1:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in it.product(*variable_dim)}

        case 'fvar':
            if variable_dim == 0:
                generated_variable = model_object.NewIntVar(lb, ub, variable_name)
            else:
                if isinstance(variable_dim,set):
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim}
                elif len(variable_dim) == 1:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.NewIntVar(lb, ub, f"{variable_name}{key}") for key in it.product(*variable_dim)}

    return generated_variable
