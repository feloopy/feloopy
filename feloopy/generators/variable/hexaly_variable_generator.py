# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0] if variable_bound[0] is not None else -2147483647
    ub = variable_bound[1] if variable_bound[1] is not None else 2147483647

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    match variable_type:

        case 'pvar':

            if variable_dim == 0:
                generated_variable = model_object.float(0 if lb is None else lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.float(0 if lb is None else lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.float(0 if lb is None else lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'bvar':

            if variable_dim == 0:
                generated_variable = model_object.bool(name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.bool(name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.bool(name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'ivar':

            if variable_dim == 0:
                generated_variable = model_object.int(lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.int(lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.int(lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'fvar':

            if variable_dim == 0:
                generated_variable = model_object.float(lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.float(lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.float(lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'ptvar':

            if variable_dim == 0:
                generated_variable = model_object.float(0 if lb is None else lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.float(0 if lb is None else lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.float(0 if lb is None else lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'btvar':

            if variable_dim == 0:
                generated_variable = model_object.bool(name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.bool(name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.bool(name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'itvar':

            if variable_dim == 0:
                generated_variable = model_object.int(lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.int(lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.int(lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'ftvar':

            if variable_dim == 0:
                generated_variable = model_object.float(lb, ub, name=variable_name)
            else:
                if len(variable_dim) == 1:
                    generated_variable = {key: model_object.float(lb, ub, name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    generated_variable = {key: model_object.float(lb, ub, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

    return generated_variable
