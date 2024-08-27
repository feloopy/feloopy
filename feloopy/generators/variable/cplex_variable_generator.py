# Copyright (c) 2022-2024, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    match variable_type:

        case 'pvar':

            '''
            Positive Variable Generator
            '''

            if variable_dim == 0:
                GeneratedVariable = model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
            else:
                if isinstance(variable_dim,set):
                    GeneratedVariable = model_object.continuous_var_dict(variable_dim, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
                elif len(variable_dim) == 1:
                    GeneratedVariable = {key: model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    GeneratedVariable = {key: model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'bvar':

            '''
            Binary Variable Generator
            '''

            if variable_dim == 0:
                GeneratedVariable = model_object.binary_var(name=variable_name)
            else:
                if isinstance(variable_dim,set):
                    GeneratedVariable = model_object.binary_var_dict(variable_dim, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
                elif len(variable_dim) == 1:
                    GeneratedVariable = {key: model_object.binary_var(name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    GeneratedVariable = {key: model_object.binary_var(name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'ivar':

            '''
            Integer Variable Generator
            '''

            if variable_dim == 0:
                GeneratedVariable = model_object.integer_var(lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
            else:
                if isinstance(variable_dim,set):
                    GeneratedVariable = model_object.integer_var_dict(variable_dim, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
                elif len(variable_dim) == 1:
                    GeneratedVariable = {key: model_object.integer_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    GeneratedVariable = {key: model_object.integer_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'fvar':

            '''
            Free Variable Generator
            '''
            if variable_dim == 0:
                GeneratedVariable = model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
            else:
                if isinstance(variable_dim,set):
                    GeneratedVariable = model_object.continuous_var_dict(variable_dim, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)
                elif len(variable_dim) == 1:
                    GeneratedVariable = {key: model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}
                else:
                    GeneratedVariable = {key: model_object.continuous_var(lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

    return GeneratedVariable
