# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import highspy as highs_interface
import itertools as it

sets = it.product

POSITIVE = highs_interface.HighsVarType.kContinuous
INTEGER = highs_interface.HighsVarType.kInteger
BINARY = highs_interface.HighsVarType.kInteger
FREE = highs_interface.HighsVarType.kContinuous
INFINITY = highs_interface.kHighsInf

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    variable_bound = [
        variable_bound[0] if variable_bound[0] is not None else -INFINITY,
        variable_bound[1] if variable_bound[1] is not None else +INFINITY,
    ]

    if isinstance(variable_dim,set):
        variable_dim=[variable_dim]

    match variable_type:

        case 'fvar' | 'ftvar':

            if variable_dim == 0:

                generated_variable = model_object.addVariable(
                    type=FREE, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)

            else:

                if len(variable_dim) == 1:

                    generated_variable = {key: model_object.addVariable(
                        type=FREE, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}

                else:

                    generated_variable = {key: model_object.addVariable(
                        type=FREE, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'pvar' | 'ptvar':

            if variable_dim == 0:

                generated_variable = model_object.addVariable(
                    type=POSITIVE, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)

            else:

                if len(variable_dim) == 1:

                    generated_variable = {key: model_object.addVariable(
                        type=POSITIVE, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}

                else:

                    generated_variable = {key: model_object.addVariable(
                        type=POSITIVE, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'bvar' | 'btvar':

            if variable_dim == 0:

                generated_variable = model_object.addVariable(
                    type=BINARY, lb=0, ub=1, name=variable_name)

            else:

                if len(variable_dim) == 1:

                    generated_variable = {key: model_object.addVariable(
                        type=BINARY, lb=0, ub=1, name=f"{variable_name}{key}") for key in variable_dim[0]}

                else:

                    generated_variable = {key: model_object.addVariable(
                        type=BINARY, lb=0, ub=1, name=f"{variable_name}{key}") for key in sets(*variable_dim)}

        case 'ivar' | 'itvar':

            if variable_dim == 0:

                generated_variable = model_object.addVariable(
                    type=INTEGER, lb=variable_bound[0], ub=variable_bound[1], name=variable_name)

            else:

                if len(variable_dim) == 1:

                    generated_variable = {key: model_object.addVariable(
                        type=INTEGER, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in variable_dim[0]}

                else:

                    generated_variable = {key: model_object.addVariable(
                        type=INTEGER, lb=variable_bound[0], ub=variable_bound[1], name=f"{variable_name}{key}") for key in sets(*variable_dim)}

    return generated_variable
