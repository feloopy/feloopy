# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
import casadi as cas

sets = it.product


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    match variable_type:

        case 'pvar':  # Continuous variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                
                model_object.subject_to(GeneratedVariable >= variable_bound[0])
                if variable_bound[1]:
                    model_object.subject_to(GeneratedVariable <= variable_bound[1])

            else:
                if len(variable_dim) == 1:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in variable_dim[0]
                    }
                    for key in variable_dim[0]:
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1]:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1]:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

        case 'bvar':  # Binary variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.subject_to(GeneratedVariable >= 0)
                model_object.subject_to(GeneratedVariable <= 1.5)
                model_object.subject_to(GeneratedVariable == cas.floor(GeneratedVariable))

            else:
                if len(variable_dim) == 1:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in variable_dim[0]
                    }
                    for key in variable_dim[0]:
                        model_object.subject_to(GeneratedVariable[key] >= 0)
                        model_object.subject_to(GeneratedVariable[key] <= 1.5)
                        model_object.subject_to(GeneratedVariable == cas.floor(GeneratedVariable))

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.subject_to(GeneratedVariable[key] >= 0)
                        model_object.subject_to(GeneratedVariable[key] <= 1.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

        case 'ivar':  # Integer variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.subject_to(GeneratedVariable >= variable_bound[0])
                model_object.subject_to(GeneratedVariable <= variable_bound[1]+0.5)
                model_object.subject_to(GeneratedVariable == cas.floor(GeneratedVariable))

            else:
                if len(variable_dim) == 1:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in variable_dim[0]
                    }
                    for key in variable_dim[0]:
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        model_object.subject_to(GeneratedVariable[key] <= variable_bound[1]+0.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        model_object.subject_to(GeneratedVariable[key] <= variable_bound[1]+0.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

        case 'fvar':  # Free variable (unbounded continuous)

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()

            else:
                if len(variable_dim) == 1:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in variable_dim[0]
                    }

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }

    return GeneratedVariable
