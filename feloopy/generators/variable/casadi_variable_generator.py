# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
import casadi as cas

sets = it.product


def _safe_init(ub, lb):
    """Pick a safe initial value for CasADi Opti variables.
    Default 0 causes issues for log, sqrt, etc."""
    if lb is not None and ub is not None:
        return (lb + ub) / 2.0
    elif lb is not None:
        return max(lb, 1.0) if lb <= 0 else float(lb)
    return 0.0


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    match variable_type:

        case 'pvar':  # Continuous variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.set_initial(GeneratedVariable, _safe_init(variable_bound[1], variable_bound[0]))
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
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1]:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1]:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

        case 'bvar':  # Binary variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.set_initial(GeneratedVariable, 0.5)
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
                        model_object.set_initial(GeneratedVariable[key], 0.5)
                        model_object.subject_to(GeneratedVariable[key] >= 0)
                        model_object.subject_to(GeneratedVariable[key] <= 1.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.set_initial(GeneratedVariable[key], 0.5)
                        model_object.subject_to(GeneratedVariable[key] >= 0)
                        model_object.subject_to(GeneratedVariable[key] <= 1.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

        case 'ivar':  # Integer variable

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.set_initial(GeneratedVariable, _safe_init(variable_bound[1], variable_bound[0]))
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
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        model_object.subject_to(GeneratedVariable[key] <= variable_bound[1]+0.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        model_object.subject_to(GeneratedVariable[key] <= variable_bound[1]+0.5)
                        model_object.subject_to(GeneratedVariable[key] == cas.floor(GeneratedVariable[key]))

        case 'fvar':  # Free variable (unbounded continuous)

            if variable_dim == 0:
                GeneratedVariable = model_object.variable()
                model_object.set_initial(GeneratedVariable, _safe_init(variable_bound[1], variable_bound[0]))
                if variable_bound[0] is not None:
                    model_object.subject_to(GeneratedVariable >= variable_bound[0])
                if variable_bound[1] is not None:
                    model_object.subject_to(GeneratedVariable <= variable_bound[1])

            else:
                if len(variable_dim) == 1:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in variable_dim[0]
                    }
                    for key in variable_dim[0]:
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        if variable_bound[0] is not None:
                            model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1] is not None:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

                else:
                    GeneratedVariable = {
                        key: model_object.variable()
                        for key in sets(*variable_dim)
                    }
                    for key in sets(*variable_dim):
                        model_object.set_initial(GeneratedVariable[key], _safe_init(variable_bound[1], variable_bound[0]))
                        if variable_bound[0] is not None:
                            model_object.subject_to(GeneratedVariable[key] >= variable_bound[0])
                        if variable_bound[1] is not None:
                            model_object.subject_to(GeneratedVariable[key] <= variable_bound[1])

    return GeneratedVariable
