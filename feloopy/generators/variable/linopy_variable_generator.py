# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import itertools as it
import numpy as np


def _coords(variable_dim):
    if isinstance(variable_dim, (set, frozenset)):
        return sorted(variable_dim)
    return [sorted(d) if isinstance(d, (set, frozenset)) else d
            for d in variable_dim]

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0]
    ub = variable_bound[1]

    match variable_type:

        case 'pvar':
            
            lb = variable_bound[0] if variable_bound[0] is not None else 0
            ub = variable_bound[1] if variable_bound[1] is not None else np.inf
            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, name=variable_name)
            else:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name)

        case 'bvar':
            lb = variable_bound[0] if variable_bound[0] is not None else 0
            ub = variable_bound[1] if variable_bound[1] is not None else 1
            
            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(name=variable_name, binary=True)
            else:
                GeneratedVariable = model_object.add_variables(coords=_coords(variable_dim), name=variable_name, binary=True)

        case 'ivar':
            lb = variable_bound[0] if variable_bound[0] is not None else 0
            ub = variable_bound[1] if variable_bound[1] is not None else np.inf
            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, name=variable_name, integer=True)
            else:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name, integer=True)

        case 'fvar':

            lb = variable_bound[0] if variable_bound[0] is not None else -np.inf
            ub = variable_bound[1] if variable_bound[1] is not None else np.inf
            
            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, name=variable_name)
            else:
                GeneratedVariable = model_object.add_variables(lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name)

        case 'ptvar':
            
            
            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, name=variable_name, positive=True)
            else:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name, positive=True)

        case 'btvar':

            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(
                    name=variable_name, binary=True)
            else:
                GeneratedVariable = model_object.add_variables(
                    coords=_coords(variable_dim), name=variable_name, binary=True)

        case 'itvar':

            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, name=variable_name, integer=True)
            else:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name, integer=True)

        case 'ftvar':

            if variable_dim == 0:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, name=variable_name)
            else:
                GeneratedVariable = model_object.add_variables(
                    lower=lb, upper=ub, coords=_coords(variable_dim), name=variable_name)

    return GeneratedVariable
