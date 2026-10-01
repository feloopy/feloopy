# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyomo.environ as pyomo_interface
import itertools as it

from ...helpers._pyomo_types import register_pyomo_numeric_types

register_pyomo_numeric_types()

sets = it.product

variable = pyomo_interface.Var

POSITIVE = pyomo_interface.NonNegativeReals
BINARY = pyomo_interface.Binary
INTEGER = pyomo_interface.NonNegativeIntegers
FREE = pyomo_interface.Reals

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    _domain_map = {
        'pvar': POSITIVE, 'ptvar': POSITIVE,
        'bvar': BINARY, 'btvar': BINARY,
        'ivar': INTEGER, 'itvar': INTEGER,
        'fvar': FREE, 'ftvar': FREE, 'rvar': FREE,
    }

    domain = _domain_map.get(variable_type, FREE)
    lb, ub = variable_bound[0], variable_bound[1]

    if variable_dim == 0:
        model_object.add_component(variable_name, variable(initialize=0, domain=domain, bounds=(lb, ub)))
    elif len(variable_dim) == 1:
        model_object.add_component(variable_name, variable([i for i in variable_dim[0]], domain=domain, bounds=(lb, ub)))
    else:
        model_object.add_component(variable_name, variable([i for i in sets(*variable_dim)], domain=domain, bounds=(lb, ub)))

    return model_object.component(variable_name)