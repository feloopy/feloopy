# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

sets = it.product


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    try:
        import pyscipopt as scip
    except ImportError:
        raise ImportError(
            "The 'pyscipopt' package is required for SCIP interface. "
            "Install it with: pip install PySCIPOpt"
        )

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    lb = variable_bound[0] if variable_bound[0] is not None else -1e20
    ub = variable_bound[1] if variable_bound[1] is not None else 1e20

    if isinstance(lb, (int, float)):
        lb = float(lb)
    else:
        lb = -1e20

    if isinstance(ub, (int, float)):
        ub = float(ub)
    else:
        ub = 1e20

    vtype_map = {
        'pvar': 'CONTINUOUS',
        'fvar': 'CONTINUOUS',
        'bvar': 'BINARY',
        'ivar': 'INTEGER',
    }

    vtype = vtype_map.get(variable_type, 'CONTINUOUS')

    if variable_dim == 0:
        var = model_object.addVar(variable_name, vtype=vtype, lb=lb, ub=ub)
        return var
    else:
        if len(variable_dim) == 1:
            indices = list(variable_dim[0])
        else:
            indices = list(sets(*variable_dim))

        var_dict = {}
        for key in indices:
            if isinstance(key, tuple):
                name = f"{variable_name}[{','.join(str(k) for k in key)}]"
            else:
                name = f"{variable_name}[{key}]"
            var = model_object.addVar(name, vtype=vtype, lb=lb, ub=ub)
            var_dict[key] = var

        if len(indices) == 1:
            return {indices[0]: var_dict[indices[0]]}
        return var_dict
