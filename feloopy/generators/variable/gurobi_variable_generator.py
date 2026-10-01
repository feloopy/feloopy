# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import gurobipy as gurobi_interface
import itertools as it

sets = it.product

POSITIVE = gurobi_interface.GRB.CONTINUOUS
INTEGER = gurobi_interface.GRB.INTEGER
BINARY = gurobi_interface.GRB.BINARY
FREE = gurobi_interface.GRB.CONTINUOUS
SEMICONT = gurobi_interface.GRB.SEMICONT
SEMIINT = gurobi_interface.GRB.SEMIINT
INFINITY = gurobi_interface.GRB.INFINITY

VTYPE_MAP = {
    'pvar': POSITIVE, 'bvar': BINARY, 'ivar': INTEGER, 'fvar': FREE,
    'ptvar': POSITIVE, 'ftvar': FREE, 'btvar': BINARY, 'itvar': INTEGER,
    'rvar': FREE, 'dvar': FREE,
    'scont': SEMICONT, 'sint': SEMIINT,
    'scvar': SEMICONT, 'sivar': SEMIINT,
}

TENSOR_TYPES = {'ptvar', 'ftvar', 'btvar', 'itvar'}


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    bound = list(variable_bound)
    lb = bound[0] if bound[0] is not None else -INFINITY
    ub = bound[1] if bound[1] is not None else +INFINITY

    vtype = VTYPE_MAP.get(variable_type, FREE)

    def _make_scalar(keys=None):
        return model_object.addVar(vtype=vtype, lb=lb, ub=ub, name=variable_name)

    def _make_array(keys):
        return model_object.addVars(
            keys, vtype=vtype, lb=lb, ub=ub, name=variable_name
        )

    def _make_tensor(shape):
        return model_object.addMVar(shape, vtype=vtype, lb=lb, ub=ub, name=variable_name)

    if variable_dim == 0:
        return _make_scalar()

    if isinstance(variable_dim, set):
        keys = list(variable_dim)
    elif len(variable_dim) == 1:
        keys = variable_dim[0]
    else:
        keys = list(sets(*variable_dim))

    if variable_type in TENSOR_TYPES:
        shape = tuple(len(dim) for dim in variable_dim)
        return _make_tensor(shape)

    return _make_array(keys)
