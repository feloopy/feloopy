# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mosek
import itertools as it
from .mosek_expression import MosekVar

sets = it.product

INF = 1e30

CONT = mosek.variabletype.type_cont
INT = mosek.variabletype.type_int
FREE = mosek.boundkey.fr
LO = mosek.boundkey.lo
UP = mosek.boundkey.up
RA = mosek.boundkey.ra
FX = mosek.boundkey.fx

def _add_var(task, name, vtype, bk, lb, ub):
    idx = task.getnumvar()
    task.appendvars(1)
    task.putvartype(idx, vtype)
    task.putvarbound(idx, bk, lb, ub)
    task.putvarname(idx, name)
    return MosekVar(task, idx, name)

def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0] if variable_bound[0] is not None else -INF
    ub = variable_bound[1] if variable_bound[1] is not None else INF

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    def _bound_key(lb, ub, lo_val):
        if lb == -INF and ub == INF:
            return FREE, 0, 0
        elif lb == -INF:
            return UP, 0, ub
        elif ub == INF:
            return LO, lb, 0
        else:
            return RA, lb, ub

    match variable_type:

        case 'pvar' | 'ptvar':
            vtype = CONT
            lo_val = max(0, lb)
            if variable_dim == 0:
                return _add_var(model_object, variable_name, vtype, LO if lo_val >= 0 else RA, lo_val, ub)
            else:
                dim_list = list(variable_dim[0]) if len(variable_dim) == 1 else list(sets(*variable_dim))
                return {key: _add_var(model_object, f"{variable_name}{key}", vtype, LO if lo_val >= 0 else RA, lo_val, ub) for key in dim_list}

        case 'bvar' | 'btvar':
            vtype = INT
            if variable_dim == 0:
                return _add_var(model_object, variable_name, vtype, RA, 0, 1)
            else:
                dim_list = list(variable_dim[0]) if len(variable_dim) == 1 else list(sets(*variable_dim))
                return {key: _add_var(model_object, f"{variable_name}{key}", vtype, RA, 0, 1) for key in dim_list}

        case 'ivar' | 'itvar':
            vtype = INT
            bk, bl, bu = _bound_key(lb, ub, lb)
            if variable_dim == 0:
                return _add_var(model_object, variable_name, vtype, bk, bl, bu)
            else:
                dim_list = list(variable_dim[0]) if len(variable_dim) == 1 else list(sets(*variable_dim))
                return {key: _add_var(model_object, f"{variable_name}{key}", vtype, bk, bl, bu) for key in dim_list}

        case 'fvar' | 'ftvar':
            vtype = CONT
            bk, bl, bu = _bound_key(lb, ub, lb)
            if variable_dim == 0:
                return _add_var(model_object, variable_name, vtype, bk, bl, bu)
            else:
                dim_list = list(variable_dim[0]) if len(variable_dim) == 1 else list(sets(*variable_dim))
                return {key: _add_var(model_object, f"{variable_name}{key}", vtype, bk, bl, bu) for key in dim_list}
