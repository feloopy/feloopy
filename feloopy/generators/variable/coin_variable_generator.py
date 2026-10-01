# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
import math

from pyomo.environ import (
    Var, NonNegativeReals, NonNegativeIntegers, Binary, Integers, Any, ConcreteModel, value, Reals
)

from ...helpers._pyomo_types import register_pyomo_numeric_types

register_pyomo_numeric_types()

sets = it.product


def _initial_value(lb, ub, integer=False):
    """Pick a starting point inside the bounds, away from exactly zero.

    Bonmin and Couenne run Ipopt on the NLP relaxation, and Ipopt
    evaluates expression derivatives at the starting point. Terms such as
    ``x ** 0.5`` have no derivative at zero (``pow'(0, 0.5)``), so the
    default all-zero start aborts the solve before the first iteration.
    Starting at a non-degenerate interior point keeps those gradients
    well defined; the solver still drives the variables to their optimal
    values.
    """
    if lb is not None and math.isfinite(lb) and ub is not None and math.isfinite(ub):
        v = lb if lb >= ub else (lb + ub) / 2.0
    elif lb is not None and math.isfinite(lb):
        v = lb + 1.0
    elif ub is not None and math.isfinite(ub):
        v = ub - 1.0
    else:
        v = 1.0

    if v == 0.0:
        if ub is not None and math.isfinite(ub) and ub > 0:
            v = 0.5 if ub >= 0.5 else ub / 2.0
        elif lb is not None and math.isfinite(lb) and lb < 0:
            v = -0.5 if lb <= -0.5 else lb / 2.0

    if integer:
        lo = math.ceil(lb) if lb is not None and math.isfinite(lb) else None
        hi = math.floor(ub) if ub is not None and math.isfinite(ub) else None
        iv = int(round(v))
        if lo is not None and iv < lo:
            iv = lo
        if hi is not None and iv > hi:
            iv = hi
        if iv == 0:
            if hi is None or hi >= 1:
                iv = 1
            elif lo is not None and lo <= -1:
                iv = -1
        v = float(iv)

    return v


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    lb = variable_bound[0]
    ub = variable_bound[1]

    start_p = _initial_value(0.0 if lb is None else max(lb, 0.0), ub)
    start_i = _initial_value(lb, ub, integer=True)
    start_f = _initial_value(lb, ub)
    start_b = _initial_value(0.0, 1.0, integer=True)

    match variable_type:

        case 'pvar':

            if variable_dim == 0:
                var = Var(within=NonNegativeReals, bounds=(lb, ub), initialize=start_p)
                var.construct()
                model_object.add_component(variable_name, var)
                return var

            else:
                if len(variable_dim) == 1:
                    indices = list(variable_dim[0])
                else:
                    indices = list(sets(*variable_dim))
                var = Var(indices, within=NonNegativeReals, bounds=(lb, ub), initialize=start_p)
                var.construct()
                model_object.add_component(variable_name, var)
                if len(indices) == 1:
                    return {indices[0]: var[indices[0]]}
                return {key: var[key] for key in indices}

        case 'bvar':

            if variable_dim == 0:
                var = Var(within=Binary, initialize=start_b)
                var.construct()
                model_object.add_component(variable_name, var)
                return var

            else:
                if len(variable_dim) == 1:
                    indices = list(variable_dim[0])
                else:
                    indices = list(sets(*variable_dim))
                var = Var(indices, within=Binary, initialize=start_b)
                var.construct()
                model_object.add_component(variable_name, var)
                if len(indices) == 1:
                    return {indices[0]: var[indices[0]]}
                return {key: var[key] for key in indices}

        case 'ivar':

            if variable_dim == 0:
                var = Var(within=Integers, bounds=(lb, ub), initialize=start_i)
                var.construct()
                model_object.add_component(variable_name, var)
                return var

            else:
                if len(variable_dim) == 1:
                    indices = list(variable_dim[0])
                else:
                    indices = list(sets(*variable_dim))
                var = Var(indices, within=Integers, bounds=(lb, ub), initialize=start_i)
                var.construct()
                model_object.add_component(variable_name, var)
                if len(indices) == 1:
                    return {indices[0]: var[indices[0]]}
                return {key: var[key] for key in indices}

        case 'fvar':

            if variable_dim == 0:
                var = Var(within=Reals, bounds=(lb, ub), initialize=start_f)
                var.construct()
                model_object.add_component(variable_name, var)
                return var

            else:
                if len(variable_dim) == 1:
                    indices = list(variable_dim[0])
                else:
                    indices = list(sets(*variable_dim))
                var = Var(indices, within=Reals, bounds=(lb, ub), initialize=start_f)
                var.construct()
                model_object.add_component(variable_name, var)
                if len(indices) == 1:
                    return {indices[0]: var[indices[0]]}
                return {key: var[key] for key in indices}

    return None
