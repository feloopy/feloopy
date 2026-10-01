# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import ast
import math as mt

from ..variable.gamspy_variable_generator import _fmt, gams_safe_name


def _split_query(query):
    """Split 'name[0]' into ('name', 0) and 'name[(0, 0)]' into ('name', (0, 0)).

    Plain scalar names are returned unchanged with a ``None`` key.
    """
    if '[' not in query or not query.endswith(']'):
        return query, None
    base, inner = query.rsplit('[', 1)
    inner = inner[:-1]
    try:
        key = ast.literal_eval(f'({inner},)')
        if len(key) == 1:
            key = key[0]
    except (ValueError, SyntaxError):
        key = inner
    return base, key


def _lookup_variable(container, name_or_var):
    if hasattr(name_or_var, 'toValue'):
        return name_or_var
    variables = list(container.getVariables())
    if isinstance(name_or_var, str):
        for v in variables:
            if v.name == name_or_var:
                return v
        base, key = _split_query(name_or_var)
        candidates = []
        if key is None:
            candidates.append(gams_safe_name(name_or_var))
        else:
            candidates.append(_fmt(gams_safe_name(base), key))
            candidates.append(_fmt(base, key))
        for cand in candidates:
            for v in variables:
                if v.name == cand:
                    return v
    raise KeyError(f"Variable '{name_or_var}' not found in GAMSPy Container")


def _lookup_equation(equation_dict, name_or_eq):
    if name_or_eq in equation_dict:
        return equation_dict[name_or_eq]
    if isinstance(name_or_eq, str) and gams_safe_name(name_or_eq) in equation_dict:
        return equation_dict[gams_safe_name(name_or_eq)]
    raise KeyError(f"Equation '{name_or_eq}' not found in equation dict")


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]
    equation_dict = result[0][0]
    objective_variable = result[0][1]
    gamspy_model = result[0][2]

    match input1:

        case 'variable':
            var = _lookup_variable(model_object, input2)
            return var.toValue()

        case 'status':
            return gamspy_model.status

        case 'objective':
            return objective_variable.toValue()

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            try:
                return gamspy_model.objective_estimation
            except Exception:
                return None

        case 'dual':
            eq = _lookup_equation(equation_dict, input2)
            try:
                return eq.records['marginal'].item()
            except Exception:
                return None

        case 'slack':
            eq = _lookup_equation(equation_dict, input2)
            try:
                upper = eq.records['upper'].item()
                lower = eq.records['lower'].item()
                level = eq.records['level'].item()
                if upper != mt.inf:
                    return upper - level
                if lower != mt.inf:
                    return level - lower
                return None
            except Exception:
                return None
