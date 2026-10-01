# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast

import pymprog as pymprog_interface

pymprog_status_dict = {5: "optimal"}


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


def _resolve_variable(model_object, result, input2):
    variables = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
    if isinstance(input2, str):
        if input2 in variables:
            return variables[input2]
        base, key = _split_query(input2)
        candidate = variables.get(base)
        if key is None:
            return candidate
        if isinstance(candidate, dict):
            return candidate.get(key)
        return None
    try:
        _ = input2.primal
        return input2
    except Exception:
        return input2


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            var = _resolve_variable(model_object, result, input2)
            if var is not None:
                try:
                    return var.primal
                except Exception:
                    return None
            return None

        case 'status':

            return pymprog_status_dict.get(pymprog_interface.status(), 'Not Optimal')

        case 'objective':

            return pymprog_interface.vobj()

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return None

        case 'dual':

            return None

        case 'slack':

            return None
