# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast

import pyomo.environ as pyomo_interface


def _split_query(query):
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


def _resolve_variable(model_object, input2):
    if not isinstance(input2, str) and hasattr(input2, 'value'):
        return input2.value
    if not isinstance(input2, str):
        return pyomo_interface.value(input2)
    base, key = _split_query(input2)
    comp = model_object.component(base)
    if comp is not None and key is not None:
        try:
            return pyomo_interface.value(comp[key])
        except Exception:
            return None
    if comp is not None:
        return pyomo_interface.value(comp)
    return None


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            return _resolve_variable(model_object, input2)

        case 'status':

            return result[0].solver.termination_condition

        case 'objective':

            return pyomo_interface.value(model_object.OBJ)

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return None
        
        case 'dual':

            try:
                return model_object.dual[model_object.c[input2]]
            except Exception:
                return None

        case 'slack':

            upper_slack = model_object.c[input2].uslack()
            lower_slack = model_object.c[input2].lslack()

            return min(upper_slack, lower_slack)
        
        case 'rc':
            ""
    