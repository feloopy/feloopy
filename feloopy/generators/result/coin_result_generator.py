# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast
import re
import logging
from pyomo.environ import value as pyomo_value

logging.getLogger('pyomo').setLevel(logging.CRITICAL)


def _parse_pyomo_var(model_object, var_str):
    match = re.match(r'^(.+)\[(.+)\]$', var_str)
    if match:
        comp_name = match.group(1)
        index_str = match.group(2)
        comp = getattr(model_object, comp_name)
        try:
            index = ast.literal_eval(f'({index_str},)')
            if len(index) == 1:
                index = index[0]
        except (ValueError, SyntaxError):
            index = index_str
        return comp[index]
    else:
        return getattr(model_object, var_str)


def Get(model_object, result, input1, input2=None):
    input1 = input1[0]

    pyomo_result = result[0]

    match input1:
        case 'variable':
            if isinstance(input2, str):
                try:
                    var = _parse_pyomo_var(model_object, input2)
                    if var.is_constant():
                        return float(var.value)
                    if var.value is not None:
                        return var.value
                    return None
                except Exception:
                    return None
            elif hasattr(input2, 'value'):
                try:
                    return pyomo_value(input2)
                except Exception:
                    return input2.value
            return None

        case 'status':
            try:
                status = pyomo_result.solver.status
                tc = pyomo_result.solver.termination_condition
                status_str = str(status).lower()
                if 'ok' in status_str or 'optimal' in str(tc).lower():
                    return 'optimal'
                elif 'infeasible' in str(tc).lower():
                    return 'infeasible'
                else:
                    return str(tc)
            except Exception:
                return 'unknown'

        case 'objective':
            try:
                obj_name = '_felooopy_obj'
                obj = getattr(model_object, obj_name)
                return pyomo_value(obj)
            except Exception:
                return None

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            return None

        case 'dual':
            return None

        case 'slack':
            return None
