# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ...helpers.formatter import *

SEEKER_STATUS_MAP = {
    6: 'optimal',
    0: 'unoptimized',
    2: 'infeasible',
    3: 'infeasible',
    1: 'malformed',
    5: 'bounded',
    4: 'time limit',
}


def _get_variable_value(model_object, variable):
    if variable is None:
        return None
    try:
        return model_object.evaluate(variable)
    except Exception:
        try:
            return variable.get_value()
        except Exception:
            return None


def Get(model_object, result, input1, input2=None):
    input1 = input1[0]

    match input1:

        case 'variable':
            return _get_variable_value(model_object, input2)

        case 'status':
            try:
                status = model_object.get_status()
                status_val = status.value if hasattr(status, 'value') else int(status)
                return SEEKER_STATUS_MAP.get(status_val, 'unknown')
            except Exception:
                return 'unknown'

        case 'objective':
            return result[0].get_value() if hasattr(result[0], 'get_value') else result[0]

        case 'time':
            return result[1][1] - result[1][0]

        case 'bound':
            return None

        case 'dual':
            try:
                lp = model_object.lp()
                if lp is not None:
                    duals = lp.get_row_duals()
                    if isinstance(input2, int) and 0 <= input2 < len(duals):
                        return duals[input2]
                    return duals
            except Exception:
                pass
            return None

        case 'slack':
            return None

        case 'iis':
            return None

        case 'mipgap':
            return None

        case 'solcount':
            try:
                status = model_object.get_status()
                status_val = status.value if hasattr(status, 'value') else int(status)
                return 1 if status_val == 6 else 0
            except Exception:
                return 0

        case 'rc':
            return None

        case 'feasibility':
            try:
                return {
                    'max_vio': None,
                    'bound_vio': None,
                    'constr_vio': None,
                    'int_vio': None,
                    'compl_vio': None,
                }
            except Exception:
                return None

        case 'sensitivity_obj':
            return None, None

        case 'sensitivity_bound':
            return None, None, None, None

        case 'basis':
            return None

        case 'constrbasis':
            return None

        case 'pool':
            return None

        case 'pool_obj':
            return None

        case 'objn':
            return None
