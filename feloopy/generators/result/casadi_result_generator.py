# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

CASADI_STATUS_MAP = {
    'Solve_Succeeded': 'optimal',
    'Infeasible_Problem_Detected': 'infeasible',
    'Maximum_Iterations_Exceeded': 'maximum iterations exceeded',
    'Maximum_CpuTime_Exceeded': 'time limit',
    'Restoration_Failure': 'restoration failure',
    'Diverging_Iterates': 'diverging iterates',
    'Not_Enough_Degrees_Of_Freedom': 'not enough degrees of freedom',
    'Invalid_Problem_Definition': 'invalid problem definition',
    'Unrecoverable_Exception': 'unrecoverable exception',
    'Insufficient_Memory': 'insufficient memory',
}

import ast


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


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':
            variables = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
            if isinstance(input2, str):
                if input2 in variables:
                    entry = variables[input2]
                else:
                    base, key = _split_query(input2)
                    candidate = variables.get(base)
                    if key is None:
                        entry = candidate
                    elif isinstance(candidate, dict):
                        entry = candidate.get(key)
                    else:
                        entry = None
                if entry is None:
                    return None
                return float(result[0].value(entry))
            else:
                return float(result[0].value(input2))

        case 'status':
            raw = model_object.return_status()
            return CASADI_STATUS_MAP.get(raw, raw)

        case 'objective':
            return float(result[0].value(model_object.f))

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            return None

        case 'dual':
            return None

        case 'slack':
            return None
