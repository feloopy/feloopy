# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast


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


def _resolve_query(features, query):
    base, key = _split_query(query)
    for (_vtype, vname), entry in (features.get('variables') or {}).items():
        if vname != base:
            continue
        if isinstance(entry, dict):
            return entry.get(key) if key is not None else None
        return entry if key is None else None
    return None


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            if isinstance(input2, str):
                if not isinstance(model_object, dict):
                    return None
                input2 = _resolve_query(model_object, input2)
                if input2 is None:
                    return None

            val = getattr(input2, 'value', None)
            if val is None:
                return None
            if len(val) == 1:

                return val[0]

            else:
                return val

        case 'status':

            return result[0][0][0].status

        case 'objective':

            return result[0][0][0].value

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return None
        
        case 'dual':

            if len(result[0][1][input2].dual_value)==1:

                return result[0][1][input2].dual_value[0]
            else:

                return result[0][1][input2].dual_value

        case 'slack':
            
            print('Warning: Slacks are not supported. Expr value is returned.')

            #Needs to be corrected

            if len(result[0][1][input2].dual_value)==1:

                return result[0][1][input2].expr.value[0]
            else:
                
                return result[0][1][input2].expr.value
        