# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ..picat_expression import PicatVar, PicatExpr
from ..variable.picat_variable_generator import _picat_safe_name


def Get(model_object, result, input1, input2=None):
    """Extract results from the Picat solution."""
    thing = input1[0]
    directions = input1[1]
    obj_idx = input1[2]

    status_code = result[0][0]
    parsed_values = result[0][1]
    time_begin = result[1][0]
    time_end = result[1][1]

    match thing:

        case 'variable':

            if status_code != 0:
                return None

            def _picat_lookup(name):
                if name in parsed_values:
                    return parsed_values[name]
                if '[' in name and name.endswith(']'):
                    base, idx = name.rsplit('[', 1)
                    idx = idx.rstrip(']')
                    if idx.startswith('(') and idx.endswith(')'):
                        idx = idx[1:-1]
                    parts = [p.strip() for p in idx.split(',')] if idx else []
                    safe_base = _picat_safe_name(base)
                    candidates = []
                    if parts:
                        candidates.append(f"{safe_base}_{'_'.join(parts)}")
                    candidates.append(f"{safe_base}_{idx}")
                    for b in (base, base[0].upper() + base[1:] if base else base):
                        if parts:
                            candidates.append(f"{b}_{'_'.join(parts)}")
                        candidates.append(f"{b}_{idx}")
                    for cand in candidates:
                        if cand in parsed_values:
                            return parsed_values[cand]
                    return None
                if isinstance(name, str):
                    for cand in (_picat_safe_name(name),
                                 name[0].upper() + name[1:] if name else name):
                        if cand in parsed_values:
                            return parsed_values[cand]
                return None

            if isinstance(input2, dict):
                values = {}
                for key, var in input2.items():
                    name = var.name if hasattr(var, 'name') else str(key)
                    values[key] = _picat_lookup(name)
                return values

            if isinstance(input2, (PicatVar, PicatExpr)):
                name = input2.name if hasattr(input2, 'name') else input2._picat_expr
                return _picat_lookup(name)

            if isinstance(input2, str):
                return _picat_lookup(input2)

            return parsed_values

        case 'status':

            if status_code == 0:
                return 'optimal'
            else:
                return 'infeasible'

        case 'objective':

            if status_code != 0:
                return None

            obj_name = None
            if hasattr(model_object, 'features'):
                features = model_object.features
                objectives = features.get('objectives', [])
                if obj_idx < len(objectives):
                    obj_expr = objectives[obj_idx]
                    if hasattr(obj_expr, '_picat_expr'):
                        obj_name = obj_expr._picat_expr
                    elif isinstance(obj_expr, str):
                        obj_name = obj_expr

            if obj_name and obj_name in parsed_values:
                return parsed_values[obj_name]

            for key, value in parsed_values.items():
                if key.startswith('FELOOPY_OBJ'):
                    return value

            return None

        case 'time':

            return time_end - time_begin

        case 'bound':

            return None

        case 'num_branches':

            return 0

        case 'num_conflicts':

            return 0

        case 'wall_time':

            return time_end - time_begin

        case 'dual':

            return None

        case 'slack':

            return None

    return None
