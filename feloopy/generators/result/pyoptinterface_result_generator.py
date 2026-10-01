# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyoptinterface as poi
from ...helpers.formatter import *
from ...helpers.reporter import left_align


poi_status_dict = {
    poi.TerminationStatusCode.OPTIMAL: 'optimal',
    poi.TerminationStatusCode.INFEASIBLE: 'infeasible',
    poi.TerminationStatusCode.INFEASIBLE_OR_UNBOUNDED: 'infeasible or unbounded',
    poi.TerminationStatusCode.DUAL_INFEASIBLE: 'unbounded',
    poi.TerminationStatusCode.TIME_LIMIT: 'time limit',
    poi.TerminationStatusCode.ITERATION_LIMIT: 'iteration limit',
    poi.TerminationStatusCode.NODE_LIMIT: 'node limit',
    poi.TerminationStatusCode.SOLUTION_LIMIT: 'solution limit',
    poi.TerminationStatusCode.INTERRUPTED: 'interrupted',
    poi.TerminationStatusCode.MEMORY_LIMIT: 'memory limit',
    poi.TerminationStatusCode.NUMERICAL_ERROR: 'numerical',
    poi.TerminationStatusCode.OPTIMIZE_NOT_CALLED: 'loaded',
}


def _resolve_variable(model_object, input2):
    stored = getattr(model_object, '_feloopy_variable_indices', None)

    if isinstance(input2, str):
        import re
        bracket_match = re.match(r'^(.*?)\[(.+)\]$', input2)
        if bracket_match:
            var_name = bracket_match.group(1)
            idx_str = bracket_match.group(2)
            idx_parts = [x.strip().strip('()') for x in idx_str.split(',')]
            def _parse_idx(val):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return val.strip("'\"")
            idx_tuple = tuple(_parse_idx(x) for x in idx_parts if x)
            if len(idx_tuple) == 1:
                target = f"{var_name}{idx_tuple[0]}"
            else:
                target = f"{var_name}{idx_tuple}"
        else:
            target = input2
        if stored and target in stored:
            return stored[target]
        return None
    elif isinstance(input2, (tuple, list)) and len(input2) >= 2:
        var_name = input2[0]
        idx = input2[1]
        if isinstance(idx, (tuple, list)) and len(idx) == 1:
            idx_val = idx[0]
        else:
            idx_val = idx
        for full_name in [f"{var_name}{idx}", f"{var_name}{idx_val}"]:
            result = _resolve_variable(model_object, full_name)
            if result is not None:
                return result
        return None
    elif isinstance(input2, (tuple, list)) and len(input2) == 1:
        return _resolve_variable(model_object, input2[0])
    elif hasattr(input2, 'get_variable_name'):
        return input2
    return input2


def _resolve_constraint(model_object, input2):
    if isinstance(input2, str):
        stored = getattr(model_object, '_feloopy_constraint_indices', None)
        if stored and input2 in stored:
            return stored[input2]
        return None
    return input2


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            try:
                var = _resolve_variable(model_object, input2)
                if var is not None:
                    return model_object.get_value(var)
                return model_object.get_value(input2)
            except Exception:
                return None

        case 'status':

            try:
                raw = model_object.get_model_attribute(poi.ModelAttribute.TerminationStatus)
                return poi_status_dict.get(raw, str(raw))
            except Exception:
                return 'unknown'

        case 'objective':

            try:
                return model_object.get_obj_value()
            except Exception:
                return None

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            try:
                return model_object.get_model_attribute(poi.ModelAttribute.ObjectiveBound)
            except Exception:
                return None

        case 'dual':

            constr = _resolve_constraint(model_object, input2)
            if constr is not None:
                try:
                    return model_object.get_constraint_dual(constr)
                except Exception:
                    return None
            return None

        case 'slack':

            constr = _resolve_constraint(model_object, input2)
            if constr is not None:
                try:
                    activity = model_object.get_constraint_primal(constr)
                    meta = getattr(model_object, '_feloopy_constraint_meta', {}).get(input2, None)
                    if meta is not None and meta['sense'] == poi.Geq:
                        return activity - meta['rhs']
                    elif meta is not None and meta['sense'] == poi.Leq:
                        return meta['rhs'] - activity
                    elif meta is not None and meta['sense'] == poi.Eq:
                        return abs(activity - meta['rhs'])
                    return 0.0
                except Exception:
                    return None
            return None

        case 'rc':

            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return model_object.get_variable_dual(var)
                except Exception:
                    return None
            return None

        case 'iis':

            output = ''
            for sense in [poi.Leq, poi.Geq, poi.Eq]:
                try:
                    nc = model_object.number_of_constraints(sense)
                    for i in range(nc):
                        c = poi.ConstraintIndex(i, sense)
                        try:
                            is_iis = model_object.get_constraint_attribute(poi.ConstraintAttribute.IIS, c)
                            if is_iis:
                                name = model_object.get_constraint_name(c)
                                output += left_align("con: %s" % name, rt=True)
                                output += "\n"
                        except Exception:
                            pass
                except Exception:
                    pass
            return output

        case 'mipgap':

            try:
                return model_object.get_model_attribute(poi.ModelAttribute.RelativeGap)
            except Exception:
                return None

        case 'solcount':

            try:
                return model_object.get_model_attribute(poi.ModelAttribute.NodeCount)
            except Exception:
                return 0

        case 'objn':

            try:
                return model_object.get_obj_value()
            except Exception:
                return None

        case 'sensitivity_obj' | 'sensitivity_bound' | 'basis' | 'constrbasis' | 'pool' | 'pool_obj':

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
