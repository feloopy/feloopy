# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import sys

import highspy as highs_interface
from ...helpers.formatter import *

highs_model_status_dict = {
    highs_interface.HighsModelStatus.kOptimal: 'optimal',
    highs_interface.HighsModelStatus.kInfeasible: 'infeasible',
    highs_interface.HighsModelStatus.kUnbounded: 'unbounded',
    highs_interface.HighsModelStatus.kUnboundedOrInfeasible: 'infeasible or unbounded',
    highs_interface.HighsModelStatus.kUnknown: 'unknown',
    highs_interface.HighsModelStatus.kNotset: 'not set',
    highs_interface.HighsModelStatus.kModelEmpty: 'model empty',
    highs_interface.HighsModelStatus.kModelError: 'model error',
    highs_interface.HighsModelStatus.kPresolveError: 'presolve error',
    highs_interface.HighsModelStatus.kSolveError: 'solve error',
    highs_interface.HighsModelStatus.kPostsolveError: 'postsolve error',
    highs_interface.HighsModelStatus.kTimeLimit: 'time limit',
    highs_interface.HighsModelStatus.kIterationLimit: 'iteration limit',
    highs_interface.HighsModelStatus.kSolutionLimit: 'solution limit',
    highs_interface.HighsModelStatus.kObjectiveBound: 'objective bound',
    highs_interface.HighsModelStatus.kObjectiveTarget: 'objective target',
    highs_interface.HighsModelStatus.kMemoryLimit: 'memory limit',
    highs_interface.HighsModelStatus.kInterrupt: 'interrupt',
    highs_interface.HighsModelStatus.kLoadError: 'load error',
}

def _variable_names(model_object):
    try:
        num_col = model_object.getNumCol()
        cached = getattr(model_object, '_flp_var_names', None)
        if cached is not None and cached[0] == num_col:
            return cached[1], cached[2]
        names = model_object.allVariableNames()
        index = {}
        for i, name in enumerate(names):
            if name not in index:  # first match wins, like the scan below
                index[name] = i
        try:
            model_object._flp_var_names = (num_col, names, index)
        except AttributeError:
            pass
        return names, index
    except AttributeError:
        # not a highs model object: keep the previous behaviour
        names = model_object.allVariableNames()
        return names, None


def Get(model_object, result, input1, input2=None, input=None):
    input1 = input1[0]

    match input1:
        case 'variable':
            if isinstance(input2, str):
                col_values = model_object.getSolution().col_value
                names, name_index = _variable_names(model_object)
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
                if name_index is not None:
                    i = name_index.get(target)
                    if i is None:
                        return None
                    return col_values[i]
                for i, name in enumerate(names):
                    if name == target:
                        return col_values[i]
                return None
            return model_object.val(input2)
        
        case 'status':
            model_status = model_object.getModelStatus()
            return highs_model_status_dict.get(model_status, str(model_status))

        case 'objective':
            model_status = model_object.getModelStatus()
            if model_status in (
                highs_interface.HighsModelStatus.kInfeasible,
                highs_interface.HighsModelStatus.kUnbounded,
                highs_interface.HighsModelStatus.kUnboundedOrInfeasible,
                highs_interface.HighsModelStatus.kNotset,
                highs_interface.HighsModelStatus.kModelEmpty,
                highs_interface.HighsModelStatus.kModelError,
                highs_interface.HighsModelStatus.kPresolveError,
                highs_interface.HighsModelStatus.kSolveError,
                highs_interface.HighsModelStatus.kPostsolveError,
                highs_interface.HighsModelStatus.kLoadError,
            ):
                return None
            return model_object.getObjectiveValue()

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            if input is not None:
                has_mip = any(c > 0 for c in [
                    input.get('integer_variable_counter', [0])[0],
                    input.get('binary_variable_counter', [0])[0],
                ])
                if not has_mip:
                    return None
            try:
                status, val = model_object.getInfoValue('mip_dual_bound')
                if status == highs_interface.HighsStatus.kOk:
                    return val
            except:
                pass
            return None

        case 'ogr':
            try:
                has_mip = False
                if input is not None:
                    has_mip = any(c > 0 for c in [
                        input.get('integer_variable_counter', [0])[0],
                        input.get('binary_variable_counter', [0])[0],
                    ])
                if not has_mip:
                    model_status = model_object.getModelStatus()
                    if model_status == highs_interface.HighsModelStatus.kOptimal:
                        return 0.0
                    return None
                status, val = model_object.getInfoValue('mip_dual_bound')
                if status != highs_interface.HighsStatus.kOk:
                    return None
                bound = val
                obj = model_object.getObjectiveValue()
                if obj is None or bound is None:
                    return None
                if obj == float('inf') or obj == float('-inf'):
                    return None
                if bound == float('inf') or bound == float('-inf'):
                    return None
                abs_obj = abs(obj)
                abs_bound = abs(bound)
                denom = max(abs_obj, abs_bound, 1.0)
                return abs(obj - bound) / denom
            except:
                return None

        case 'dual':
            if input2 is None:
                return None
            try:
                sol = model_object.getSolution()
                row_duals = sol.row_dual
                lp_data = (input or {}).get('lp_data', {})
                row_names = lp_data.get('row_names', [])
                target = str(input2)
                for i, name in enumerate(row_names):
                    if name == target:
                        if i < len(row_duals):
                            return row_duals[i]
                        return None
                return None
            except Exception:
                return None

        case 'slack':
            if input2 is None:
                return None
            sol = model_object.getSolution()
            row_values = sol.row_value
            lp_data = (input or {}).get('lp_data', {})
            row_names = lp_data.get('row_names', [])
            row_lower = lp_data.get('row_lower', [])
            row_upper = lp_data.get('row_upper', [])
            target = str(input2)
            for i, name in enumerate(row_names):
                if name == target:
                    ax = row_values[i]
                    lb = row_lower[i] if i < len(row_lower) else float('-inf')
                    ub = row_upper[i] if i < len(row_upper) else float('inf')
                    upper_slack = ub - ax if ub < float('inf') else float('inf')
                    lower_slack = ax - lb if lb > float('-inf') else float('inf')
                    return min(upper_slack, lower_slack)
            return None