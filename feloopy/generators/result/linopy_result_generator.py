# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import sys
import re
import numpy as np

def _import_linopy_quietly():
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    saved_stdout_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        os.dup2(devnull, stdout_fd)
        from linopy import Model as LINOPYMODEL
        from linopy import LinearExpression
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(devnull)
    return LINOPYMODEL, LinearExpression

LINOPYMODEL, LinearExpression = _import_linopy_quietly()


def Get(model_object, result, input1, input2=None):

    directions = +1 if input1[1][input1[2]] == 'min' else -1

    input1 = input1[0]

    match input1:

        case 'variable':
            if input2 is None:
                return None
            target = str(input2)
            bracket_match = re.match(r'^(.*?)\[(.+)\]$', target)
            if bracket_match:
                var_name = bracket_match.group(1)
                idx_str = bracket_match.group(2)
                # queries use Python tuple formatting ("x[(0, 0)]")
                if idx_str.startswith('(') and idx_str.endswith(')'):
                    idx_str = idx_str[1:-1]
                idx_parts = [x.strip() for x in idx_str.split(',')]
            else:
                var_name = target
                idx_parts = None
            try:
                solution = model_object.solution
                if var_name in solution:
                    val = solution[var_name]
                    if idx_parts is not None and hasattr(val, 'values'):
                        vals = val.values
                        def _parse_linopy_idx(v):
                            try:
                                return int(v)
                            except (ValueError, TypeError):
                                return v.strip("'\"")
                        parsed = [_parse_linopy_idx(x) for x in idx_parts]
                        if vals.ndim == 1:
                            return float(vals[parsed[0]])
                        elif vals.ndim == 2:
                            return float(vals[parsed[0], parsed[1]])
                        elif vals.ndim == 3:
                            return float(vals[parsed[0], parsed[1], parsed[2]])
                    if hasattr(val, 'values'):
                        v = val.values
                        if isinstance(v, np.ndarray) and v.ndim == 0:
                            return float(v)
                        elif isinstance(v, np.ndarray) and v.size == 1:
                            return float(v.flat[0])
                        return v
                    return val
            except Exception:
                pass
            return None

        case 'status':

            return result[0][1]

        case 'objective':
            try:
                solution = model_object.solution
                obj_expr = model_object.objective.expression
                if hasattr(obj_expr, 'vars') and hasattr(obj_expr, 'coeffs'):
                    total = 0.0
                    vars_flat = obj_expr.vars.values.flatten()
                    coeffs_flat = obj_expr.coeffs.values.flatten()
                    for var_idx, coeff in zip(vars_flat, coeffs_flat):
                        if var_idx < 0:
                            continue
                        for var_name in solution:
                            var_data = solution[var_name]
                            vals = var_data.values
                            if vals.ndim == 0:
                                if int(var_idx) == list(solution.keys()).index(var_name):
                                    total += float(coeff) * float(vals)
                            else:
                                flat_idx = int(var_idx)
                                if flat_idx < vals.size:
                                    total += float(coeff) * float(vals.flat[flat_idx])
                    return total
            except Exception:
                pass
            return None

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            try:
                return result[0][0].report.dual_bound
            except:
                return None

        case 'dual':
            if input2 is None:
                return None
            try:
                target = str(input2)
                if target in model_object.constraints:
                    con = model_object.constraints[target]
                    return float(con.dual.values)
            except Exception:
                pass
            return None

        case 'slack':
            if input2 is None:
                return None
            try:
                target = str(input2)
                if target in model_object.constraints:
                    con = model_object.constraints[target]
                    rhs_val = float(con.rhs.values)
                    sign_val = str(con.sign.values)
                    coeffs = con.coeffs.values
                    var_indices = con.vars.values
                    solution = model_object.solution
                    lhs_val = 0.0
                    for coeff, var_idx in zip(coeffs.flat, var_indices.flat):
                        if var_idx < 0:
                            continue
                        for var_name in solution:
                            var_data = solution[var_name]
                            if hasattr(var_data, 'values'):
                                vals = var_data.values
                                if vals.ndim == 0:
                                    if int(var_idx) == list(solution.keys()).index(var_name):
                                        lhs_val += float(coeff) * float(vals)
                                else:
                                    flat_idx = int(var_idx)
                                    if flat_idx < vals.size:
                                        lhs_val += float(coeff) * float(vals.flat[flat_idx])
                    if sign_val == '<=':
                        return rhs_val - lhs_val
                    elif sign_val == '>=':
                        return lhs_val - rhs_val
                    else:
                        return 0.0
            except Exception:
                pass
            return None
