# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast
import re


def _parse_var_string(var_str):
    match = re.match(r'^(.+)\[(.+)\]$', var_str)
    if match:
        comp_name, idx_str = match.group(1), match.group(2)
        try:
            idx = ast.literal_eval(f'({idx_str},)')
            if len(idx) == 1:
                idx = idx[0]
        except (ValueError, SyntaxError):
            idx = idx_str
        return comp_name, idx
    return var_str, None


def Get(model_object, result, input1, input2=None):
    input1 = input1[0]

    uno_result = result[0]

    match input1:
        case 'variable':
            if hasattr(input2, 'index') and not isinstance(input2, str):
                return uno_result.primal_solution[input2.index]
            elif isinstance(input2, int):
                return uno_result.primal_solution[input2]
            elif isinstance(input2, str):
                comp_name, idx = _parse_var_string(input2)
                for key, var in result[2].items():
                    if key[1] == comp_name:
                        if idx is not None and isinstance(var, dict):
                            if idx in var:
                                return uno_result.primal_solution[var[idx].index]
                            try:
                                idx_alt = ast.literal_eval(str(idx))
                            except (ValueError, SyntaxError):
                                idx_alt = idx
                            if idx_alt in var:
                                return uno_result.primal_solution[var[idx_alt].index]
                            return None
                        elif hasattr(var, 'index'):
                            return uno_result.primal_solution[var.index]
                        elif isinstance(var, dict):
                            vals = []
                            for k in sorted(var.keys()):
                                vals.append(uno_result.primal_solution[var[k].index])
                            return vals
                return None
            return None

        case 'status':
            status_enum_name = getattr(uno_result.optimization_status, 'name', str(uno_result.optimization_status)).lower()
            status_map = {
                'success': 'optimal',
                'infeasible': 'infeasible',
                'unbounded': 'unbounded',
                'max_iterations': 'iteration limit',
                'time_limit': 'time limit',
                'algorithmic_error': 'error',
                'evaluation_error': 'error',
                'not_optimal': 'not optimal',
            }
            return status_map.get(status_enum_name, status_enum_name)

        case 'objective':
            return uno_result.solution_objective

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            return None

        case 'dual':
            return None

        case 'slack':
            return None
