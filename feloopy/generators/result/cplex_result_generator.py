# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ...helpers.formatter import *
from ...helpers.reporter import left_align


def _normalize_name(name):
    return (name.replace('[', '').replace(']', '')
                .replace('(', '').replace(')', '').replace(' ', ''))


def _resolve_variable(model_object, input2):
    if isinstance(input2, str):
        try:
            v = model_object.get_var_by_name(input2)
            if v is not None:
                return v
        except Exception:
            pass
        target = _normalize_name(input2)
        for v in model_object.iter_variables():
            if v.name == input2:
                return v
            if _normalize_name(v.name) == target:
                return v
        return None
    try:
        _ = input2.solution_value
        return input2
    except Exception:
        return input2


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.solution_value
                except Exception:
                    return None
            return None

        case 'status':

            return model_object.solve_details.status

        case 'objective':

            return model_object.objective_value

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            try:
                return model_object.solve_details.best_bound
            except Exception:
                return None

        case 'ogr':

            try:
                bound = model_object.solve_details.best_bound
                obj = model_object.objective_value
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
            except Exception:
                return None

        case 'dual':

            try:
                all_duals = model_object.cplex.solution.get_dual_values()
                for lc in model_object.iter_linear_constraints():
                    if lc.name == input2:
                        return all_duals[lc.index]
            except Exception:
                return None

        case 'slack':

            try:
                all_slacks = model_object.cplex.solution.get_linear_slacks()
                for lc in model_object.iter_linear_constraints():
                    if lc.name == input2:
                        return all_slacks[lc.index]
            except Exception:
                return None

        case 'rc':

            try:
                return input2.reduced_cost
            except Exception:
                return None

        case 'iis':

            try:
                model_object.cplex.solution.advanced.compute_iis()
                output = ''
                for i, c in enumerate(model_object.iter_linear_constraints()):
                    if model_object.cplex.solution.advanced.get_iis(c.index):
                        output += left_align(f"con: {c.name}", rt=True)
                        output += "\n"
                for i, v in enumerate(model_object.iter_variables()):
                    if (model_object.cplex.solution.advanced.get_iis_lower_bound(v) or
                            model_object.cplex.solution.advanced.get_iis_upper_bound(v)):
                        output += left_align(f"var: {v.name}", rt=True)
                        output += "\n"
                return output
            except Exception:
                return ''

        case 'mipgap':

            try:
                return model_object.solve_details.mip_relative_gap
            except Exception:
                return None

        case 'solcount':

            try:
                return model_object.solve_details.processed_nodes
            except Exception:
                return 0

        case 'objn':

            try:
                return model_object.objective_value
            except Exception:
                return None

        case 'sensitivity_obj':

            try:
                all_coeffs = model_object.cplex.solution.advanced.get_reduced_costs()
                var_name = str(input2) if input2 is not None else ''
                for v in model_object.iter_variables():
                    if v.name == var_name:
                        return all_coeffs[v.index], None
            except Exception:
                return None, None

        case 'sensitivity_bound':

            try:
                all_sense = model_object.cplex.solution.advanced.get_objective_coefficient_ranges()
                var_name = str(input2) if input2 is not None else ''
                for v in model_object.iter_variables():
                    if v.name == var_name:
                        lo, up = all_sense[v.index]
                        return lo, up, None, None
            except Exception:
                return None, None, None, None

        case 'basis':

            try:
                var_status = model_object.cplex.solution.basis.get_basis()
                var_name = str(input2) if input2 is not None else ''
                for v in model_object.iter_variables():
                    if v.name == var_name:
                        return var_status[0][v.index]
            except Exception:
                return None

        case 'constrbasis':

            try:
                con_status = model_object.cplex.solution.basis.get_basis()
                con_name = str(input2) if input2 is not None else ''
                for c in model_object.iter_linear_constraints():
                    if c.name == con_name:
                        return con_status[1][c.index]
            except Exception:
                return None

        case 'pool':

            try:
                pool_soln = input2 if input2 is not None else 0
                return model_object.cplex.solution.pool.get_values(pool_soln)
            except Exception:
                return None

        case 'pool_obj':

            try:
                pool_soln = input2 if input2 is not None else 0
                return model_object.cplex.solution.pool.get_objective_value(pool_soln)
            except Exception:
                return None

        case 'feasibility':

            try:
                details = model_object.solve_details
                return {
                    'max_vio': details.max_integer_violations if hasattr(details, 'max_integer_violations') else None,
                    'bound_vio': details.max_bound_violations if hasattr(details, 'max_bound_violations') else None,
                    'constr_vio': details.max_constraint_violations if hasattr(details, 'max_constraint_violations') else None,
                    'int_vio': details.max_integer_violations if hasattr(details, 'max_integer_violations') else None,
                    'compl_vio': None,
                }
            except Exception:
                return None
