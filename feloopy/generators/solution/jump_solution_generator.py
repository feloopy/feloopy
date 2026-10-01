# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
import os

import numpy as np

from ..variable.jump_expression import JumpVar, JumpExpr, JumpConstr, _expr_to_str, _is_jump_obj, jl_safe_name

jump_solver_selector = {
    'cbc': 'Cbc',
    'glpk': 'GLPK',
    'clp': 'Clp',
    'cplex': 'CPLEX',
    'gurobi': 'Gurobi',
    'highs': 'HiGHS',
    'ipopt': 'Ipopt',
    'knitro': 'Knitro',
    'mosek': 'Mosek',
    'scip': 'SCIP',
    'xpress': 'Xpress',
    'osicbc': 'OSICbc',
    'osiglpk': 'OSIGLPK',
    'cosmo': 'COSMO',
    'path': 'PATH',
    'profound': 'Proximal',
    'alpine': 'Alpine.jl',
    'artelys_knitro': 'Artelys Knitro',
    'baron': 'BARON',
    'bonmin': 'Bonmin',
    'cdc': 'CDCS',
    'cdd': 'CDD',
    'clarabel': 'Clarabel.jl',
    'copt': 'COPT',
    'couenne': 'Couenne',
    'csdp': 'CSDP',
    'daqp': 'DAQP',
    'dsdp': 'DSDP',
    'eago': 'EAGO.jl',
    'ecos': 'ECOS',
    'fico_xpress': 'FICO Xpress',
    'hypatia': 'Hypatia.jl',
    'juniper': 'Juniper.jl',
    'loraine': 'Loraine.jl',
    'madnlp': 'MadNLP.jl',
    'maingo': 'MAiNGO',
    'manopt': 'Manopt.jl',
    'minotaur': 'Minotaur',
    'minizinc': 'MiniZinc',
    'nlopt': 'NLopt',
    'octeract': 'Octeract',
    'optim': 'Optim.jl',
    'osqp': 'OSQP',
    'pajarito': 'Pajarito.jl',
    'pavito': 'Pavito.jl',
    'penbmi': 'Penbmi',
    'percival': 'Percival.jl',
    'polyjump_kkt': 'PolyJuMP.KKT',
    'polyjump_qcqp': 'PolyJuMP.QCQP',
    'raposa': 'RAPOSa',
    'scs': 'SCS',
    'sdpa': 'SDPA',
    'sdplr': 'SDPLR',
    'sdpnal': 'SDPNAL',
    'sdpt3': 'SDPT3',
    'sedumi': 'SeDuMi',
    'status_switching_qp': 'StatusSwitchingQP.jl',
    'tulip': 'Tulip.jl'
}


def _var_declaration(value):
    if isinstance(value, JumpVar):
        return value.declaration
    return str(value)


def _obj_str(expression):
    if _is_jump_obj(expression):
        return _expr_to_str(expression)
    return str(expression)


def _constr_str(expression):
    if _is_jump_obj(expression):
        return _expr_to_str(expression)
    return str(expression)


def generate_solution(features):
    from ...helpers.julia_server import get_jl, jl_eval_silent, jl_eval_streaming

    jl = get_jl()

    solver_name = features['solver_name']

    if solver_name not in jump_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'jump'! \nPossible fixes: "
            "\n1) Check the solver name. \n2) Use another interface. " % solver_name)

    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    objective_id = features['objective_being_optimized']
    log = features['log']
    solver_options = features['solver_options']

    model_code = ""
    model_code += "\nusing " + jump_solver_selector[solver_name]
    model_code += features.get("jlcode_preamble", "")
    model_code += features.get("jlcode_data", "")
    model_code += features['model_object_before_solve']

    model_code += features.get("jlcode_before_variables", "")
    for value in features['variables'].values():
        model_code += _var_declaration(value)

    if time_limit is not None:
        model_code += f'\nset_optimizer_attribute(jlmodel, "time_limit", {time_limit})'

    if thread_count is not None:
        model_code += f'\nset_optimizer_attribute(jlmodel, "threads", {thread_count})'

    if relative_gap is not None:
        model_code += f'\nset_optimizer_attribute(jlmodel, "mip_gap", {relative_gap})'

    if absolute_gap is not None:
        model_code += f'\nset_optimizer_attribute(jlmodel, "mip_gap_abs", {absolute_gap})'

    if log:
        model_code += '\nset_optimizer_attribute(jlmodel, "output_flag", true)'
    else:
        model_code += '\nset_optimizer_attribute(jlmodel, "output_flag", false)'

    for key, val in solver_options.items():
        if key.startswith("---"):
            continue
        if val is None:
            continue
        if isinstance(val, bool):
            val_str = "true" if val else "false"
        elif isinstance(val, str):
            val_str = f'"{val}"'
        else:
            val_str = str(val)
        model_code += f'\nset_optimizer_attribute(jlmodel, "{key}", {val_str})'

    model_code += features.get("jlcode_before_objectives", "")
    match directions[objective_id]:
        case 'min':
            model_code += f"\n@objective(jlmodel, Min, {_obj_str(model_objectives[objective_id])})"
        case 'max':
            model_code += f"\n@objective(jlmodel, Max, {_obj_str(model_objectives[objective_id])})"

    model_code += features.get("jlcode_before_constraints", "")
    counter = 0
    for constraint in model_constraints:
        label = constraint_labels[counter]
        expr_str = _constr_str(constraint)
        if label is None:
            model_code += f"\n@constraint(jlmodel, c{counter + 1}, {expr_str})"
        else:
            model_code += f"\n@constraint(jlmodel, {jl_safe_name(label)}, {expr_str})"
        counter += 1

    for jl_name, fixed_value in (features.get('_jump_fix') or {}).items():
        model_code += f"\nfix({jl_name}, {fixed_value!r}; force = true)"

    for jl_name, start_value in (features.get('_jump_start') or {}).items():
        model_code += f"\nset_start_value({jl_name}, {start_value!r})"

    model_code += f"\nset_optimizer(jlmodel, {jump_solver_selector[solver_name]}.Optimizer)"
    model_code += "\nelapsed_time = @elapsed begin"
    model_code += "\n  optimize!(jlmodel)"
    model_code += "\nend"

    for key in features['variables']:
        jl_name = jl_safe_name(key[1])
        model_code += f"\n{jl_name} = value.({jl_name})"

    model_code += features.get("jlcode_before_solve", "")
    if log:
        jl_eval_streaming(jl, model_code)
    else:
        jl_eval_silent(jl, model_code)
    jl.seval(features.get("jlcode_after_solve", ""))

    result = {}
    status = str(jl.termination_status(jl.jlmodel))
    result["status"] = status

    if "optimal" in status.lower() or "feasible" in status.lower():
        objective_value = float(jl.objective_value(jl.jlmodel))
        solutions = {}
        for key in features['variables']:
            name = key[1]
            val = getattr(jl, jl_safe_name(name))
            try:
                solutions[name] = np.array(val)
            except Exception:
                solutions[name] = val
        result["objective_value"] = objective_value
        result["solutions"] = solutions

        try:
            dual = {}
            for label in constraint_labels:
                if label is not None:
                    dual[label] = float(jl.shadow_price(getattr(jl, jl_safe_name(label))))
            for label_idx in range(len(constraint_labels)):
                if constraint_labels[label_idx] is None:
                    auto_label = f"c{label_idx + 1}"
                    dual[auto_label] = float(jl.shadow_price(getattr(jl, auto_label)))
            result["dual"] = dual
        except Exception:
            pass

    generated_solution = [result, [0, float(jl.elapsed_time)]]

    file_path = './__pycache__/data.json'
    if os.path.exists(file_path):
        os.remove(file_path)
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.listdir(dir_path):
            os.rmdir(dir_path)

    return generated_solution
