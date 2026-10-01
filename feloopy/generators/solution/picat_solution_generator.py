# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
from ..picat_expression import PicatVar, PicatExpr, PicatConstraint
from ..picat_binary import run_picat


def _to_picat_expr(operand):
    """Convert an operand to its Picat expression string."""
    if isinstance(operand, PicatVar):
        return operand._picat_expr
    elif isinstance(operand, PicatExpr):
        return operand._picat_expr
    elif isinstance(operand, PicatConstraint):
        return operand.to_picat()
    elif isinstance(operand, (list, tuple)):
        return '[' + ', '.join(_to_picat_expr(x) for x in operand) + ']'
    elif isinstance(operand, bool):
        return 'true' if operand else 'false'
    elif isinstance(operand, str):
        return operand
    else:
        return str(operand)


def _parse_constraint(constraint):
    """Parse a FelooPy constraint into Picat constraint strings."""
    results = []

    if isinstance(constraint, PicatConstraint):
        left = constraint.left
        op = constraint.op
        right = constraint.right
        results.append(f"({left} {op} {right})")
        return results

    if isinstance(constraint, (list, tuple)):
        if len(constraint) == 3 and isinstance(constraint[1], str):
            lhs, sense, rhs = constraint
            sense_map = {
                '<=': '#<=', '>=': '#>=', '==': '#=',
                '!=': '#!=', '<': '#<', '>': '#>',
            }
            picat_op = sense_map.get(sense, sense)
            lhs_str = _to_picat_expr(lhs)
            rhs_str = _to_picat_expr(rhs)
            results.append(f"({lhs_str} {picat_op} {rhs_str})")
            return results

        if len(constraint) >= 3 and isinstance(constraint[1], str):
            picat_strs = []
            for elem in constraint:
                if isinstance(elem, (list, tuple)):
                    picat_strs.extend(_parse_constraint(elem))
                elif isinstance(elem, PicatConstraint):
                    picat_strs.append(elem.to_picat())
                elif isinstance(elem, str):
                    sense_map = {
                        '<=': '#<=', '>=': '#>=', '==': '#=',
                        '!=': '#!=', '<': '#<', '>': '#>',
                    }
                    picat_strs.append(sense_map.get(elem, elem))
                else:
                    picat_strs.append(_to_picat_expr(elem))
            if len(picat_strs) == 3:
                results.append(f"({picat_strs[0]} {picat_strs[1]} {picat_strs[2]})")
            else:
                results.extend(picat_strs)
            return results

    if isinstance(constraint, str):
        results.append(constraint)
        return results

    return results


def _generate_code(features):
    """Generate Picat source code from the model."""
    model_object = features['model_object_before_solve']
    objectives = features['objectives']
    directions = features['directions']
    constraints = features['constraints']
    constraint_labels = features['constraint_labels']
    time_limit = features.get('time_limit')
    thread_count = features.get('thread_count')
    solver_options = features.get('solver_options', {})
    objective_id = features.get('objective_being_optimized', 0)
    log = features.get('log', False)

    picat_vars = []
    picat_constraints = []
    picat_outputs = []
    interval_defs = []

    for var in model_object.variables:
        lb = var.lb
        ub = var.ub
        name = var.name
        picat_vars.append(f"    {name} :: {lb}..{ub}")

    for var in model_object.variables:
        picat_outputs.append(f'    printf("{var.name}="), write({var.name}), nl')

    for var in model_object.variables:
        model_object.add_output_var(var.name)

    for ci, constraint in enumerate(constraints):
        parsed = _parse_constraint(constraint)
        for p in parsed:
            picat_constraints.append(f"    {p}")

    for ci, constraint_str in enumerate(model_object.constraint_strings):
        picat_constraints.append(f"    {constraint_str}")

    for interval_info in model_object.intervals:
        name = interval_info.get('name', f'intv_{len(interval_defs)}')
        lb = interval_info.get('lb', 0)
        ub = interval_info.get('ub', 10000)
        interval_defs.append(f"    {name} :: {lb}..{ub}")

    var_section = ',\n'.join(picat_vars) if picat_vars else ''
    con_section = ',\n'.join(picat_constraints) if picat_constraints else ''
    intv_section = ',\n'.join(interval_defs) if interval_defs else ''
    out_section = ',\n'.join(picat_outputs) if picat_outputs else ''

    has_objective = len(objectives) > 0 and objective_id < len(objectives)
    enumerate_all = solver_options.get('enumerate', False)

    if has_objective:
        obj_expr = objectives[objective_id]
        if hasattr(obj_expr, '_picat_expr'):
            obj_str = obj_expr._picat_expr
        elif isinstance(obj_expr, (int, float)):
            obj_str = str(obj_expr)
        else:
            obj_str = str(obj_expr)

        direction = directions[objective_id] if objective_id < len(directions) else 'min'

        search_options = []
        if thread_count is not None:
            search_options.append(f"threads={thread_count}")
        for k, v in solver_options.items():
            if k.startswith("---"):
                continue
            if v is None:
                continue
            if k not in ('enumerate', 'log', 'threads', 'debug'):
                search_options.append(f"{k}={v}")

        opt_str = ''
        if direction == 'min':
            opt_str = '$min(' + obj_str + ')'
        elif direction == 'max':
            opt_str = '$max(' + obj_str + ')'

        solve_vars = [v.name for v in model_object.variables]
        solve_vars_str = '[' + ', '.join(solve_vars) + ']' if solve_vars else '[]'

        verb = 'solve_all' if enumerate_all else 'solve'
        if opt_str:
            solve_stmt = f"    {verb}([{opt_str}], {solve_vars_str})"
        else:
            opts = 'ff' if not search_options else ', '.join(search_options)
            solve_stmt = f"    {verb}([{opts}], {solve_vars_str})"
    else:
        solve_vars = [v.name for v in model_object.variables]
        solve_vars_str = '[' + ', '.join(solve_vars) + ']' if solve_vars else '[]'
        verb = 'solve_all' if enumerate_all else 'solve'
        solve_stmt = f"    {verb}({solve_vars_str})"

    obj_outputs = []
    if has_objective:
        for oi in range(len(objectives)):
            oexpr = objectives[oi]
            if hasattr(oexpr, '_picat_expr'):
                ostr = oexpr._picat_expr
            elif isinstance(oexpr, (int, float)):
                ostr = str(oexpr)
            else:
                ostr = str(oexpr)
            obj_outputs.append(f'    printf("FELOOPY_OBJ{oi}="), write({ostr}), nl')

    sections = []
    sections.append("import cp.")
    sections.append("")
    sections.append("main =>")

    if var_section:
        sections.append(var_section + ",")
    if intv_section:
        sections.append(intv_section + ",")
    if con_section:
        sections.append(con_section + ",")
    sections.append(solve_stmt + ",")

    all_outputs = []
    if obj_outputs:
        all_outputs.extend(obj_outputs)
    if out_section:
        all_outputs.append(out_section)

    if all_outputs:
        sections.append('    printf("FELOOPY_SOLUTION\\n"),')
        joined = ',\n'.join(all_outputs)
        sections.append(joined + ",")
        sections.append('    printf("FELOOPY_END\\n").')
    else:
        sections[-1] = solve_stmt + "."

    code = '\n'.join(sections)

    if log:
        print("\n--- Generated Picat Code ---")
        print(code)
        print("-----------------------------\n")

    return code


def _parse_picat_output(stdout):
    """Parse the Picat stdout to extract variable values."""
    values = {}
    in_solution = False

    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line == 'FELOOPY_SOLUTION':
            in_solution = True
            continue
        if line == 'FELOOPY_END':
            break
        if in_solution and '=' in line:
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            try:
                values[key] = int(val)
            except ValueError:
                try:
                    values[key] = float(val)
                except ValueError:
                    values[key] = val

    return values


def generate_solution(features):
    """Generate Picat source code, execute the binary, and parse results."""
    time_limit = features.get('time_limit')
    log = features.get('log', False)

    source_code = _generate_code(features)

    time_solve_begin = timeit.default_timer()
    stdout, stderr, exit_code, _ = run_picat(source_code, time_limit=time_limit)
    time_solve_end = timeit.default_timer()

    parsed_values = _parse_picat_output(stdout) if exit_code == 0 else {}

    if exit_code == 0 and parsed_values:
        status_code = 0
    elif exit_code == 0 and not parsed_values:
        status_code = 0
    else:
        status_code = 1

    if log:
        print(f"\n--- Picat Output ---")
        if stdout.strip():
            print(stdout.strip())
        if stderr.strip() and exit_code != 0:
            print(f"Error: {stderr.strip()}")
        print(f"--------------------\n")

    generated_solution = [
        [status_code, parsed_values],
        [time_solve_begin, time_solve_end]
    ]

    return generated_solution
