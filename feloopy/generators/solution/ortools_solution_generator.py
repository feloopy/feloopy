# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ortools.linear_solver import pywraplp as ortools_interface
from ..model.ortools_proxy import OrtoolsModelProxy, OrtoolsExprProxy, OrtoolsComparisonProxy, OrtoolsVarProxy, INFINITY
import timeit

ortools_solver_selector = {
    'clp': 'CLP_LINEAR_PROGRAMMING',
    'cbc': 'CBC_MIXED_INTEGER_PROGRAMMING',
    'scip': 'SCIP_MIXED_INTEGER_PROGRAMMING',
    'glop': 'GLOP_LINEAR_PROGRAMMING',
    'bop': 'BOP_INTEGER_PROGRAMMING',
    'sat': 'SAT_INTEGER_PROGRAMMING',
    'gurobi_': 'GUROBI_LINEAR_PROGRAMMING',
    'gurobi': 'GUROBI_MIXED_INTEGER_PROGRAMMING',
    'cplex_': 'CPLEX_LINEAR_PROGRAMMING',
    'cplex': 'CPLEX_MIXED_INTEGER_PROGRAMMING',
    'xpress_': 'XPRESS_LINEAR_PROGRAMMING',
    'xpress': 'XPRESS_MIXED_INTEGER_PROGRAMMING',
    'glpk_': 'GLPK_LINEAR_PROGRAMMING',
    'glpk': 'GLPK_MIXED_INTEGER_PROGRAMMING',
    'highs_': 'HIGHS_LINEAR_PROGRAMMING',
    'highs': 'HIGHS_MIXED_INTEGER_PROGRAMMING',
    'pdlp': 'PDLP_LINEAR_PROGRAMMING',
}

lp_fallback = {
    'scip': 'GLOP_LINEAR_PROGRAMMING',
    'highs': 'GLOP_LINEAR_PROGRAMMING',
    'cbc': 'GLOP_LINEAR_PROGRAMMING',
}


def _materialize_proxy(proxy, features):
    solver_name = features['solver_name']
    has_int = features['integer_variable_counter'][0] > 0
    has_bin = features['binary_variable_counter'][0] > 0
    has_mip = has_int or has_bin

    if has_mip:
        solver_type = ortools_solver_selector.get(solver_name, 'SCIP_MIXED_INTEGER_PROGRAMMING')
    else:
        solver_type = lp_fallback.get(solver_name, ortools_solver_selector.get(solver_name, 'GLOP_LINEAR_PROGRAMMING'))

    real_solver = ortools_interface.Solver.CreateSolver(solver_type)
    if real_solver is None:
        from ...helpers.solver_guides import resolve_commercial
        commercial = resolve_commercial(solver_name)
        hint = (" Install the commercial binding with: flp setup {} "
                "(prints the vendor's official guide).".format(commercial)
                if commercial else "")
        raise RuntimeError(
            "Solver backend '%s' is not available in this ortools build "
            "(not linked in or license not found). "
            "Use another interface or a solver linked into ortools "
            "(e.g. CLP, GLOP, CBC, SCIP).%s" % (solver_type, hint))

    real_vars = {}
    for spec in proxy.variable_specs:
        name = spec['name']
        lb = spec['lb'] if abs(spec['lb']) < INFINITY else -real_solver.infinity()
        ub = spec['ub'] if abs(spec['ub']) < INFINITY else real_solver.infinity()
        if spec['is_integer']:
            int_lb = int(lb) if abs(lb) < 1e15 else -2147483647
            int_ub = int(ub) if abs(ub) < 1e15 else 2147483647
            real_vars[name] = real_solver.IntVar(int_lb, int_ub, name)
        else:
            real_vars[name] = real_solver.NumVar(lb, ub, name)

    features['model_object_before_solve'] = real_solver
    features['_ortools_var_map'] = real_vars

    return real_solver, real_vars


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    save = features['save_solver_log']
    save_model = features['write_model_file']
    email = features['email_address']
    max_iterations = features['max_iterations']
    solver_options = features['solver_options']

    if solver_name not in ortools_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'ortools'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    if isinstance(model_object, OrtoolsModelProxy):
        real_solver, real_vars = _materialize_proxy(model_object, features)
    else:
        real_solver = model_object
        real_vars = features.get('_ortools_var_map', {})

    match debug:

        case False | True:

            obj_expr = model_objectives[objective_id]
            if isinstance(obj_expr, OrtoolsExprProxy):
                obj_expr = obj_expr.evaluate(real_vars)
            elif isinstance(obj_expr, OrtoolsVarProxy):
                obj_expr = real_vars[obj_expr._name]

            match directions[objective_id]:

                case "min":
                    real_solver.Minimize(obj_expr)

                case "max":
                    real_solver.Maximize(obj_expr)

            if len(model_constraints) != 0:
                counter = 0
                for constraint in model_constraints:
                    if constraint_labels[counter] is None:
                        constraint_labels[counter] = f"con[{counter}]"
                    if isinstance(constraint, OrtoolsComparisonProxy):
                        real_solver.Add(constraint.evaluate(real_vars), name=constraint_labels[counter])
                    else:
                        real_solver.Add(constraint, name=constraint_labels[counter])
                    counter += 1

            solverParams = ortools_interface.MPSolverParameters()

            if time_limit is not None:
                real_solver.set_time_limit(time_limit)

            if thread_count is not None:
                real_solver.SetNumThreads(thread_count)

            if relative_gap is not None:
                solverParams.SetDoubleParam(
                    solverParams.RELATIVE_MIP_GAP, relative_gap)

            if absolute_gap is not None:
                pass

            if log:
                pass

            from ..init_generator import flush_init
            flush_init(features, force=True)

            time_solve_begin = timeit.default_timer()
            result = real_solver.Solve()
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end]]

    return generated_solution
