# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import picos as picos_interface
import timeit

from ...helpers.solver_executables import missing_solver_message

picos_solver_selector = {'cplex': 'cplex',
                         'cvxopt': 'cvxopt',
                         'ecos': 'ecos',
                         'glpk': 'glpk',
                         'gurobi': 'gurobi',
                         'mosek': 'mosek',
                         'mskfsn': 'mskfsn',
                         'osqp': 'osqp',
                         'scip': 'scip',
                         'smcp': 'smcp'}


_picos_setup_name = {'scip': 'pyscipopt',
                     'glpk': 'cvxopt',
                     'mskfsn': 'mosek'}


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

    constraint_dict = dict()
    if len(model_constraints)!=0:
        if any(constraint_labels)!=None:
            for i in range(len(model_constraints)):
                constraint_dict[constraint_labels[i]] = model_constraints[i]

    if solver_name not in picos_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'picos'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))


    try:
        available = {str(s) for s in picos_interface.AvailableSolvers()}
    except Exception:
        available = None  # older/newer picos API: fall back to its own error
    if available is not None and solver_name not in available:
        raise RuntimeError(
            missing_solver_message(_picos_setup_name.get(solver_name, solver_name)))

    match debug:

        case False | True:

            match directions[objective_id]:
                case "min":
                    model_object.set_objective(
                        'min', model_objectives[objective_id])
                case "max":
                    model_object.set_objective(
                        'max', model_objectives[objective_id])

            counter=0
            for constraint in model_constraints:
                model_object.add_constraint(constraint)
                counter+=1

            time_solve_begin = timeit.default_timer()
            result = model_object.solve(solver=solver_name)
            time_solve_end = timeit.default_timer()
            generated_solution = [[result,constraint_dict], [time_solve_begin, time_solve_end]]

    return generated_solution
