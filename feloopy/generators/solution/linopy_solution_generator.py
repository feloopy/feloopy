# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from linopy import Model as LINOPYMODEL
from linopy import LinearExpression
import timeit

linopy_solver_selector = {'cbc': 'cbc',
                          'glpk': 'glpk',
                          'highs': 'highs',
                          'gurobi': 'gurobi',
                          'xpress': 'xpress',
                          'cplex': 'cplex',
                          'copt', 'copt'}

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
    
    if solver_name not in linopy_solver_selector.keys():
        raise RuntimeError("Using solver '%s' is not supported by 'linopy'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))
    match debug:
        case False:
            match directions[objective_id]:
                case "min":
                    model_object.add_objective(model_objectives[objective_id])
                case "max":
                    model_object.add_objective(-1*model_objectives[objective_id])
            for constraint in model_constraints: model_object.add_constraint(constraint)
            time_solve_begin = timeit.default_timer()
            result = model_object.solve(solver_name=solver_name)
            time_solve_end = timeit.default_timer()
            generated_solution = [result, [time_solve_begin, time_solve_end]]
    return generated_solution
