# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


from gamspy import Equation, Model, Options

from ..variable.gamspy_variable_generator import gams_safe_name

import timeit

solvers = [
    'alphaecp', 'antigone', 'baron', 'cbc', 'conopt', 'convert',
    'copt', 'cplex', 'de', 'decis', 'dicopt', 'emp', 'empsp',
    'examiner', 'gamschk', 'gurobi', 'guss', 'highs', 'ipopt',
    'jams', 'kestrel', 'knitro', 'lindo', 'lindoglobal', 'miles',
    'minos', 'mosek', 'mps2gms', 'mpsge', 'msnlp', 'nlpec',
    'octeract', 'odh', 'path', 'pathnlp', 'quad', 'sbb', 'scip',
    'shot', 'snopt', 'soplex', 'xpress',
]

gams_solver_selector = {key.lower(): key.upper() for key in solvers}


def _detect_problem_type(features):
    import re

    has_integer = features['integer_variable_counter'][0] > 0
    has_binary = features['binary_variable_counter'][0] > 0
    has_mixed_integer = has_integer or has_binary

    nonlinear_ops = ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs',
                     'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'log10']
    has_nonlinear = False
    is_quadratic = False

    def _check(expr_str):
        nonlocal has_nonlinear, is_quadratic
        for op in nonlinear_ops:
            if op + '(' in expr_str or op + ' ' in expr_str:
                has_nonlinear = True
                return
        if re.search(r'\*\*\s*\d+', expr_str):
            degree = int(re.search(r'\*\*\s*(\d+)', expr_str).group(1))
            if degree > 2:
                has_nonlinear = True
            elif degree == 2:
                is_quadratic = True

    for obj in features['objectives']:
        _check(str(obj))
    for con in features['constraints']:
        _check(str(con))

    if has_mixed_integer:
        if has_nonlinear or is_quadratic:
            return 'MINLP'
        return 'MIP'

    if has_nonlinear:
        return 'NLP'
    if is_quadratic:
        return 'QCP'
    return 'LP'


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    solver_options = features['solver_options']

    if solver_name not in gams_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'gams'! \n"
            "Possible fixes: \n1) Check the solver name. \n"
            "2) Use another interface. \n" % (solver_name))

    gams_options = {}

    if time_limit is not None:
        gams_options['time_limit'] = time_limit
    if thread_count is not None:
        gams_options['threads'] = thread_count
    if relative_gap is not None:
        gams_options['relative_optimality_gap'] = relative_gap
    if absolute_gap is not None:
        gams_options['absolute_optimality_gap'] = absolute_gap

    if log:
        import sys
        output = sys.stdout
    else:
        output = None

    obj_name = f"obj_{objective_id}"
    obj_equation = Equation(model_object, name=obj_name, type="regular")
    obj_equation[...] = model_objectives[objective_id]

    equation_dict = {}
    for counter, constraint in enumerate(model_constraints):
        raw_label = constraint_labels[counter] or f"con_{counter}"
        safe_label = gams_safe_name(raw_label)
        equation_dict[safe_label] = Equation(model_object, name=safe_label, type="regular")
        equation_dict[safe_label][...]= constraint

    problem_type = _detect_problem_type(features)

    gamspy_model = Model(
        model_object,
        name='feloopy_model',
        equations=model_object.getEquations(),
        problem=problem_type,
        sense=directions[objective_id],
        objective=features['objective_variable'],
    )

    time_solve_begin = timeit.default_timer()
    gamspy_model.solve(
        solver=gams_solver_selector.get(solver_name),
        options=Options(**gams_options),
        output=output,
    )
    time_solve_end = timeit.default_timer()

    generated_solution = [
        [equation_dict, features['objective_variable'], gamspy_model],
        [time_solve_begin, time_solve_end],
    ]

    return generated_solution
