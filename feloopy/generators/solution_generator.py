# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np


def generate_solution(features):

    if features.get('auto_linearize'):
        from ..classes.linearization import unwrap_proxies
        features['objectives'] = unwrap_proxies(features['objectives'])
        features['constraints'] = unwrap_proxies(features['constraints'])

    _original_constraints = features.get('constraints')
    _original_labels = features.get('constraint_labels')
    features['constraints'], features['constraint_labels'] = _sanitize_constraints(features)
    try:
        return _dispatch_solution(features)
    finally:
        features['constraints'] = _original_constraints
        features['constraint_labels'] = _original_labels


def _sanitize_constraints(features):
    """Return solver-safe ``(constraints, labels)`` lists.
    """
    cons = features.get('constraints')
    if not cons:
        return cons, features.get('constraint_labels')
    labs = features.get('constraint_labels')
    kept_constraints = []
    kept_labels = [] if labs is not None else None
    for i, constraint in enumerate(cons):
        if constraint is None or isinstance(constraint, (bool, np.bool_)):
            if constraint is not None and not bool(constraint):
                from ..helpers.error import ConstantConstraintError
                where = ""
                if labs is not None and i < len(labs) and labs[i] is not None:
                    where = " (%s)" % labs[i]
                raise ConstantConstraintError(
                    "A constraint%s contains no variables and evaluates to "
                    "False, so the model can never be satisfied. Check the "
                    "data feeding it." % where)
            continue  # vacuous True / placeholder row adds no model row
        if isinstance(constraint, (int, float, np.integer, np.floating)):
            continue  # bare number: not a constraint object
        kept_constraints.append(constraint)
        if kept_labels is not None and i < len(labs):
            kept_labels.append(labs[i])
    return kept_constraints, kept_labels


def _dispatch_solution(features):

    match features['interface_name']:

        case 'pulp':

            from .solution import pulp_solution_generator
            ModelSolution = pulp_solution_generator.generate_solution(features)

        case 'casadi':

            from .solution import casadi_solution_generator
            ModelSolution = casadi_solution_generator.generate_solution(features)

        case 'pyomo':

            from .solution import pyomo_solution_generator
            ModelSolution = pyomo_solution_generator.generate_solution(
                features)

        case 'insideopt':

            from .solution import seeker_solution_generator
            ModelSolution = seeker_solution_generator.generate_solution(features)

        case 'insideopt-demo':

            from .solution import seeker_solution_generator
            ModelSolution = seeker_solution_generator.generate_solution(features)

        case 'gams':

            from .solution import gamspy_solution_generator
            ModelSolution = gamspy_solution_generator.generate_solution(features)

        case 'highs':

            from .solution import highs_solution_generator
            ModelSolution = highs_solution_generator.generate_solution(features)

        case 'jump':

            from .solution import jump_solution_generator
            ModelSolution = jump_solution_generator.generate_solution(features)
                                           
        case 'ortools':

            from .solution import ortools_solution_generator
            ModelSolution = ortools_solution_generator.generate_solution(
                features)

        case 'ortools_cp':

            from .solution import ortools_cp_solution_generator
            ModelSolution = ortools_cp_solution_generator.generate_solution(
                features)

        case 'gekko':

            from .solution import gekko_solution_generator
            ModelSolution = gekko_solution_generator.generate_solution(
                features)

        case 'picos':

            from .solution import picos_solution_generator
            ModelSolution = picos_solution_generator.generate_solution(
                features)

        case 'mathopt':

            from .solution import mathopt_solution_generator
            ModelSolution = mathopt_solution_generator.generate_solution(
                features)
            
        case name if 'pyoptinterface' in name:


            from .solution import pyoptinterface_solution_generator
            ModelSolution = pyoptinterface_solution_generator.generate_solution(
                features)

        case 'cvxpy':

            from .solution import cvxpy_solution_generator
            ModelSolution = cvxpy_solution_generator.generate_solution(
                features)

        case 'cylp':

            from .solution import cylp_solution_generator
            ModelSolution = cylp_solution_generator.generate_solution(features)

        case 'pymprog':

            from .solution import pymprog_solution_generator
            ModelSolution = pymprog_solution_generator.generate_solution(
                features)

        case 'cplex':

            from .solution import cplex_solution_generator
            ModelSolution = cplex_solution_generator.generate_solution(features)
            
        case 'cplex_cp':

            from .solution import cplex_cp_solution_generator
            ModelSolution = cplex_cp_solution_generator.generate_solution(
                features)

        case 'gurobi':

            from .solution import gurobi_solution_generator
            ModelSolution = gurobi_solution_generator.generate_solution(
                features)

        case 'copt':

            from .solution import copt_solution_generator
            ModelSolution = copt_solution_generator.generate_solution(
                features)

        case 'xpress':

            from .solution import xpress_solution_generator
            ModelSolution = xpress_solution_generator.generate_solution(
                features)

        case 'mip':

            from .solution import mip_solution_generator
            ModelSolution = mip_solution_generator.generate_solution(features)

        case 'linopy':

            from .solution import linopy_solution_generator
            ModelSolution = linopy_solution_generator.generate_solution(
                features)

        case 'rsome_ro':

            from .solution import rsome_ro_solution_generator
            ModelSolution = rsome_ro_solution_generator.generate_solution(
                features)

        case 'rsome_dro':

            from .solution import rsome_dro_solution_generator
            ModelSolution = rsome_dro_solution_generator.generate_solution(features)

        case 'uno':

            from .solution import uno_solution_generator
            ModelSolution = uno_solution_generator.generate_solution(features)

        case 'bonmin' | 'couenne':

            from .solution import coin_solution_generator
            ModelSolution = coin_solution_generator.generate_solution(features)

        case 'scip':

            from .solution import scip_solution_generator
            ModelSolution = scip_solution_generator.generate_solution(features)

        case 'hexaly':

            from .solution import hexaly_solution_generator
            ModelSolution = hexaly_solution_generator.generate_solution(features)

        case 'mosek':

            from .solution import mosek_solution_generator
            ModelSolution = mosek_solution_generator.generate_solution(features)

        case 'picat':

            from .solution import picat_solution_generator
            ModelSolution = picat_solution_generator.generate_solution(features)

    return ModelSolution
