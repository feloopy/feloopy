# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ortools.linear_solver import pywraplp as ortools_interface

ortools_status_dict = {0: "optimal", 1: "feasible", 2: "infeasible",
                       3: "unbounded", 4: "abnormal", 5: "model_invalid", 6: "not_solved"}


def _normalize_name(name):
    return (name.replace('[', '').replace(']', '')
                .replace('(', '').replace(')', '').replace(' ', ''))


def _resolve_variable(model_object, input2):
    if hasattr(input2, '_name'):
        input2 = input2._name
    if hasattr(input2, 'solution_value'):
        if input2.solution_value() is not None:
            return input2
    if not isinstance(input2, str):
        return input2
    for v in model_object.variables():
        if v.name() == input2:
            return v
    target = _normalize_name(input2)
    if target:
        for v in model_object.variables():
            if _normalize_name(v.name()) == target:
                return v
    return input2


def _resolve_constraint(model_object, name):
    if isinstance(name, str):
        return model_object.LookupConstraint(name)
    return name


def Get(model_object, result, input1, input2=None):

    directions = +1 if input1[1][input1[2]] == 'min' else -1
    input1 = input1[0]

    match input1:

        case 'variable':

            var = _resolve_variable(model_object, input2)
            if hasattr(var, 'solution_value'):
                return var.solution_value()
            return None

        case 'status':

            return ortools_status_dict.get(result[0], "Not Optimal")

        case 'objective':

            return model_object.Objective().Value()

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            return None

        case 'ogr':

            return None

        case 'dual':

            constraint = _resolve_constraint(model_object, input2)
            if constraint is not None:
                return constraint.dual_value()
            return None

        case 'slack':

            constraint = _resolve_constraint(model_object, input2)
            if constraint is not None:
                activities = model_object.ComputeConstraintActivities()
                idx = constraint.index()
                activity = activities[idx]
                lb = constraint.lb()
                ub = constraint.ub()
                if lb > -model_object.infinity() and ub < model_object.infinity():
                    return min(ub - activity, activity - lb)
                elif ub < model_object.infinity():
                    return ub - activity
                elif lb > -model_object.infinity():
                    return activity - lb
                return 0.0
            return None

        case 'rc':

            var = _resolve_variable(model_object, input2)
            if hasattr(var, 'reduced_cost'):
                return var.reduced_cost()
            return None
