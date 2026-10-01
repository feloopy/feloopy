# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import sys

import gurobipy as gurobi_interface
from ...helpers.formatter import *
from ...helpers.reporter import left_align

gurobi_status_dict = {
    gurobi_interface.GRB.LOADED: 'loaded',
    gurobi_interface.GRB.OPTIMAL: 'optimal',
    gurobi_interface.GRB.INFEASIBLE: 'infeasible',
    gurobi_interface.GRB.INF_OR_UNBD: 'infeasible or unbounded',
    gurobi_interface.GRB.UNBOUNDED: 'unbounded',
    gurobi_interface.GRB.CUTOFF: 'cutoff',
    gurobi_interface.GRB.ITERATION_LIMIT: 'iteration limit',
    gurobi_interface.GRB.NODE_LIMIT: 'node limit',
    gurobi_interface.GRB.TIME_LIMIT: 'time limit',
    gurobi_interface.GRB.SOLUTION_LIMIT: 'solution limit',
    gurobi_interface.GRB.INTERRUPTED: 'interrupted',
    gurobi_interface.GRB.NUMERIC: 'numerical',
    gurobi_interface.GRB.SUBOPTIMAL: 'suboptimal',
    gurobi_interface.GRB.INPROGRESS: 'inprogress'
}

grb = gurobi_interface.GRB


def _normalize_name(name):
    return (name.replace('[', '').replace(']', '')
                .replace('(', '').replace(')', '').replace(' ', ''))


def _resolve_variable(model_object, input2):
    if isinstance(input2, str):
        try:
            v = model_object.getVarByName(input2)
            if v is not None:
                return v
        except Exception:
            pass
        target = _normalize_name(input2)
        for v in model_object.getVars():
            if v.varName == input2:
                return v
            if _normalize_name(v.varName) == target:
                return v
        return None
    try:
        _ = input2.X
        return input2
    except Exception:
        return input2


def _resolve_constraint(model_object, input2):
    if isinstance(input2, str):
        try:
            c = model_object.getConstrByName(input2)
            if c is not None:
                return c
        except Exception:
            pass
        try:
            c = model_object.getQConstrByName(input2)
            if c is not None:
                return c
        except Exception:
            pass
        try:
            c = model_object.getGenConstrByName(input2)
            if c is not None:
                return c
        except Exception:
            pass
        return None
    return input2


def Get(model_object, result, input1, input2=None):
    input1 = input1[0]

    match input1:
        case 'variable':
            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.X
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'status':
            return gurobi_status_dict.get(model_object.status, 'unknown')

        case 'objective':
            try:
                return model_object.ObjVal
            except (gurobi_interface.GurobiError, AttributeError):
                try:
                    return model_object.ObjNVal
                except (gurobi_interface.GurobiError, AttributeError):
                    return 0.0

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            try:
                return model_object.ObjBound
            except (gurobi_interface.GurobiError, AttributeError):
                return None

        case 'dual':
            constr = _resolve_constraint(model_object, input2)
            if constr is not None:
                try:
                    return constr.Pi
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'slack':
            constr = _resolve_constraint(model_object, input2)
            if constr is not None:
                try:
                    return constr.Slack
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'rc':
            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.RC
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'iis':
            model_object.computeIIS()

            output = ''

            constrs = model_object.getConstrs()
            vars = model_object.getVars()

            for i, c in enumerate(constrs):
                if c.IISConstr:
                    output += left_align(f"con: {c.constrName}", rt=True)
                    if i != len(constrs) - 1 or i == 0:
                        output += "\n"

            for i, v in enumerate(vars):
                if v.IISLB > 0 or v.IISUB > 0:
                    output += left_align(f"var: {v.varName}", rt=True)
                    if i != len(vars) - 1 or i == 0:
                        output += "\n"

            return output

        case 'mipgap':
            try:
                return model_object.MIPGap
            except (gurobi_interface.GurobiError, AttributeError):
                return None

        case 'solcount':
            try:
                return model_object.SolCount
            except (gurobi_interface.GurobiError, AttributeError):
                return 0

        case 'objn':
            n = input2 if input2 is not None else 0
            try:
                return model_object.ObjNVal
            except (gurobi_interface.GurobiError, AttributeError):
                return None

        case 'sensitivity_obj':
            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.SAObjLow, var.SAObjUp
                except (gurobi_interface.GurobiError, AttributeError):
                    return None, None
            return None, None

        case 'sensitivity_bound':
            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.SALBLow, var.SALBUp, var.SAUBLow, var.SAUBUp
                except (gurobi_interface.GurobiError, AttributeError):
                    return None, None, None, None
            return None, None, None, None

        case 'basis':
            var = _resolve_variable(model_object, input2)
            if var is not None:
                try:
                    return var.VBasis
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'constrbasis':
            constr = _resolve_constraint(model_object, input2)
            if constr is not None:
                try:
                    return constr.CBasis
                except (gurobi_interface.GurobiError, AttributeError):
                    return None
            return None

        case 'pool':
            n = input2 if input2 is not None else 0
            try:
                model_object.setParam('SolutionNumber', n)
                return [v.Xn for v in model_object.getVars()]
            except (gurobi_interface.GurobiError, AttributeError):
                return None

        case 'pool_obj':
            n = input2 if input2 is not None else 0
            try:
                model_object.setParam('SolutionNumber', n)
                return model_object.PoolObjVal
            except (gurobi_interface.GurobiError, AttributeError):
                return None

        case 'feasibility':
            try:
                return {
                    'max_vio': model_object.MaxVio,
                    'bound_vio': model_object.BoundVio,
                    'constr_vio': model_object.ConstrVio,
                    'int_vio': model_object.IntVio,
                    'compl_vio': model_object.ComplVio,
                }
            except (gurobi_interface.GurobiError, AttributeError):
                return None
