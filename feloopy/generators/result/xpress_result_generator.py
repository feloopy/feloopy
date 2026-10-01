# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import xpress as xpress_interface
from ...helpers.formatter import *
from ...helpers.reporter import left_align


def _resolve_constraint(model_object, input2):
    if isinstance(input2, str):
        try:
            for c in model_object.getConstraint():
                if c.name == input2:
                    return c
        except Exception:
            pass
        return None
    return input2


def _normalize_name(name):
    """Strip brackets, parentheses and spaces so index formats can be compared.

    Queries use Python tuple formatting ("x[(0, 0)]") while the variable
    generator names variables as "x(0, 0)" / "x0" — normalize both sides
    before matching.
    """
    return (name.replace('[', '').replace(']', '')
                .replace('(', '').replace(')', '').replace(' ', ''))


def _resolve_variable(model_object, input2):
    if isinstance(input2, str):
        try:
            for v in model_object.getVariable():
                if v.name == input2:
                    return v
            target = _normalize_name(input2)
            if target:
                for v in model_object.getVariable():
                    if _normalize_name(v.name) == target:
                        return v
        except Exception:
            pass
        return None
    return input2


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            try:
                if isinstance(input2, str):
                    v = _resolve_variable(model_object, input2)
                    if v is None:
                        return None
                    val = getattr(v, 'sol', None)
                    if val is not None:
                        return float(val)
                    return model_object.getSolution(v)
                val = getattr(input2, 'sol', None)
                if val is not None:
                    return float(val)
                return model_object.getSolution(input2)
            except Exception:
                return None

        case 'status':

            try:
                solstatus = model_object.attributes.solstatus
                return str(solstatus).split('.')[-1].lower()
            except Exception:
                return 'unknown'

        case 'objective':

            return model_object.attributes.objval

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            try:
                return model_object.attributes.bestbound
            except Exception:
                return None

        case 'dual':

            try:
                all_duals = model_object.getDuals()
                c = _resolve_constraint(model_object, input2)
                if c is not None:
                    idx = model_object.getIndex(c)
                    return all_duals[idx]
            except Exception:
                return None

        case 'slack':

            try:
                all_slacks = model_object.getSlacks()
                c = _resolve_constraint(model_object, input2)
                if c is not None:
                    idx = model_object.getIndex(c)
                    return all_slacks[idx]
            except Exception:
                return None

        case 'rc':

            try:
                all_rc = model_object.getRedCosts()
                v = _resolve_variable(model_object, input2)
                if v is not None:
                    idx = model_object.getIndex(v)
                    return all_rc[idx]
            except Exception:
                return None

        case 'iis':

            try:
                model_object.iis()
                output = ''
                iis_rows = model_object.getIISRow()
                for idx in iis_rows:
                    c = model_object.getConstraint(idx)
                    output += left_align("con: %s" % c.name, rt=True)
                    output += "\n"
                iis_cols = model_object.getIISCol()
                for idx in iis_cols:
                    v = model_object.getVariable(idx)
                    output += left_align("var: %s" % v.name, rt=True)
                    output += "\n"
                return output
            except Exception:
                return ''

        case 'mipgap':

            try:
                return model_object.attributes.mipgap
            except Exception:
                return None

        case 'solcount':

            try:
                return model_object.attributes.sols
            except Exception:
                return 0

        case 'objn':

            try:
                return model_object.attributes.objval
            except Exception:
                return None

        case 'sensitivity_obj':

            try:
                v = _resolve_variable(model_object, input2)
                if v is not None:
                    idx = model_object.getIndex(v)
                    return None, None
            except Exception:
                return None, None

        case 'sensitivity_bound':

            try:
                v = _resolve_variable(model_object, input2)
                if v is not None:
                    idx = model_object.getIndex(v)
                    return None, None, None, None
            except Exception:
                return None, None, None, None

        case 'basis':

            try:
                v = _resolve_variable(model_object, input2)
                if v is not None:
                    return None
            except Exception:
                return None

        case 'constrbasis':

            try:
                c = _resolve_constraint(model_object, input2)
                if c is not None:
                    return None
            except Exception:
                return None

        case 'pool':

            try:
                pool_soln = input2 if input2 is not None else 0
                return model_object.getSolution(pool_soln)
            except Exception:
                return None

        case 'pool_obj':

            try:
                pool_soln = input2 if input2 is not None else 0
                return model_object.getObjVal(pool_soln)
            except Exception:
                return None

        case 'feasibility':

            try:
                return {
                    'max_vio': None,
                    'bound_vio': None,
                    'constr_vio': None,
                    'int_vio': None,
                    'compl_vio': None,
                }
            except Exception:
                return None
