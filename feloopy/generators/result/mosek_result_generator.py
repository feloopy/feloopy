# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mosek

status_mosek = {
    mosek.solsta.optimal: 1,
    mosek.solsta.prim_feas: 1,
    mosek.solsta.dual_feas: 0,
    mosek.solsta.prim_and_dual_feas: 1,
    mosek.solsta.prim_infeas_cer: 0,
    mosek.solsta.dual_infeas_cer: 0,
    mosek.solsta.prim_illposed_cer: 0,
    mosek.solsta.dual_illposed_cer: 0,
    mosek.solsta.integer_optimal: 1,
}

INF = 1e30


def Get(model_object, model_solution, indicator, variable_name_with_index, input=None):
    Thing = indicator[0]

    match Thing:

        case 'variable':
            return get_variable(model_object, variable_name_with_index, model_solution, input=input)

        case 'dual':
            return get_dual(model_object, indicator, model_solution, input=input)

        case 'slack':
            return get_slack(model_object, indicator, model_solution, input=input)

        case 'status':
            return get_status(model_object, model_solution)

        case 'objective':
            return get_objective(model_object, model_solution)


def get_variable(model_object, variable_name_with_index, generated_solution, input=None):
    sol = mosek.soltype.bas
    xx = [0.0] * model_object.getnumvar()
    model_object.getxx(sol, xx)

    variable_names = variable_name_with_index

    if input is not None:
        if isinstance(input, str):
            idx = variable_names.get(input)
            return float(xx[idx]) if idx is not None else None
        elif isinstance(input, list):
            return {name: float(xx[variable_names[name]]) for name in input if name in variable_names}
    else:
        return {name: float(xx[idx]) for name, idx in variable_names.items()}


def get_dual(model_object, indicator, generated_solution, input=None):
    sol = mosek.soltype.bas
    y = [0.0] * model_object.getnumcon()
    model_object.gety(sol, y)

    constraint_labels = indicator[3] if len(indicator) > 3 else []

    if input is not None:
        if isinstance(input, str):
            idx = None
            for i, label in enumerate(constraint_labels):
                if label == input:
                    idx = i
                    break
            return float(y[idx]) if idx is not None else None
        elif isinstance(input, list):
            result = {}
            for name in input:
                for i, label in enumerate(constraint_labels):
                    if label == name:
                        result[name] = float(y[i])
                        break
            return result
    else:
        return {label: float(y[i]) for i, label in enumerate(constraint_labels)}


def get_slack(model_object, indicator, generated_solution, input=None):
    sol = mosek.soltype.bas
    slc = [0.0] * model_object.getnumcon()
    model_object.getslc(sol, slc)

    constraint_labels = indicator[3] if len(indicator) > 3 else []

    if input is not None:
        if isinstance(input, str):
            idx = None
            for i, label in enumerate(constraint_labels):
                if label == input:
                    idx = i
                    break
            return float(slc[idx]) if idx is not None else None
        elif isinstance(input, list):
            result = {}
            for name in input:
                for i, label in enumerate(constraint_labels):
                    if label == name:
                        result[name] = float(slc[i])
                        break
            return result
    else:
        return {label: float(slc[i]) for i, label in enumerate(constraint_labels)}


def get_status(model_object, generated_solution):
    sol = mosek.soltype.bas
    try:
        prosta = model_object.getprosta(sol)
        solsta = model_object.getsolsta(sol)
    except mosek.Error:
        return 0

    if prosta == mosek.prosta.prim_infeas:
        return 0
    elif prosta == mosek.prosta.dual_infeas:
        return 0
    elif solsta in (mosek.solsta.optimal, mosek.solsta.integer_optimal, mosek.solsta.prim_feas):
        return 1
    else:
        return 0


def get_objective(model_object, generated_solution):
    sol = mosek.soltype.bas
    return model_object.getprimalobj(sol)


def get_bound(model_object):
    return None
