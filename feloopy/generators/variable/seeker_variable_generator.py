# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
sets = it.product


def _make_continuous(model_object, lb, ub):
    return model_object.continuous(low=lb, high=ub)


def _make_categorical(model_object, lb, ub):
    return model_object.categorical(low=lb, high=ub)


def _make_vars(model_object, variable_type, lb, ub, variable_dim):
    if variable_type in ('pvar', 'fvar', 'rvar', 'dvar', 'ftvar', 'ptvar'):
        factory = lambda: _make_continuous(model_object, lb, ub)
    elif variable_type in ('bvar', 'btvar'):
        factory = lambda: _make_categorical(model_object, 0, 1)
    elif variable_type in ('ivar', 'itvar'):
        factory = lambda: _make_categorical(model_object, lb, ub)
    else:
        factory = lambda: _make_continuous(model_object, lb, ub)

    if variable_dim == 0:
        return factory()

    if isinstance(variable_dim, set):
        return {key: factory() for key in variable_dim}

    if len(variable_dim) == 1:
        return {key: factory() for key in variable_dim[0]}

    return {key: factory() for key in sets(*variable_dim)}


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):
    lb, ub = variable_bound[0], variable_bound[1]
    return _make_vars(model_object, variable_type, lb, ub, variable_dim)
