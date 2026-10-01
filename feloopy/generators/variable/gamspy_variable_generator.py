# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from gamspy import Variable

import itertools as it
sets = it.product


def gams_safe_name(name):
    if not name:
        return name
    s = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in str(name))
    if not (s[0].isalpha() or s[0] == '_'):
        s = '_' + s
    return s


def _fmt(name, key):
    if isinstance(key, tuple):
        return f"{name}_{'_'.join(str(k) for k in key)}"
    return f"{name}{key}"


def _make_var(container, name, vtype, domain=None, lb=None, ub=None):
    if domain is not None:
        v = Variable(container, name=name, type=vtype, domain=domain)
    else:
        v = Variable(container, name=name, type=vtype)
    if lb is not None:
        v.lo = lb
    if ub is not None:
        v.up = ub
    return v


def _parse_bound(variable_bound, variable_type):
    if variable_bound is None or variable_bound == 0:
        return None, None
    if not isinstance(variable_bound, (list, tuple)):
        return None, None
    if len(variable_bound) == 2:
        lb, ub = variable_bound
        if variable_type == 'pvar':
            lb = max(0, lb) if lb is not None else 0
        return lb, ub
    return None, None


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    vtype_map = {
        'pvar': 'positive',
        'bvar': 'binary',
        'ivar': 'integer',
        'fvar': 'free',
    }
    vtype = vtype_map.get(variable_type, 'free')

    lb, ub = _parse_bound(variable_bound, variable_type)

    if variable_dim == 0:
        return _make_var(model_object, gams_safe_name(variable_name), vtype, lb=lb, ub=ub)

    domain = variable_dim[0] if len(variable_dim) == 1 else sets(*variable_dim)
    safe_name = gams_safe_name(variable_name)
    return {
        key: _make_var(model_object, _fmt(safe_name, key), vtype, lb=lb, ub=ub)
        for key in domain
    }
