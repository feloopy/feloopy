# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""OR-Tools (pywraplp) warm start.
"""


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _name_of(variable):
    name = getattr(variable, '_name', None)
    if name is None:
        getter = getattr(variable, 'name', None)
        name = getter() if callable(getter) else getter
    return name


def _stage(features, variable, value):
    features.setdefault('_ortools_mip_start', {})[id(variable)] = (
        variable, float(value))


def _fix_spec(features, variable, value):
    """Update the proxy variable spec so materialization picks the bound up."""
    proxy = _get_model(features)
    specs = getattr(proxy, 'variable_specs', None)
    name = _name_of(variable)
    if specs is None or name is None:
        return False
    for spec in specs:
        if spec.get('name') == name:
            spec['lb'] = float(value)
            spec['ub'] = float(value)
            return True
    return False


def set_init_value(features, variable, value, fix):
    if fix:
        _fix_spec(features, variable, value)
        for attr in ('_lb', '_ub'):
            try:
                setattr(variable, attr, float(value))
            except Exception:
                pass
    else:
        _stage(features, variable, value)


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)


def finalize(features):
    staged = features.get('_ortools_mip_start')
    if not staged:
        return

    solver = features.get('model_object_before_solve')
    real_vars = features.get('_ortools_var_map') or {}
    if solver is None or not real_vars:
        return

    variables = []
    values = []
    for variable, value in staged.values():
        real = real_vars.get(_name_of(variable))
        if real is None:
            continue
        variables.append(real)
        values.append(value)

    if variables:
        solver.SetHint(variables, values)
