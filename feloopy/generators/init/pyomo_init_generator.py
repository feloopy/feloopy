# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def set_init_value(features, variable, value, fix):
    if fix:
        variable.fix(value)
    else:
        variable.set_value(value)
        features['_pyomo_warmstart'] = True


def set_init_values(features, assignments, fix=False):
    for variable, value in assignments:
        set_init_value(features, variable, value, fix)
