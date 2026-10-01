# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def set_init_value(features, variable, value, fix):
    if fix:
        variable.lb = value
        variable.ub = value
        variable.value = [value]
    else:
        variable.value = [value]
