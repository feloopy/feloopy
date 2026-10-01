# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""GAMSPY/GAMS warm start.

``Variable.level`` is the GAMS ``.l`` attribute, which GAMS forwards to the
solver as its starting point.
"""


def set_init_value(features, variable, value, fix):
    if fix:
        variable.level = float(value)
        variable.lo = float(value)
        variable.up = float(value)
    else:
        variable.level = float(value)
