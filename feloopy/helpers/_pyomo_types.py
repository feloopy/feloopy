# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Eager registration of feloopy numeric proxies with pyomo.
"""

def register_pyomo_numeric_types():
    """Register feloopy scalar proxies as pyomo numeric types.

    Safe to call repeatedly and safe to call when pyomo is not installed: every
    import is guarded, and registration itself cannot fail in a way that should
    break model generation.
    """
    try:
        from pyomo.common.numeric_types import RegisterNumericType
    except Exception:
        return
    try:
        from ..operators.data_handler import DataRef
    except Exception:
        return
    try:
        RegisterNumericType(DataRef)
    except Exception:
        pass
