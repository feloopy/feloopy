# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def to_indexed_dict(value):
    """Return *value* as a plain ``dict`` when it is a dict-like variable
    container, otherwise ``None``.
    """
    if isinstance(value, dict):
        return value
    try:
        items = getattr(value, 'items', None)
        if callable(items):
            return dict(items())
    except Exception:
        return None
    return None


def as_var_group(value):
    """Normalize a freshly created variable into an ``{index: var}`` mapping.
    """
    container = to_indexed_dict(value)
    if container is not None:
        return container
    return {None: value}
