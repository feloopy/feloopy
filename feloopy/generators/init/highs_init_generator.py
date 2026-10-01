# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import highspy


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def _build_solution(model_object, col_vals):
    """Build a HighsSolution from {col_idx: value} and set it on the model."""
    n = model_object.getNumCol()
    vals = [0.0] * n
    for col_idx, value in col_vals.items():
        vals[col_idx] = float(value)
    sol = highspy.HighsSolution()
    sol.col_value = vals
    model_object.setSolution(sol)


def set_init_value(features, variable, value, fix=False):
    """Set a single variable's warm-start value or fix its bounds.

    Warm-start values are staged in a pending dict on features and flushed
    as one complete solution by set_init_values or flush_pending.
    """
    model_object = _get_model(features)
    col_idx = variable.index

    if fix:
        model_object.changeColBounds(col_idx, value, value)
    else:
        pending = features.setdefault('_highs_pending_init', {})
        pending[col_idx] = float(value)

    return model_object


def set_init_values(features, assignments, fix=False):
    """Batch apply (variable, scalar_value) pairs as a single solution."""
    model_object = _get_model(features)

    # Merge pending single-var values staged by set_init_value
    pending = features.pop('_highs_pending_init', {})
    col_vals = dict(pending)

    if fix:
        for variable, value in assignments:
            model_object.changeColBounds(variable.index, value, value)
        return model_object

    for variable, value in assignments:
        col_vals[variable.index] = float(value)

    if col_vals:
        _build_solution(model_object, col_vals)
    return model_object


def flush_pending(features):
    """Apply any staged single-var warm-start values as one solution."""
    pending = features.pop('_highs_pending_init', None)
    if not pending:
        return
    model_object = _get_model(features)
    _build_solution(model_object, pending)
