# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def set_init_value(features, variable, value, fix=False):
    """Set a single variable's warm-start value or fix its bounds.

    Warm-start values are staged in a pending solution stored on features
    and flushed as one complete solution by set_init_values or flush_pending.
    """
    model_object = _get_model(features)

    if fix:
        variable.lb = float(value)
        variable.ub = float(value)
    else:
        pending = features.setdefault('_scip_pending_init', {})
        pending[id(variable)] = (variable, float(value))

    return model_object


def set_init_values(features, assignments, fix=False):
    """Batch apply (variable, scalar_value) pairs as a single SCIP solution."""
    model_object = _get_model(features)

    # Merge pending single-var values staged by set_init_value
    pending = features.pop('_scip_pending_init', {})
    all_assignments = list(pending.values()) + list(assignments)

    if fix:
        for variable, value in all_assignments:
            variable.lb = float(value)
            variable.ub = float(value)
        return model_object

    if not all_assignments:
        return model_object

    sol = model_object.createSol()
    for variable, value in all_assignments:
        model_object.setSolVal(sol, variable, float(value))
    model_object.addSol(sol, free=True)
    return model_object


def flush_pending(features):
    """Apply any staged single-var warm-start values as one solution."""
    pending = features.get('_scip_pending_init')
    if not pending:
        return
    assignments = list(pending.values())
    features.pop('_scip_pending_init', None)
    model_object = _get_model(features)
    sol = model_object.createSol()
    for variable, value in assignments:
        model_object.setSolVal(sol, variable, float(value))
    model_object.addSol(sol, free=True)
