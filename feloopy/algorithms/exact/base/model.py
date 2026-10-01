# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Shared model utilities for decomposition algorithms.

Provides solver-neutral helpers for creating, solving, and inspecting
feloopy models used across Benders, Column Generation, Lagrangian, and
Multi-level algorithms.
"""

# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_fn, directions, obj_index, solver_opts,
                 level_name='master'):
    """Build a feloopy model from a user-supplied factory function.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(m) -> m`` that builds the model.
    directions : list[str]
        Optimization direction, e.g. ``['min']``.
    obj_index : int
        Which objective to optimise (default 0).
    solver_opts : dict
        Solver configuration keys (``interface_name``, ``solver_name``, …).
    level_name : str
        ``'master'`` or ``'subproblem'`` — controls which solver keys are used.

    Returns
    -------
    m : feloopy.model
        Configured model ready to solve.
    """
    from feloopy.feloopy import model as _feloopy_model

    if level_name == 'subproblem':
        interface = solver_opts.get(
            'sub_interface_name',
            solver_opts.get('interface_name', 'highs'))
        solver_name = solver_opts.get(
            'sub_solver_name',
            solver_opts.get('solver_name', interface))
    else:
        interface = solver_opts.get('interface_name', 'highs')
        solver_name = solver_opts.get('solver_name', interface)

    m = _feloopy_model(interface=interface, validate=False)
    m = model_fn(m)
    m.features['directions'] = directions
    m.features['objective_being_optimized'] = obj_index
    m.features['solver_name'] = solver_name
    for key, val in solver_opts.items():
        m.features[key] = val
    m.features['model_object_before_solve'] = m.model
    return m


# ---------------------------------------------------------------------------
# Model solving
# ---------------------------------------------------------------------------

def solve_model(m, solution_generator, constraint_ids=None,
                relax_integrality_flag=False):
    """Solve a feloopy model with optional constraint filtering.

    Parameters
    ----------
    m : feloopy.model
        Model to solve.
    solution_generator : object
        FelooPy solution generator.
    constraint_ids : list[str] | None
        If given, only keep constraints whose labels are in this set.
    relax_integrality_flag : bool
        When *True*, temporarily relax integer/binary variables.

    Returns
    -------
    bool
        ``True`` if the solve reported healthy.
    """
    original_constraints = None
    original_labels = None

    if constraint_ids is not None:
        original_constraints = list(m.features['constraints'])
        original_labels = list(m.features['constraint_labels'])
        filtered_c = []
        filtered_l = []
        for c, label in zip(original_constraints, original_labels):
            if label in constraint_ids:
                filtered_c.append(c)
                filtered_l.append(label)
        m.features['constraints'] = filtered_c
        m.features['constraint_labels'] = filtered_l

    if relax_integrality_flag:
        relax_integrality(m)

    if m.features.get('_cached_highs'):
        import timeit as _ti
        t0 = _ti.default_timer()
        m.model.run()
        t1 = _ti.default_timer()
        m.solution = m.model.getSolution(), [t0, t1]
        _cache_highs_lp_data(m)
    else:
        m.solution = solution_generator.generate_solution(m.features)

    if original_constraints is not None:
        m.features['constraints'] = original_constraints
        m.features['constraint_labels'] = original_labels

    return m.healthy()


def _cache_highs_lp_data(m):
    """Cache LP matrix data from a HiGHS model for later dual extraction."""
    try:
        lp = m.model.getLp()
        m.features['lp_data'] = {
            'n_cols': lp.num_col_,
            'n_rows': lp.num_row_,
            'row_lower': list(lp.row_lower_),
            'row_upper': list(lp.row_upper_),
            'col_lower': list(lp.col_lower_),
            'col_upper': list(lp.col_upper_),
            'col_cost': list(lp.col_cost_),
            'integrality': list(lp.integrality_),
            'col_names': list(lp.col_names_) if lp.col_names_ else [],
            'row_names': list(lp.row_names_) if lp.row_names_ else [],
            'A_col_pointers': list(lp.a_matrix_.start_),
            'A_row_indices': list(lp.a_matrix_.index_),
            'A_values': list(lp.a_matrix_.value_),
        }
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Integrality relaxation
# ---------------------------------------------------------------------------

def relax_integrality(m):
    """Temporarily relax all integer/binary variables to continuous.

    Supports HiGHS, Gurobi, CPLEX, Xpress, PuLP, COPT, SCIP, and
    a generic fallback.
    """
    interface = m.features.get('interface_name', '')

    if interface == 'highs':
        _relax_highs(m)
        return

    for var_key, var_obj in m.features.get('variables', {}).items():
        var_type = var_key[0]
        if var_type not in ('ivar', 'bvar'):
            continue
        _relax_generic_var(m, interface, var_key, var_obj)

    _trigger_model_update(m)


def _relax_highs(m):
    import highspy
    for var_key, var_obj in m.features.get('variables', {}).items():
        if var_key[0] not in ('ivar', 'bvar'):
            continue
        if isinstance(var_obj, dict):
            for idx, v in var_obj.items():
                try:
                    col_idx = v.index if hasattr(v, 'index') else v._col
                    m.model.changeColIntegrality(
                        col_idx, highspy.HighsVarType.kContinuous)
                except Exception:
                    pass
        elif hasattr(var_obj, 'index'):
            try:
                m.model.changeColIntegrality(
                    var_obj.index, highspy.HighsVarType.kContinuous)
            except Exception:
                pass


def _relax_generic_var(m, interface, var_key, var_obj):
    if interface == 'gurobi':
        _set_gurobi_vtype(var_obj, 'C')
    elif interface == 'cplex':
        _set_cplex_vtype(var_obj, 'C')
    elif interface == 'xpress':
        _set_xpress_type(var_obj, 'C')
    elif interface == 'pulp':
        _set_pulp_cat(var_obj, 'Continuous')
    elif interface == 'copt':
        _set_copt_type(var_obj, 0)
    elif interface == 'scip':
        _set_scip_type(m, var_obj)
    else:
        _set_fallback_vtype(var_obj)


def _set_gurobi_vtype(var_obj, vtype):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                v.setAttr('VType', vtype)
            except Exception:
                try:
                    v.VType = vtype
                except Exception:
                    pass
    else:
        try:
            var_obj.setAttr('VType', vtype)
        except Exception:
            try:
                var_obj.VType = vtype
            except Exception:
                pass
        if hasattr(var_obj, '__iter__'):
            try:
                for v in var_obj.flat:
                    try:
                        v.setAttr('VType', vtype)
                    except Exception:
                        try:
                            v.VType = vtype
                        except Exception:
                            pass
            except Exception:
                pass


def _set_cplex_vtype(var_obj, vtype):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                v.set_vtype(vtype)
            except Exception:
                pass
    else:
        try:
            if hasattr(var_obj, 'set_vtype'):
                var_obj.set_vtype(vtype)
        except Exception:
            pass


def _set_xpress_type(var_obj, vtype):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                v.setAttrib('Type', vtype)
            except Exception:
                pass


def _set_pulp_cat(var_obj, cat):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                v.cat = cat
            except Exception:
                pass
    else:
        try:
            if hasattr(var_obj, 'cat'):
                var_obj.cat = cat
        except Exception:
            pass


def _set_copt_type(var_obj, vtype):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                v.Type = vtype
            except Exception:
                pass


def _set_scip_type(m, var_obj):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                m.model.chgVarType(v, 'CONTINUOUS')
            except Exception:
                pass


def _set_fallback_vtype(var_obj):
    if isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                if hasattr(v, 'VType'):
                    v.VType = 'C'
                elif hasattr(v, 'cat'):
                    v.cat = 'Continuous'
            except Exception:
                pass
    elif hasattr(var_obj, '__iter__'):
        try:
            for v in var_obj.flat:
                if hasattr(v, 'VType'):
                    v.VType = 'C'
                elif hasattr(v, 'cat'):
                    v.cat = 'Continuous'
        except Exception:
            pass
    else:
        try:
            if hasattr(var_obj, 'VType'):
                var_obj.VType = 'C'
            elif hasattr(var_obj, 'cat'):
                var_obj.cat = 'Continuous'
        except Exception:
            pass


def _trigger_model_update(m):
    try:
        update = getattr(m.model, 'update', None)
        if callable(update):
            update()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Variable / objective / status / dual accessors
# ---------------------------------------------------------------------------

def get_var_value(m, var_name):
    """Extract the value of a variable from a solved model."""
    try:
        return m.get_numpy_var(var_name)
    except Exception:
        return None


def get_obj(m):
    """Return the objective value of a solved model, or *None*."""
    try:
        return m.get_objective()
    except Exception:
        return None


def get_status(m):
    """Return the status string of a solved model."""
    try:
        return m.get_status()
    except Exception:
        return 'unknown'


def get_dual(m, constraint_label):
    """Return the dual value for a named constraint, or *None*."""
    try:
        return m.get_dual(constraint_label)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HiGHS / Gurobi bound-based variable fixing
# ---------------------------------------------------------------------------

def build_var_col_map(m):
    """Map ``(prefix, name) -> col_idx`` for all variables in *m*.

    Supports HiGHS and Gurobi native models.
    """
    mapping = {}
    model_obj = m.model
    interface = m.features.get('interface_name', '')

    if interface == 'highs':
        n_cols = model_obj.getNumCol()
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            for (p, name), var_obj in m.features.get('variables', {}).items():
                if p != prefix:
                    continue
                if isinstance(var_obj, dict):
                    mapping[(prefix, name)] = {}
                    for idx, v in var_obj.items():
                        try:
                            col_idx = v.index if hasattr(v, 'index') else v._col
                            mapping[(prefix, name)][idx] = col_idx
                        except Exception:
                            pass
                else:
                    try:
                        col_idx = (var_obj.index if hasattr(var_obj, 'index')
                                   else var_obj._col)
                        mapping[(prefix, name)] = col_idx
                    except Exception:
                        pass
    elif interface == 'gurobi':
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            for (p, name), var_obj in m.features.get('variables', {}).items():
                if p != prefix:
                    continue
                if isinstance(var_obj, dict):
                    mapping[(prefix, name)] = {}
                    for idx, v in var_obj.items():
                        try:
                            mapping[(prefix, name)][idx] = v.index
                        except Exception:
                            pass
                else:
                    try:
                        mapping[(prefix, name)] = var_obj.index
                    except Exception:
                        pass

    return mapping


def capture_original_bounds(m, var_col_map, shared_vars):
    """Save original bounds for shared variables before fixing."""
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    orig = {}

    for var_name in shared_vars:
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            if interface == 'highs':
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        try:
                            b = model_obj.getColBounds(col_idx)
                            orig[(prefix, var_name, idx)] = (b[0], b[1])
                        except Exception:
                            pass
                else:
                    try:
                        b = model_obj.getColBounds(col_info)
                        orig[(prefix, var_name)] = (b[0], b[1])
                    except Exception:
                        pass
            elif interface == 'gurobi':
                if isinstance(col_info, dict):
                    for idx, v in col_info.items():
                        try:
                            orig[(prefix, var_name, idx)] = (v.LB, v.UB)
                        except Exception:
                            pass
                else:
                    try:
                        orig[(prefix, var_name)] = (col_info.LB, col_info.UB)
                    except Exception:
                        pass
            break
    return orig


def fix_variable_bound(m, var_col_map, var_name, value):
    """Fix a variable to a specific value via model bounds."""
    interface = m.features.get('interface_name', '')
    model_obj = m.model

    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        col_info = var_col_map.get((prefix, var_name))
        if col_info is None:
            continue

        if interface == 'highs':
            if isinstance(col_info, dict):
                for idx, col_idx in col_info.items():
                    try:
                        model_obj.changeColBounds(col_idx, float(value[idx]),
                                                   float(value[idx]))
                    except Exception:
                        pass
            else:
                try:
                    model_obj.changeColBounds(col_info, float(value),
                                               float(value))
                except Exception:
                    pass
        elif interface == 'gurobi':
            if isinstance(col_info, dict):
                for idx, v in col_info.items():
                    try:
                        v.LB = float(value[idx])
                        v.UB = float(value[idx])
                    except Exception:
                        pass
            else:
                try:
                    col_info.LB = float(value)
                    col_info.UB = float(value)
                except Exception:
                    pass
        break


def unfix_variable_bound(m, var_col_map, var_name, original_bounds):
    """Restore original bounds for a previously fixed variable."""
    interface = m.features.get('interface_name', '')
    model_obj = m.model

    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        col_info = var_col_map.get((prefix, var_name))
        if col_info is None:
            continue

        if interface == 'highs':
            if isinstance(col_info, dict):
                for idx, col_idx in col_info.items():
                    try:
                        lb, ub = original_bounds.get(
                            (prefix, var_name, idx), (-1e20, 1e20))
                        model_obj.changeColBounds(col_idx, lb, ub)
                    except Exception:
                        pass
            else:
                try:
                    lb, ub = original_bounds.get(
                        (prefix, var_name), (-1e20, 1e20))
                    model_obj.changeColBounds(col_info, lb, ub)
                except Exception:
                    pass
        elif interface == 'gurobi':
            if isinstance(col_info, dict):
                for idx, v in col_info.items():
                    try:
                        lb, ub = original_bounds.get(
                            (prefix, var_name, idx), (-1e20, 1e20))
                        v.LB = lb
                        v.UB = ub
                    except Exception:
                        pass
            else:
                try:
                    lb, ub = original_bounds.get(
                        (prefix, var_name), (-1e20, 1e20))
                    col_info.LB = lb
                    col_info.UB = ub
                except Exception:
                    pass
        break
