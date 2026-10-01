# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast
import time
import numpy as np

from .enums import BendersStatus
from .result import BendersResult
from ....helpers.containers import to_indexed_dict


# ---------------------------------------------------------------------------
# Variable fixing (constraint-based, legacy)
# ---------------------------------------------------------------------------

def _relax_integrality(m):
    """Temporarily relax all integer/binary variables to continuous."""
    interface = m.features.get('interface_name', '')

    if interface == 'highs':
        model_obj = m.model
        try:
            import highspy
            for var_key, var_obj in m.features.get('variables', {}).items():
                if var_key[0] not in ('ivar', 'bvar'):
                    continue
                if isinstance(var_obj, dict):
                    for idx, v in var_obj.items():
                        try:
                            col_idx = v.index if hasattr(v, 'index') else v._col
                            model_obj.changeColIntegrality(
                                col_idx, highspy.HighsVarType.kContinuous)
                        except Exception:
                            pass
                elif hasattr(var_obj, 'index'):
                    try:
                        model_obj.changeColIntegrality(
                            var_obj.index, highspy.HighsVarType.kContinuous)
                    except Exception:
                        pass
        except ImportError:
            pass
        return

    for var_key, var_obj in m.features.get('variables', {}).items():
        var_type = var_key[0]
        if var_type not in ('ivar', 'bvar'):
            continue

        if interface == 'gurobi':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.setAttr('VType', 'C')
                    except Exception:
                        try:
                            v.VType = 'C'
                        except Exception:
                            pass
            else:
                try:
                    var_obj.setAttr('VType', 'C')
                except Exception:
                    try:
                        var_obj.VType = 'C'
                    except Exception:
                        pass
                    if hasattr(var_obj, '__iter__'):
                        try:
                            for v in var_obj.flat:
                                try:
                                    v.setAttr('VType', 'C')
                                except Exception:
                                    try:
                                        v.VType = 'C'
                                    except Exception:
                                        pass
                        except Exception:
                            pass

        elif interface == 'cplex':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.set_vtype('C')
                    except Exception:
                        pass
            else:
                try:
                    if hasattr(var_obj, 'set_vtype'):
                        var_obj.set_vtype('C')
                except Exception:
                    pass

        elif interface == 'xpress':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.setAttrib('Type', 'C')
                    except Exception:
                        pass

        elif interface == 'pulp':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.cat = 'Continuous'
                    except Exception:
                        pass
            else:
                try:
                    if hasattr(var_obj, 'cat'):
                        var_obj.cat = 'Continuous'
                except Exception:
                    pass

        elif interface == 'copt':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        v.Type = 0
                    except Exception:
                        pass

        elif interface == 'scip':
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    try:
                        m.model.chgVarType(v, 'CONTINUOUS')
                    except Exception:
                        pass

        else:
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

        try:
            update = getattr(m.model, 'update', None)
            if callable(update):
                update()
        except Exception:
            pass


def _fix_variable(m, var_name, value):
    """Fix a variable to a specific value via constraints."""
    var_obj = None
    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        var_obj = m.features['variables'].get((prefix, var_name))
        if var_obj is not None:
            break

    if var_obj is None:
        return

    # pyomo's IndexedVar and COPT's tupledict are dict-like containers
    # rather than dict subclasses; comparing the container itself against
    # a scalar falls back to object equality and would store a constant
    # False row.  to_indexed_dict guards COPT's Var, which raises
    # CoptError (not AttributeError) on unknown attribute lookups.
    if not isinstance(var_obj, dict):
        container = to_indexed_dict(var_obj)
        if container is not None:
            var_obj = container

    is_indexed = hasattr(value, '__len__') and not isinstance(value, str)
    if is_indexed and isinstance(var_obj, dict):
        for idx, v in var_obj.items():
            try:
                val = float(value[idx])
            except Exception:
                continue
            try:
                m.con(v == val, name=f'_benders_fix_{var_name}_{idx}')
            except Exception:
                try:
                    m.con(v >= val,
                           name=f'_benders_fix_{var_name}_{idx}_lb')
                    m.con(v <= val,
                           name=f'_benders_fix_{var_name}_{idx}_ub')
                except Exception:
                    pass
    else:
        try:
            scalar_val = float(value)
        except Exception:
            return
        if isinstance(var_obj, dict):
            # scalar target for an indexed variable: fix every element
            for idx, v in var_obj.items():
                try:
                    m.con(v == scalar_val,
                          name=f'_benders_fix_{var_name}_{idx}')
                except Exception:
                    pass
            return
        try:
            m.con(var_obj == scalar_val,
                   name=f'_benders_fix_{var_name}')
        except Exception:
            try:
                m.con(var_obj >= scalar_val,
                       name=f'_benders_fix_{var_name}_lb')
                m.con(var_obj <= scalar_val,
                       name=f'_benders_fix_{var_name}_ub')
            except Exception:
                pass


def _build_var_col_map(m):
    """Map (prefix, name) -> col_idx for all variables in a model."""
    mapping = {}
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    try:
        if interface == 'gurobi':
            n_cols = len(model_obj.getVars())
        else:
            n_cols = model_obj.getNumCol()
    except Exception:
        n_cols = 0
    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        for (p, name), var_obj in m.features.get('variables', {}).items():
            if p != prefix:
                continue
            if isinstance(var_obj, dict):
                mapping[(prefix, name)] = {}
                for idx, v in var_obj.items():
                    try:
                        if interface == 'gurobi':
                            col_idx = v.index if hasattr(v, 'index') else None
                            if col_idx is None:
                                col_idx = model_obj.getVarByName(v.VarName).index if hasattr(v, 'VarName') else None
                        else:
                            col_idx = v.index if hasattr(v, 'index') else v._col
                        if col_idx is not None:
                            mapping[(prefix, name)][idx] = col_idx
                    except Exception:
                        pass
            else:
                try:
                    if interface == 'gurobi':
                        col_idx = var_obj.index if hasattr(var_obj, 'index') else None
                        if col_idx is None:
                            col_idx = model_obj.getVarByName(var_obj.VarName).index if hasattr(var_obj, 'VarName') else None
                    else:
                        col_idx = var_obj.index if hasattr(var_obj, 'index') else var_obj._col
                    if col_idx is not None:
                        mapping[(prefix, name)] = col_idx
                except Exception:
                    pass
    return mapping


def _fix_variable_bound(m, var_col_map, var_name, value):
    """Fix variable via bounds on the model (HiGHS or Gurobi)."""
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        col_info = var_col_map.get((prefix, var_name))
        if col_info is None:
            continue
        if isinstance(col_info, dict):
            for idx, col_idx in col_info.items():
                try:
                    val = float(value[idx])
                    if interface == 'gurobi':
                        var = model_obj.getVars()[col_idx]
                        var.lb = val
                        var.ub = val
                    else:
                        import highspy
                        model_obj.changeColBounds(col_idx, val, val)
                except Exception:
                    pass
        else:
            try:
                val = float(value)
                if interface == 'gurobi':
                    var = model_obj.getVars()[col_idx]
                    var.lb = val
                    var.ub = val
                else:
                    import highspy
                    model_obj.changeColBounds(col_info, val, val)
                model_obj.update()
            except Exception:
                pass
        return


def _unfix_variable_bound(m, var_col_map, var_name, original_bounds):
    """Restore original bounds for a variable."""
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        col_info = var_col_map.get((prefix, var_name))
        if col_info is None:
            continue
        if isinstance(col_info, dict):
            for idx, col_idx in col_info.items():
                try:
                    lb, ub = original_bounds.get(
                        (prefix, var_name, idx), (-1e20, 1e20))
                    if interface == 'gurobi':
                        var = model_obj.getVars()[col_idx]
                        var.lb = lb
                        var.ub = ub
                    else:
                        model_obj.changeColBounds(col_idx, lb, ub)
                except Exception:
                    pass
        else:
            try:
                lb, ub = original_bounds.get(
                    (prefix, var_name), (-1e20, 1e20))
                if interface == 'gurobi':
                    var = model_obj.getVars()[col_info]
                    var.lb = lb
                    var.ub = ub
                else:
                    model_obj.changeColBounds(col_info, lb, ub)
                model_obj.update()
            except Exception:
                pass
        return


def _capture_original_bounds(m, var_col_map, shared_vars):
    """Capture original bounds for shared variables before fixing."""
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    orig = {}
    for var_name in shared_vars:
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            if isinstance(col_info, dict):
                for idx, col_idx in col_info.items():
                    try:
                        if interface == 'gurobi':
                            var = model_obj.getVars()[col_idx]
                            orig[(prefix, var_name, idx)] = (var.lb, var.ub)
                        else:
                            b = model_obj.getColBounds(col_idx)
                            orig[(prefix, var_name, idx)] = (b[0], b[1])
                    except Exception:
                        pass
            else:
                try:
                    if interface == 'gurobi':
                        var = model_obj.getVars()[col_info]
                        orig[(prefix, var_name)] = (var.lb, var.ub)
                    else:
                        b = model_obj.getColBounds(col_info)
                        orig[(prefix, var_name)] = (b[0], b[1])
                except Exception:
                    pass
            break
    return orig


def _unfix_all_bounds(m, var_col_map, shared_vars, original_bounds):
    """Restore original bounds for all shared variables."""
    model_obj = m.model
    interface = m.features.get('interface_name', '')
    for var_name in shared_vars:
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            if isinstance(col_info, dict):
                for idx, col_idx in col_info.items():
                    try:
                        lb, ub = original_bounds.get(
                            (prefix, var_name, idx), (-1e20, 1e20))
                        if interface == 'gurobi':
                            var = model_obj.getVars()[col_idx]
                            var.lb = lb
                            var.ub = ub
                        else:
                            model_obj.changeColBounds(col_idx, lb, ub)
                    except Exception:
                        pass
            else:
                try:
                    lb, ub = original_bounds.get(
                        (prefix, var_name), (-1e20, 1e20))
                    if interface == 'gurobi':
                        var = model_obj.getVars()[col_info]
                        var.lb = lb
                        var.ub = ub
                    else:
                        model_obj.changeColBounds(col_info, lb, ub)
                except Exception:
                    pass
            break


def _extract_reduced_costs(sub_m, var_col_map, shared_vars, master_vars):
    """Extract Benders cut coefficients from variable reduced costs.

    When fixing is done via bounds, the reduced cost (col_dual) of each
    fixed variable equals the shadow price of the fixing -- the Benders
    cut coefficient.
    """
    model_obj = sub_m.model
    interface = sub_m.features.get('interface_name', '')

    col_duals = None
    if interface == 'gurobi':
        try:
            all_vars = model_obj.getVars()
            col_duals = [v.RC for v in all_vars]
        except Exception:
            return {}
    else:
        try:
            sol = model_obj.getSolution()
            col_duals = sol.col_dual
        except Exception:
            return {}

    if col_duals is None:
        return {}

    fix_duals = {}
    for var_name in shared_vars:
        val = master_vars.get(var_name)
        if val is None:
            continue
        for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
            col_info = var_col_map.get((prefix, var_name))
            if col_info is None:
                continue
            if isinstance(col_info, dict):
                is_indexed = hasattr(val, '__len__') and not isinstance(val, str)
                for idx, col_idx in col_info.items():
                    if col_idx < len(col_duals):
                        label = '_benders_fix_%s_%s' % (var_name, idx)
                        fix_duals[label] = float(col_duals[col_idx])
            else:
                if col_info < len(col_duals):
                    label = '_benders_fix_%s' % var_name
                    fix_duals[label] = float(col_duals[col_info])
            break
    return fix_duals


def _derive_numerical_coeffs(solve_single_subproblem_fn, level, master_vars, shared_vars,
                             solver_opts, solution_generator, show_log, base_obj):
    """Estimate recourse slopes when the backend exposes no fixing duals."""
    coeffs = {}
    for var_name in shared_vars:
        value = master_vars.get(var_name)
        if value is None:
            continue

        if isinstance(value, dict):
            value_items = list(value.items())
        elif isinstance(value, np.ndarray) and value.size > 1:
            value_items = [(idx[0] if len(idx) == 1 else idx, float(val))
                           for idx, val in np.ndenumerate(value)]
        else:
            center = float(np.asarray(value).reshape(-1)[0])
            is_binary = center in (0.0, 1.0)
            deltas = []
            if is_binary:
                deltas = [1.0 - center, 0.0 - center]
            else:
                deltas = [1.0, -1.0]
            for delta in deltas:
                if abs(delta) < 1e-12:
                    continue
                probe = center + delta
                probe_vars = dict(master_vars)
                probe_vars[var_name] = probe
                probe_result = solve_single_subproblem_fn(
                    level, probe_vars, shared_vars, solver_opts,
                    solution_generator, show_log,
                    derive_numerical_cut=False)
                if probe_result['feasible']:
                    coeffs[var_name] = (
                        (probe_result['objective_value'] - base_obj) / delta)
                    break
            continue

        for idx, center in value_items:
            center = float(center)
            is_binary = center in (0.0, 1.0)
            deltas = []
            if is_binary:
                deltas = [1.0 - center, 0.0 - center]
            else:
                deltas = [1.0, -1.0]
            for delta in deltas:
                if abs(delta) < 1e-12:
                    continue
                probe_vars = {}
                for k, v in master_vars.items():
                    if k == var_name:
                        if isinstance(v, dict):
                            probe_vars[k] = dict(v)
                            probe_vars[k][idx] = center + delta
                        elif isinstance(v, np.ndarray):
                            probe_vars[k] = v.copy().astype(float)
                            flat_idx = int(idx) if isinstance(idx, int) else int(idx[0])
                            probe_vars[k].flat[flat_idx] = center + delta
                        else:
                            probe_vars[k] = center + delta
                    else:
                        probe_vars[k] = v
                probe_result = solve_single_subproblem_fn(
                    level, probe_vars, shared_vars, solver_opts,
                    solution_generator, show_log,
                    derive_numerical_cut=False)
                if probe_result['feasible']:
                    key = (var_name, idx)
                    coeffs[key] = (
                        (probe_result['objective_value'] - base_obj) / delta)
                    break
    return coeffs


def _compute_benders_coeffs_from_subduals(build_var_col_map_fn, get_var_col_indices_fn,
                                          sub_m, sub_duals, shared_vars, master_vars):
    """Compute Benders coefficients from constraint duals + matrix."""
    lp_data = sub_m.features.get('lp_data', {})
    row_names = lp_data.get('row_names', [])
    A_col_ptrs = lp_data.get('A_col_pointers', [])
    A_row_idx = lp_data.get('A_row_indices', [])
    A_vals = lp_data.get('A_values', [])
    n_cols = lp_data.get('n_cols', 0)

    if not A_col_ptrs or not row_names or not sub_duals:
        return {}

    label_to_row = {name: i for i, name in enumerate(row_names) if name}
    row_to_label = {i: name for name, i in label_to_row.items()}

    var_col_map = build_var_col_map_fn(sub_m)

    fix_duals = {}
    for var_name in shared_vars:
        col_indices = get_var_col_indices_fn(sub_m, var_name, var_col_map)
        if not col_indices:
            continue
        for idx, col_idx in col_indices.items():
            if col_idx is None or col_idx >= n_cols:
                continue
            start = int(A_col_ptrs[col_idx])
            end = int(A_col_ptrs[col_idx + 1]) if col_idx + 1 < len(A_col_ptrs) else len(A_row_idx)
            coeff = 0.0
            for k in range(start, end):
                row_i = int(A_row_idx[k])
                a_val = float(A_vals[k])
                lbl = row_to_label.get(row_i)
                if lbl is None:
                    continue
                pi = sub_duals.get(lbl)
                if pi is None:
                    continue
                coeff += float(pi) * a_val
            if abs(coeff) > 1e-12:
                fix_duals[f'_benders_fix_{var_name}_{idx}'] = -coeff
    return fix_duals


# ------------------------------------------------------------------
# Direct HiGHS operations (bypass feloopy overhead)
# ------------------------------------------------------------------

def _add_cut_direct_highs(m, cut, theta_col_idx, var_col_map):
    """Add a Benders cut directly to HiGHS model as a row."""
    import numpy as np
    model_obj = m.model
    indices = []
    values = []

    if cut.get('type') == 'feasibility':
        for var_name, coeff in cut.get('coeffs', {}).items():
            for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                col_info = var_col_map.get((prefix, var_name))
                if col_info is None:
                    continue
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        c = coeff.get(idx, 0.0) if isinstance(coeff, dict) else coeff
                        if abs(c) > 1e-15:
                            indices.append(col_idx)
                            values.append(c)
                else:
                    if abs(coeff) > 1e-15:
                        indices.append(col_info)
                        values.append(coeff)
                break
        lower = cut['rhs']
        upper = 1e20
    else:
        if theta_col_idx is not None:
            indices.append(theta_col_idx)
            values.append(1.0)
        for var_name, coeff in cut.get('coeffs', {}).items():
            for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                col_info = var_col_map.get((prefix, var_name))
                if col_info is None:
                    continue
                if isinstance(coeff, dict):
                    if isinstance(col_info, dict):
                        for idx, col_idx in col_info.items():
                            c = coeff.get(idx, 0.0)
                            if abs(c) > 1e-15:
                                indices.append(col_idx)
                                values.append(-c)
                elif isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        if abs(coeff) > 1e-15:
                            indices.append(col_idx)
                            values.append(-coeff)
                else:
                    if abs(coeff) > 1e-15:
                        indices.append(col_info)
                        values.append(-coeff)
                break
        lower = cut['rhs']
        upper = 1e20

    if indices:
        model_obj.addRow(lower, upper, len(indices), indices, values)


def _get_reduced_costs_highs(m):
    """Extract reduced costs from HiGHS solution."""
    try:
        sol = m.model.getSolution()
        return {i: sol.col_dual[i] for i in range(m.model.getNumCol())}
    except Exception:
        return {}


def _get_theta_col_idx(m, var_col_map):
    """Get column index of theta variable."""
    for name in ('theta', '_benders_theta'):
        for prefix in ('fvar', 'pvar'):
            col_info = var_col_map.get((prefix, name))
            if col_info is not None and not isinstance(col_info, dict):
                return col_info
    return None


def _get_var_col_indices(m, var_name, var_col_map):
    """Get column indices for a variable."""
    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
        col_info = var_col_map.get((prefix, var_name))
        if col_info is not None:
            if isinstance(col_info, dict):
                return col_info
            return {0: col_info}
    return {}


def _compute_iis(sub_m):
    """Compute Irreducible Infeasible Subsystem (IIS) of an infeasible subproblem."""
    interface = sub_m.features.get('interface_name', '')
    iis_vars = []

    if interface == 'gurobi':
        try:
            sub_m.model.computeIIS()
            constrs = sub_m.model.getConstrs()
            for c in constrs:
                if c.IISConstr:
                    row = sub_m.model.getRow(c)
                    for j in range(row.size()):
                        var = row.getVar(j)
                        vname = var.VarName
                        if vname:
                            iis_vars.append(vname)
        except Exception:
            pass
    elif interface == 'cplex':
        try:
            sub_m.model.populate()
            iis = sub_m.model.conflict.get(0)
            for status, col in zip(iis.indicators, iis.sub_cols):
                if status == 1:
                    iis_vars.append(sub_m.model.getColName(col))
        except Exception:
            pass

    return list(set(iis_vars))


# ------------------------------------------------------------------
# Result packaging
# ------------------------------------------------------------------

def _finalize_result(ub, lb, variables, iterations,
                     bounds_history, time_start, n_opt_cuts, n_feas_cuts,
                     status, obj_history=None, n_sol=0):
    """Package results into a BendersResult."""
    elapsed = time.perf_counter() - time_start
    result = BendersResult(
        status=status,
        objective=ub,
        variables=variables or {},
        iterations=iterations,
        bounds=bounds_history,
        n_optimality_cuts=n_opt_cuts,
        n_feasibility_cuts=n_feas_cuts,
        n_cuts=n_opt_cuts + n_feas_cuts,
        n_sol=n_sol,
        runtime_total=elapsed,
        gap_abs=abs(ub - lb) if ub < float('inf') and lb > float('-inf') else float('inf'),
        gap_rel=abs(ub - lb) / abs(ub) if ub != 0 and ub < float('inf') and lb > float('-inf') else float('inf'),
        lb_history=[b[0] for b in bounds_history],
        ub_history=[b[1] for b in bounds_history],
        obj_history=obj_history or [],
    )
    return result


def _build_result(objective, variables, iterations, bounds_history,
                  status, n_opt_cuts=0, n_feas_cuts=0, runtime=0.0,
                  obj_history=None, n_sol=0):
    from time import perf_counter
    return BendersResult(
        status=status if isinstance(status, BendersStatus) else (
            BendersStatus.OPTIMAL if status == 'optimal' else
            BendersStatus.MAX_ITERATIONS if status == 'max_iterations' else
            BendersStatus.INFEASIBLE if status == 'infeasible' else
            BendersStatus.UNSOLVED),
        objective=objective,
        variables=variables or {},
        iterations=iterations,
        bounds=bounds_history,
        n_optimality_cuts=n_opt_cuts,
        n_feasibility_cuts=n_feas_cuts,
        n_cuts=n_opt_cuts + n_feas_cuts,
        n_sol=n_sol,
        runtime_total=runtime,
        lb_history=[b[0] for b in bounds_history],
        ub_history=[b[1] for b in bounds_history],
        obj_history=obj_history or [],
    )


def get_convergence(result):
    """Return convergence history from the last Benders solve.

    Returns
    -------
    dict with keys:
        'lb_history': list[float] -- lower bound per iteration
        'ub_history': list[float] -- upper bound per iteration
        'obj_history': list[float] -- incumbent objective per iteration
        'gap_abs_history': list[float] -- |UB - LB| per iteration
        'gap_rel_history': list[float] -- |UB - LB| / |UB| per iteration
        'iterations': int -- total iterations executed
        'status': str -- termination status ('optimal', 'max_iterations',
                        'timeout', 'infeasible', 'unsolved')
        'n_optimality_cuts': int
        'n_feasibility_cuts': int
        'n_cuts': int
        'n_sol': int -- number of feasible solutions found
        'runtime_total': float -- wall-clock seconds
        'objective': float -- final objective value
    """
    if result is None:
        return {
            'lb_history': [], 'ub_history': [], 'obj_history': [],
            'gap_abs_history': [], 'gap_rel_history': [],
            'iterations': 0, 'status': 'unsolved',
            'n_optimality_cuts': 0, 'n_feasibility_cuts': 0,
            'n_cuts': 0, 'n_sol': 0, 'runtime_total': 0.0,
            'objective': float('inf'),
        }
    lb = list(result.lb_history)
    ub = list(result.ub_history)
    gap_abs = [abs(u - l) for u, l in zip(ub, lb)]
    gap_rel = [
        abs(u - l) / abs(u) if abs(u) > 1e-15 else 0.0
        for u, l in zip(ub, lb)
    ]
    status_str = result.status
    if hasattr(status_str, 'value'):
        status_str = status_str.value
    return {
        'lb_history': lb,
        'ub_history': ub,
        'obj_history': list(result.obj_history) if result.obj_history else [],
        'gap_abs_history': gap_abs,
        'gap_rel_history': gap_rel,
        'iterations': result.iterations,
        'status': status_str,
        'n_optimality_cuts': result.n_optimality_cuts,
        'n_feasibility_cuts': result.n_feasibility_cuts,
        'n_cuts': result.n_cuts,
        'n_sol': result.n_sol,
        'runtime_total': result.runtime_total,
        'objective': result.objective,
    }
