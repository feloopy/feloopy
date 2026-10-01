# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import ast
import math
import numpy as np


# ---------------------------------------------------------------------------
# Cut normalization utility
# ---------------------------------------------------------------------------

def _normalize_cut(cut, max_norm):
    """Normalize a cut dict's coeffs and rhs so the L2 norm of coeffs <= max_norm."""
    coeffs = cut.get('coeffs', {})
    if not coeffs:
        return cut
    flat = []
    for v in coeffs.values():
        if isinstance(v, dict):
            flat.extend(v.values())
        else:
            flat.append(v)
    norm = math.sqrt(sum(x * x for x in flat))
    if norm <= max_norm or norm < 1e-15:
        return cut
    scale = max_norm / norm
    new_coeffs = {}
    for k, v in coeffs.items():
        if isinstance(v, dict):
            new_coeffs[k] = {idx: c * scale for idx, c in v.items()}
        else:
            new_coeffs[k] = v * scale
    return {**cut, 'coeffs': new_coeffs, 'rhs': cut.get('rhs', 0) * scale}


def _normalize_cut_list(cuts, max_norm):
    return [_normalize_cut(c, max_norm) for c in cuts]


def _cut_signature(cut, tol=1e-8):
    """Create a hashable signature for duplicate detection."""
    coeffs = cut.get('coeffs', {})
    items = []
    for var_name, coeff in sorted(coeffs.items()):
        if isinstance(coeff, dict):
            items.append((var_name, tuple(sorted((i, round(c / tol)) for i, c in coeff.items()))))
        else:
            items.append((var_name, round(coeff / tol)))
    return (cut.get('type', ''), round(cut.get('rhs', 0) / tol), tuple(items), cut.get('sense', '>='))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_benders(msg, show_log):
    if not show_log:
        return
    # Suppress verbose per-subproblem diagnostics for clean tabular log
    if isinstance(msg, str) and msg.lstrip().startswith(
        ("Solving subproblem", "Subproblem", "IIS nogood",
         "Re-solving", "Artificial-var", "Gurobi",
         "Optimality cut", "Initial x_bar")):
        return
    if isinstance(msg, str) and msg.startswith("  "):
        stripped = msg.lstrip()
        if stripped.startswith(("Solving", "Subproblem", "IIS", "Re-solving",
                               "Artificial", "Gurobi", "Optimality", "Initial")):
            return
    print(f"{msg}", flush=True)


def _log_iter(iteration, lb, ub, gap, n_cuts, n_opt, n_feas, elapsed, show_log):
    if not show_log:
        return
    print(
        f"  {iteration:>5d}  "
        f"{lb:>12.4f}  {ub:>12.4f}  "
        f"{gap:>10.4f}  "
        f"{n_cuts:>4d} (+{n_opt}/{n_feas})  "
        f"{elapsed:.1f}s",
        flush=True)


# ---------------------------------------------------------------------------
# Cut generation: classical
# ---------------------------------------------------------------------------

def _generate_optimality_cut(sub_result, master_vars, complicating_vars):
    """Generate a Benders optimality cut from subproblem duals.

    For minimization, the standard Benders cut is:
        theta >= Q(x_bar) + sum_j c_j * (x_j - x_bar_j)

    where c_j = sum of linking-constraint duals involving x_j.
    By LP sensitivity, the fixing-constraint dual mu[j] = -c_j,
    so c_j = -mu[j].

    Equivalent form:
        theta >= rhs + sum_j c_j * x_j
    where rhs = Q(x_bar) - sum_j c_j * x_bar_j = Q(x_bar) + sum_j mu[j] * x_bar_j
    """
    fix_duals = sub_result.get('fix_duals', {})
    numerical_coeffs = sub_result.get('numerical_coeffs', {})
    no_useful_fix_duals = (
        not fix_duals or
        all(abs(float(value)) <= 1e-12 for value in fix_duals.values()))
    if no_useful_fix_duals and numerical_coeffs:
        rhs = sub_result.get('objective_value', 0.0)
        coeffs = {}
        for key, coeff in numerical_coeffs.items():
            if isinstance(key, tuple):
                var_name, idx = key
                y_star = master_vars.get(var_name, {})
                if isinstance(y_star, dict):
                    x_val = float(y_star.get(idx, 0))
                else:
                    x_val = float(np.asarray(y_star).flatten()[idx]) if np.asarray(y_star).size > idx else 0.0
                if var_name not in coeffs:
                    coeffs[var_name] = {}
                coeffs[var_name][idx] = coeff
            else:
                var_name = key
                x_val = float(np.asarray(
                    master_vars.get(var_name, 0)).reshape(-1)[0])
                coeffs[var_name] = coeff
            rhs -= coeff * x_val
        return {
            'rhs': rhs,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_result.get('objective_value', 0.0),
        }
    if not fix_duals:
        sub_obj = sub_result.get('objective_value', 0.0)
        return {
            'rhs': sub_obj,
            'coeffs': {},
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    sub_obj = sub_result.get('objective_value', 0.0)
    fixing_vars = sub_result.get('fixing_vars', [])
    fixing_vals = sub_result.get('fixing_vals', {})

    # Compute per-variable cut coefficients from fixing-constraint duals.
    # The fixing-constraint dual mu[j] = -sum_i(pi[i,j]) where pi are
    # the linking-constraint duals. The Benders coefficient is c_j = -mu[j].
    coeffs = {}
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()

        per_elem = {}
        for label, dual_val in fix_duals.items():
            prefix = f'_benders_fix_{var_name}_'
            if label == f'_benders_fix_{var_name}':
                per_elem[0] = float(dual_val)
            elif label.startswith(prefix):
                try:
                    index_text = label[len(prefix):]
                    if index_text.isdigit():
                        idx = int(index_text)
                    else:
                        try:
                            idx = ast.literal_eval(index_text)
                        except (ValueError, SyntaxError):
                            idx = index_text
                    # The fixing-row dual is the recourse slope for the
                    # corresponding master variable in the minimization
                    # value function.
                    per_elem[idx] = float(dual_val)
                except ValueError:
                    pass

        if per_elem:
            if len(y_arr) > 1:
                coeffs[var_name] = per_elem
            else:
                coeffs[var_name] = per_elem.get(0, 0.0)

    # Compute rhs = Q(x_bar) - sum_j c_j * x_bar_j.
    rhs_adjust = 0.0
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()
        c = coeffs.get(var_name, 0.0)
        if isinstance(c, dict):
            for idx, mu in c.items():
                if isinstance(idx, int) and idx < len(y_arr):
                    # c_j * x_bar_j = (-mu[j]) * x_bar_j
                    rhs_adjust += mu * float(y_arr[idx])
        else:
            if isinstance(c, dict):
                for idx, coefficient in c.items():
                    if isinstance(idx, int) and idx < len(y_arr):
                        rhs_adjust += coefficient * float(y_arr[idx])
            else:
                rhs_adjust += c * float(np.sum(y_arr))

    return {
        'rhs': sub_obj - rhs_adjust,
        'coeffs': coeffs,
        'type': 'optimality',
        'sub_obj': sub_obj,
    }


def _generate_feasibility_cut(sub_result, master_vars, complicating_vars):
    """Generate a Benders feasibility cut.

    When the subproblem is infeasible, we need a cut that eliminates the
    current fixing x̄ from the master's feasible region.

    Path 1 (fixing-constraint duals from artificial variables):
        The feasibility cut is:  Σ λ_i (x_i - x̄_i) ≥ 0
        which is:               Σ λ_i x_i  ≥  Σ λ_i x̄_i
        where λ_i are the reduced costs of the fixed variables from
        the always-feasible subproblem (Section 3.3.4).

    Path 2 (no good cut for binary/integer variables):
        Eliminates the current integer fixing.
    """
    fixing_vars = sub_result.get('fixing_vars', [])
    fixing_vals = sub_result.get('fixing_vals', {})
    fix_duals = sub_result.get('fix_duals', {})
    sub_duals = sub_result.get('sub_duals', {})

    all_duals = {**sub_duals, **fix_duals}

    if all_duals:
        coeffs = {}
        rhs = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()

            per_elem = {}
            for label, dual_val in all_duals.items():
                if abs(dual_val) < 1e-12:
                    continue
                prefix = f'_benders_fix_{var_name}_'
                if label == f'_benders_fix_{var_name}':
                    per_elem[0] = float(dual_val)
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        if index_text.isdigit():
                            idx = int(index_text)
                        else:
                            idx = ast.literal_eval(index_text)
                        per_elem[idx] = float(dual_val)
                    except (ValueError, SyntaxError):
                        pass

            if per_elem:
                if len(y_arr) > 1:
                    coeffs[var_name] = per_elem
                    for idx, lam in per_elem.items():
                        if isinstance(idx, int) and idx < len(y_arr):
                            rhs += lam * float(y_arr[idx])
                else:
                    lam = per_elem.get(0, 0.0)
                    coeffs[var_name] = lam
                    rhs += lam * float(np.sum(y_arr))

        if coeffs:
            return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility'}

    coeffs = {}
    rhs = 1.0
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        if isinstance(y_star, dict):
            y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
        else:
            y_arr = np.asarray(y_star, dtype=float).flatten()
        var_coeffs = {}
        if isinstance(y_star, dict):
            indices_iter = y_star.keys()
        elif isinstance(y_star, np.ndarray):
            indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
        else:
            indices_iter = [None]
        for index, value in zip(indices_iter, y_arr):
            v = float(value)
            if v < 0.5:
                var_coeffs[index] = 1.0
            else:
                var_coeffs[index] = -1.0
                rhs -= 1.0
        if var_coeffs:
            coeffs[var_name] = var_coeffs

    if not coeffs:
        for v in complicating_vars:
            coeffs[v] = 1.0

    return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility'}


# ------------------------------------------------------------------
# Advanced cut generation and acceleration techniques
# ------------------------------------------------------------------

def _add_to_cut_pool(cut, cut_pool, max_size):
    """Add cut to pool with size management."""
    cut_pool.append(cut)
    if len(cut_pool) > max_size:
        cut_pool.sort(key=lambda c: abs(c.get('rhs', 0)), reverse=True)
        del cut_pool[max_size:]


def _select_most_violated_cuts(cut_pool, master_vars, max_cuts=5):
    """Select most violated cuts from pool for current master solution."""
    if not cut_pool:
        return []
    violations = []
    for cut in cut_pool:
        violation = _compute_cut_violation(cut, master_vars)
        violations.append((violation, cut))
    violations.sort(key=lambda x: x[0], reverse=True)
    return [cut for _, cut in violations[:max_cuts]]


def _compute_cut_violation(cut, master_vars):
    """Compute violation of a cut for current master solution."""
    if cut.get('type') == 'feasibility':
        lhs = -cut['rhs']
        for var_name, coeff in cut.get('coeffs', {}).items():
            val = master_vars.get(var_name, 0)
            if isinstance(coeff, dict):
                if isinstance(val, dict):
                    for idx, c in coeff.items():
                        if idx in val:
                            lhs += c * val[idx]
                else:
                    c0 = coeff.get(0, 0.0)
                    lhs += c0 * float(np.asarray(val).flatten()[0])
            else:
                if isinstance(val, dict):
                    for v in val.values():
                        lhs += coeff * v
                else:
                    lhs += coeff * float(np.asarray(val).flatten()[0])
        return max(0, -lhs)
    else:
        theta_val = 0.0
        for tname in ('theta', '_benders_theta'):
            if tname in master_vars:
                theta_val = float(np.asarray(master_vars[tname]).flatten()[0])
                break
        if theta_val == 0.0:
            return 0.0
        expr = cut.get('rhs', 0)
        for var_name, coeff in cut.get('coeffs', {}).items():
            val = master_vars.get(var_name, 0)
            if isinstance(coeff, dict):
                if isinstance(val, dict):
                    for idx, c in coeff.items():
                        if idx in val:
                            expr += c * val[idx]
                else:
                    c0 = coeff.get(0, 0.0)
                    expr += c0 * float(np.asarray(val).flatten()[0])
            else:
                if isinstance(val, dict):
                    for v in val.values():
                        expr += coeff * v
                else:
                    expr += coeff * float(np.asarray(val).flatten()[0])
        return max(0, expr - theta_val)


def _generate_multi_cuts(sub_results, master_vars, complicating_vars):
    """Generate multiple cuts from subproblem results."""
    cuts = []
    for sub_result in sub_results:
        if not sub_result['feasible']:
            cut = _generate_feasibility_cut(sub_result, master_vars, complicating_vars)
            if cut is not None:
                cuts.append(cut)
        else:
            standard_cut = _generate_optimality_cut(sub_result, master_vars, complicating_vars)
            if standard_cut is not None:
                cuts.append(standard_cut)
            strengthened_cut = _generate_strengthened_cut(sub_result, master_vars, complicating_vars)
            if strengthened_cut is not None:
                cuts.append(strengthened_cut)
    return cuts


def _generate_strengthened_cut(sub_result, master_vars, complicating_vars):
    """Generate a strengthened optimality cut using Magnanti-Wong technique."""
    fix_duals = sub_result.get('fix_duals', {})
    if not fix_duals:
        sub_obj = sub_result.get('objective_value', 0.0)
        return {
            'rhs': sub_obj,
            'coeffs': {},
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    sub_obj = sub_result.get('objective_value', 0.0)
    fixing_vars = sub_result.get('fixing_vars', [])
    fixing_vals = sub_result.get('fixing_vals', {})
    sub_duals = sub_result.get('sub_duals', {})

    coeffs = {}
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()

        per_elem = {}
        for label, dual_val in fix_duals.items():
            prefix = f'_benders_fix_{var_name}_'
            if label == f'_benders_fix_{var_name}':
                per_elem[0] = float(dual_val)
            elif label.startswith(prefix):
                try:
                    index_text = label[len(prefix):]
                    if index_text.isdigit():
                        idx = int(index_text)
                    else:
                        try:
                            idx = ast.literal_eval(index_text)
                        except (ValueError, SyntaxError):
                            idx = index_text
                    per_elem[idx] = float(dual_val)
                except ValueError:
                    pass

        if per_elem:
            if len(y_arr) > 1:
                coeffs[var_name] = per_elem
            else:
                coeffs[var_name] = per_elem.get(0, 0.0)

    # NOTE: We do NOT clip coefficients or scale the RHS arbitrarily.
    # The standard Benders cut is: θ ≥ Q(x̄) + Σ μ_j (x_j - x̄_j)
    # where μ_j are the fixing-constraint duals. The coefficients must
    # remain as-is to preserve cut validity.

    rhs_adjust = 0.0
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()
        c = coeffs.get(var_name, 0.0)
        if isinstance(c, dict):
            for idx, mu in c.items():
                if isinstance(idx, int) and idx < len(y_arr):
                    rhs_adjust += mu * float(y_arr[idx])
        else:
            rhs_adjust += c * float(np.sum(y_arr))

    strengthened_rhs = sub_obj - rhs_adjust

    return {
        'rhs': strengthened_rhs,
        'coeffs': coeffs,
        'type': 'optimality',
        'sub_obj': sub_obj,
        'strengthened': True
    }


def _generate_pareto_optimal_cut(sub_results, master_vars, complicating_vars):
    """Generate a strengthened Benders optimality cut.

    Uses the most recent subproblem duals to generate a standard Benders cut.
    For a true Magnanti-Wong Pareto-optimal cut, one would solve an auxiliary
    LP to find the point on the dual optimal face that maximizes cut violation.
    Here we use the current duals directly, which is a valid (if not maximal)
    Benders optimality cut.

    Reference: Magnanti and Wong (1981), "Accelerating Benders Decomposition"
    """
    if not sub_results:
        return None

    base_result = sub_results[-1]
    if not base_result['feasible']:
        return None

    fix_duals = base_result.get('fix_duals', {})
    if not fix_duals:
        sub_obj = base_result.get('objective_value', 0.0)
        return {
            'rhs': sub_obj,
            'coeffs': {},
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    sub_obj = base_result.get('objective_value', 0.0)
    fixing_vars = base_result.get('fixing_vars', [])
    fixing_vals = base_result.get('fixing_vals', {})

    coeffs = {}
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()

        per_elem = {}
        for label, dual_val in fix_duals.items():
            prefix = f'_benders_fix_{var_name}_'
            if label == f'_benders_fix_{var_name}':
                per_elem[0] = float(dual_val)
            elif label.startswith(prefix):
                try:
                    index_text = label[len(prefix):]
                    if index_text.isdigit():
                        idx = int(index_text)
                    else:
                        try:
                            idx = ast.literal_eval(index_text)
                        except (ValueError, SyntaxError):
                            idx = index_text
                    per_elem[idx] = float(dual_val)
                except ValueError:
                    pass

        if per_elem:
            if len(y_arr) > 1:
                coeffs[var_name] = per_elem
            else:
                coeffs[var_name] = per_elem.get(0, 0.0)

    rhs_adjust = 0.0
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()
        c = coeffs.get(var_name, 0.0)
        if isinstance(c, dict):
            for idx, mu in c.items():
                if isinstance(idx, int) and idx < len(y_arr):
                    rhs_adjust += mu * float(y_arr[idx])
        else:
            rhs_adjust += c * float(np.sum(y_arr))

    return {
        'rhs': sub_obj - rhs_adjust,
        'coeffs': coeffs,
        'type': 'optimality',
        'sub_obj': sub_obj,
        'pareto': True
    }


def _cleanup_duplicate_cuts(stored_cuts, tolerance=1e-6):
    """Remove duplicate or very similar cuts from storage."""
    if len(stored_cuts) <= 1:
        return stored_cuts
    unique_cuts = []
    seen_cuts = set()
    for cut in stored_cuts:
        signature = _create_cut_signature(cut, tolerance)
        if signature not in seen_cuts:
            unique_cuts.append(cut)
            seen_cuts.add(signature)
    return unique_cuts


def _create_cut_signature(cut, tolerance):
    """Create a signature for cut deduplication."""
    coeffs_tuple = tuple(sorted(
        (var_name, tuple(sorted(coeff.items())) if isinstance(coeff, dict) else coeff)
        for var_name, coeff in cut.get('coeffs', {}).items()
    ))
    rhs_rounded = round(cut.get('rhs', 0.0) / tolerance) * tolerance
    cut_type = cut.get('type', 'unknown')
    return (cut_type, rhs_rounded, coeffs_tuple)


def _apply_cut_selection(stored_cuts, cut_pool, master_vars, max_cuts=10):
    """Select best cuts to add to master problem."""
    if not stored_cuts:
        return stored_cuts
    if cut_pool:
        return _select_most_violated_cuts(cut_pool, master_vars, max_cuts)
    return stored_cuts


def _add_cuts_to_master(master_m, complicating_vars, cuts=None, stored_cuts=None):
    """Add stored Benders cuts to the master model as constraints."""
    cuts_to_add = cuts if cuts is not None else stored_cuts
    theta_var = None
    for name in ('theta', '_benders_theta'):
        for prefix in ('fvar', 'pvar'):
            theta_var = master_m.features['variables'].get((prefix, name))
            if theta_var is not None:
                break
        if theta_var is not None:
            break

    has_theta = theta_var is not None
    if not has_theta:
        try:
            theta_var = master_m.fvar('_benders_theta')
            master_m.con(theta_var >= 0, name='_benders_theta_init')
            has_theta = True
        except Exception:
            pass

    for i, cut in enumerate(cuts_to_add):
        is_feasibility = cut.get('type') == 'feasibility'
        try:
            if is_feasibility:
                # Feasibility cut: sum_{j in I} y_j >= 1
                # where I = {j : y_j* = 0}
                expr = -cut['rhs']
                for var_name, coeff in cut.get('coeffs', {}).items():
                    var_obj = None
                    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                        var_obj = master_m.features['variables'].get(
                            (prefix, var_name))
                        if var_obj is not None:
                            break
                    if var_obj is None:
                        continue
                    if isinstance(var_obj, dict):
                        for idx, v in var_obj.items():
                            try:
                                current_coeff = coeff.get(idx, 0.0) if isinstance(coeff, dict) else coeff
                                expr = expr + current_coeff * v
                            except Exception:
                                pass
                    else:
                        try:
                            expr = expr + coeff * var_obj
                        except Exception:
                            pass
                master_m.con(expr >= 0, name=f'_benders_feas_cut_{i}')
            elif has_theta:
                # Optimality cut for minimization:
                # theta >= rhs + sum_j c_j * y_j
                # → theta - sum_j c_j * y_j - rhs >= 0
                expr = theta_var - cut['rhs']
                for var_name, coeff in cut.get('coeffs', {}).items():
                    var_obj = None
                    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                        var_obj = master_m.features['variables'].get(
                            (prefix, var_name))
                        if var_obj is not None:
                            break
                    if var_obj is None:
                        continue
                    if isinstance(coeff, dict):
                        if isinstance(var_obj, dict):
                            for idx, v in var_obj.items():
                                c = coeff.get(idx, 0.0)
                                if abs(c) > 1e-12:
                                    try:
                                        expr = expr - c * v
                                    except Exception:
                                        pass
                        else:
                            c0 = coeff.get(0, 0.0)
                            if abs(c0) > 1e-12:
                                try:
                                    expr = expr - c0 * var_obj
                                except Exception:
                                    pass
                    elif isinstance(var_obj, dict):
                        for idx, v in var_obj.items():
                            try:
                                expr = expr - coeff * v
                            except Exception:
                                pass
                    else:
                        try:
                            expr = expr - coeff * var_obj
                        except Exception:
                            pass
                master_m.con(expr >= 0, name=f'_benders_opt_cut_{i}')
            else:
                # No theta: theta is implicitly 0, so rhs + sum c_j y_j <= 0
                expr = -cut['rhs']
                for var_name, coeff in cut.get('coeffs', {}).items():
                    var_obj = None
                    for prefix in ('fvar', 'pvar', 'ivar', 'bvar'):
                        var_obj = master_m.features['variables'].get(
                            (prefix, var_name))
                        if var_obj is not None:
                            break
                    if var_obj is None:
                        continue
                    if isinstance(var_obj, dict):
                        for idx, v in var_obj.items():
                            try:
                                expr = expr + coeff * v
                            except Exception:
                                pass
                    else:
                        try:
                            expr = expr + coeff * var_obj
                        except Exception:
                            pass
                master_m.con(expr >= 0, name=f'_benders_cut_{i}')
        except Exception:
            pass


def _generate_combinatorial_optimality_cut(sub_result, master_vars,
                                            complicating_vars, theta_lb=0.0):
    """Generate a combinatorial Benders optimality cut.

    For integer-only subproblems where duals are unavailable, generates:
        theta >= z* - M * nogood(x_bar)
    where M = z* - theta_lb is a valid big-M.

    This follows the formulation from BendersLib's CombinatorialOC.
    """
    sub_obj = sub_result.get('objective_value', 0.0)
    if sub_obj is None:
        return None
    sub_obj = sub_result.get('objective_value', 0.0)
    if sub_obj is None:
        return None

    theta_val = 0.0
    for var_name in complicating_vars:
        val = master_vars.get(var_name, 0)
        if isinstance(val, dict):
            for v in val.values():
                theta_val += float(v)
        else:
            theta_val += float(np.asarray(val).flatten()[0])
    # Use a simple estimate: theta_lb is the lower bound from params
    big_m = sub_obj - theta_lb
    if big_m <= 0:
        big_m = max(abs(sub_obj), 1.0)

    # No-good component: at least one binary/integer variable must change
    coeffs = {}
    rhs_adj = 0.0
    for var_name in complicating_vars:
        y_star = master_vars.get(var_name, 0)
        if isinstance(y_star, dict):
            y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
        else:
            y_arr = np.asarray(y_star, dtype=float).flatten()
        var_coeffs = {}
        if isinstance(y_star, dict):
            indices_iter = y_star.keys()
        elif isinstance(y_star, np.ndarray):
            indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
        else:
            indices_iter = [None]
        for index, value in zip(indices_iter, y_arr):
            v = float(value)
            if v < 0.5:
                var_coeffs[index] = 1.0 / big_m
            else:
                var_coeffs[index] = -1.0 / big_m
                rhs_adj -= 1.0
        if var_coeffs:
            coeffs[var_name] = var_coeffs

    rhs = sub_obj / big_m + rhs_adj
    return {
        'rhs': rhs,
        'coeffs': coeffs,
        'type': 'optimality',
        'sub_obj': sub_obj,
        'sense': '>=',
    }


def _generate_nogood_cut(master_vars, complicating_vars):
    """Generate a no-good feasibility cut excluding the current solution.

    Equivalent to BendersLib's NoGoodFC:
        Σ_{i∈I₁} x_i - Σ_{i∈I₀} x_i ≤ |I₁| - 1
    """
    coeffs = {}
    rhs = 1.0
    for var_name in complicating_vars:
        y_star = master_vars.get(var_name, 0)
        if y_star is None:
            continue
        if isinstance(y_star, dict):
            y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
        else:
            y_arr = np.asarray(y_star, dtype=float).flatten()
        var_coeffs = {}
        if isinstance(y_star, dict):
            indices_iter = y_star.keys()
        elif isinstance(y_star, np.ndarray):
            indices_iter = np.ndindex(y_star.shape) if y_star.ndim > 0 else [()]
        else:
            indices_iter = [None]
        for index, value in zip(indices_iter, y_arr):
            v = float(value)
            if v < 0.5:
                var_coeffs[index] = 1.0
            else:
                var_coeffs[index] = -1.0
                rhs -= 1.0
        if var_coeffs:
            coeffs[var_name] = var_coeffs

    if not coeffs:
        for v in complicating_vars:
            coeffs[v] = 1.0
    return {'rhs': rhs, 'coeffs': coeffs, 'type': 'feasibility', 'sense': '>='}


def _generate_generalized_optimality_cut(sub_result, master_vars,
                                        complicating_vars):
    """Generate a generalized Benders optimality cut using Lagrange multipliers.

    Form: theta >= f(x_bar) + lambda^T * (b - A*x)
    where lambda are the Lagrange multipliers from the subproblem.

    Falls back to numerical coefficients when duals are unavailable
    (e.g., binary first-stage variables with HiGHS).
    """
    fix_duals = sub_result.get('fix_duals', {})
    sub_obj = sub_result.get('objective_value', 0.0)

    # Check if we have useful duals
    has_useful_duals = (
        fix_duals and
        not all(abs(float(v)) <= 1e-12 for v in fix_duals.values()))

    # Fall back to numerical probing coefficients when duals are empty
    numerical_coeffs = sub_result.get('numerical_coeffs', {})
    if not has_useful_duals and numerical_coeffs:
        coeffs = {}
        rhs = sub_obj
        for key, coeff in numerical_coeffs.items():
            if isinstance(key, tuple):
                var_name, idx = key
                y_star = master_vars.get(var_name, {})
                if isinstance(y_star, dict):
                    x_val = float(y_star.get(idx, 0))
                else:
                    flat = np.asarray(y_star).flatten()
                    x_val = float(flat[idx]) if idx < len(flat) else 0.0
                if var_name not in coeffs:
                    coeffs[var_name] = {}
                coeffs[var_name][idx] = coeff
            else:
                var_name = key
                x_val = float(np.asarray(
                    master_vars.get(var_name, 0)).reshape(-1)[0])
                coeffs[var_name] = coeff
            rhs -= coeff * x_val
        return {
            'rhs': rhs,
            'coeffs': coeffs,
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    if not fix_duals:
        sub_obj = sub_result.get('objective_value', 0.0)
        return {
            'rhs': sub_obj,
            'coeffs': {},
            'type': 'optimality',
            'sub_obj': sub_obj,
        }

    sub_obj = sub_result.get('objective_value', 0.0)
    fixing_vars = sub_result.get('fixing_vars', [])
    fixing_vals = sub_result.get('fixing_vals', {})

    coeffs = {}
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()

        per_elem = {}
        for label, dual_val in fix_duals.items():
            prefix = f'_benders_fix_{var_name}_'
            if label == f'_benders_fix_{var_name}':
                per_elem[0] = float(dual_val)
            elif label.startswith(prefix):
                try:
                    index_text = label[len(prefix):]
                    if index_text.isdigit():
                        idx = int(index_text)
                    else:
                        try:
                            idx = ast.literal_eval(index_text)
                        except (ValueError, SyntaxError):
                            idx = index_text
                    per_elem[idx] = float(dual_val)
                except ValueError:
                    pass

        if per_elem:
            if len(y_arr) > 1:
                coeffs[var_name] = per_elem
            else:
                coeffs[var_name] = per_elem.get(0, 0.0)

    # Generalized cut (Geoffrion 1972):
    #   theta >= f(x_bar) - lambda^T A (x - x_bar)
    # Rewritten: theta + lambda^T A x >= f(x_bar) + lambda^T A x_bar
    # We use the form: theta >= sub_obj - sum(c_j * (x_j - x_bar_j))
    # which equals: theta - sum(c_j * x_j) >= sub_obj - sum(c_j * x_bar_j)
    # so rhs = sub_obj - sum(c_j * x_bar_j)
    rhs_adjust = 0.0
    for var_name in fixing_vars:
        y_star = fixing_vals.get(var_name, 0)
        y_arr = np.asarray(
            list(y_star.values()) if isinstance(y_star, dict) else y_star,
            dtype=float).flatten()
        c = coeffs.get(var_name, 0.0)
        if isinstance(c, dict):
            for idx, mu in c.items():
                if isinstance(idx, int) and idx < len(y_arr):
                    rhs_adjust += mu * float(y_arr[idx])
        else:
            rhs_adjust += c * float(np.sum(y_arr))

    return {
        'rhs': sub_obj - rhs_adjust,
        'coeffs': coeffs,
        'type': 'optimality',
        'sub_obj': sub_obj,
        'generalized': True,
        'sense': '>=',
    }


def _logic_cuts(master_vars, subproblem_result, logic_cut_callback):
    if logic_cut_callback is None:
        return []
    returned = logic_cut_callback(master_vars, subproblem_result)
    if returned is None:
        return []
    cuts = returned if isinstance(returned, (list, tuple)) else [returned]
    from .enums import BendersError
    for cut in cuts:
        if not isinstance(cut, dict) or 'rhs' not in cut or 'coeffs' not in cut:
            raise BendersError(
                "Logic cut callbacks must return cut dictionaries with rhs and coeffs")
        cut.setdefault('type', 'feasibility')
    return list(cuts)


def _generate_cut_for_method(sub_result, x_bar, complicating_vars, active_method):
    """Dispatch to the appropriate cut generator based on the active method."""
    if active_method == 'generalized':
        return _generate_generalized_optimality_cut(
            sub_result, x_bar, complicating_vars)
    elif active_method == 'combinatorial':
        return _generate_combinatorial_optimality_cut(
            sub_result, x_bar, complicating_vars)
    elif active_method == 'integer_l_shaped':
        return _generate_combinatorial_optimality_cut(
            sub_result, x_bar, complicating_vars)
    elif active_method == 'generalized_l_shaped':
        return _generate_generalized_optimality_cut(
            sub_result, x_bar, complicating_vars)
    else:
        return _generate_optimality_cut(
            sub_result, x_bar, complicating_vars)


def _generate_l_shaped_aggregated_cut(sub_results, master_vars, complicating_vars):
    """Generate a single aggregated L-shaped cut across all scenarios.

    Form: theta >= sum_omega p_omega * pi_omega^T (b_omega - T_omega * x)
    """
    total_rhs = 0.0
    total_coeffs = {}

    for sub_result in sub_results:
        if not sub_result.get('feasible', False):
            continue
        prob = sub_result.get('probability', 1.0)
        fix_duals = sub_result.get('fix_duals', {})
        sub_obj = sub_result.get('objective_value', 0.0)
        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        # Compute per-scenario coefficients
        scenario_rhs = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                mu = 0.0
                idx = 0
                if label == f'_benders_fix_{var_name}':
                    mu = float(dual_val)
                    idx = 0
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        idx = int(index_text) if index_text.isdigit() else 0
                        mu = float(dual_val)
                    except ValueError:
                        continue
                if isinstance(idx, int) and idx < len(y_arr):
                    total_rhs += prob * mu * float(y_arr[idx])

        total_rhs += prob * sub_obj

        # Aggregate coefficients
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            y_arr = np.asarray(
                list(y_star.values()) if isinstance(y_star, dict) else y_star,
                dtype=float).flatten()
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                mu = 0.0
                idx = 0
                if label == f'_benders_fix_{var_name}':
                    mu = float(dual_val)
                    idx = 0
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        idx = int(index_text) if index_text.isdigit() else 0
                        mu = float(dual_val)
                    except ValueError:
                        continue
                if var_name not in total_coeffs:
                    total_coeffs[var_name] = {}
                if isinstance(idx, int):
                    total_coeffs[var_name][idx] = total_coeffs[var_name].get(idx, 0.0) + prob * mu

    if not total_coeffs:
        return None
    return {
        'rhs': total_rhs,
        'coeffs': total_coeffs,
        'type': 'optimality',
        'sub_obj': total_rhs,
        'sense': '>=',
    }


def _generate_integer_l_shaped_cut(sub_results, master_vars, complicating_vars, theta_lb=0.0):
    """Generate an aggregated integer L-shaped cut (Laporte & Louveaux 1993).

    Uses combinatorial (no-good) cuts for stochastic problems with
    integer complicating variables. Each scenario produces a
    CombinatorialOC cut; these are aggregated with probability weights.

    Form: theta >= sum_omega p_omega * (z_omega - M_omega * nogood(x, x_bar_omega))
    """
    total_rhs = 0.0
    total_coeffs = {}

    for sub_result in sub_results:
        if not sub_result.get('feasible', False):
            continue
        prob = sub_result.get('probability', 1.0)
        sub_obj = sub_result.get('objective_value', 0.0)
        if sub_obj is None:
            continue

        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        big_m = sub_obj - theta_lb
        if big_m <= 0:
            big_m = max(abs(sub_obj), 1.0)

        scenario_rhs_adj = 0.0
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()
            var_indices = y_star.keys() if isinstance(y_star, dict) else (
                np.ndindex(y_star.shape) if isinstance(y_star, np.ndarray) and y_star.ndim > 0 else [None])
            for index, value in zip(var_indices, y_arr):
                v = float(value)
                if var_name not in total_coeffs:
                    total_coeffs[var_name] = {}
                if v < 0.5:
                    total_coeffs[var_name][index] = total_coeffs[var_name].get(index, 0.0) + prob * (1.0 / big_m)
                else:
                    total_coeffs[var_name][index] = total_coeffs[var_name].get(index, 0.0) + prob * (-1.0 / big_m)
                    scenario_rhs_adj -= 1.0

        total_rhs += prob * (sub_obj / big_m + scenario_rhs_adj)

    if not total_coeffs:
        return None
    return {
        'rhs': total_rhs,
        'coeffs': total_coeffs,
        'type': 'optimality',
        'sub_obj': total_rhs,
        'sense': '>=',
    }


def _generate_generalized_l_shaped_cut(sub_results, master_vars, complicating_vars):
    """Generate an aggregated generalized L-shaped cut.

    Uses generalized Benders cuts (Lagrange multiplier based) for
    stochastic problems with convex (nonlinear) recourse. Each scenario
    produces a GeneralizedOC cut; these are aggregated with probability weights.

    Form: theta >= sum_omega p_omega * (f_omega(x_bar) + grad_f_omega(x_bar)^T (x - x_bar))
    """
    total_rhs = 0.0
    total_coeffs = {}

    for sub_result in sub_results:
        if not sub_result.get('feasible', False):
            continue
        prob = sub_result.get('probability', 1.0)
        fix_duals = sub_result.get('fix_duals', {})
        sub_obj = sub_result.get('objective_value', 0.0)
        if sub_obj is None:
            continue

        fixing_vars = sub_result.get('fixing_vars', [])
        fixing_vals = sub_result.get('fixing_vals', {})

        scenario_rhs = prob * sub_obj
        for var_name in fixing_vars:
            y_star = fixing_vals.get(var_name, 0)
            if isinstance(y_star, dict):
                y_arr = np.asarray(list(y_star.values()), dtype=float).flatten()
            else:
                y_arr = np.asarray(y_star, dtype=float).flatten()
            var_indices = y_star.keys() if isinstance(y_star, dict) else (
                np.ndindex(y_star.shape) if isinstance(y_star, np.ndarray) and y_star.ndim > 0 else [None])
            for label, dual_val in fix_duals.items():
                prefix = f'_benders_fix_{var_name}_'
                mu = 0.0
                idx = 0
                if label == f'_benders_fix_{var_name}':
                    mu = float(dual_val)
                    idx = 0
                elif label.startswith(prefix):
                    try:
                        index_text = label[len(prefix):]
                        idx = int(index_text) if index_text.isdigit() else 0
                        mu = float(dual_val)
                    except ValueError:
                        continue
                if isinstance(idx, int) and idx < len(y_arr):
                    # Generalized: rhs = f(x_bar) + lambda^T A x_bar
                    scenario_rhs += prob * mu * float(y_arr[idx])
                if var_name not in total_coeffs:
                    total_coeffs[var_name] = {}
                if isinstance(idx, int):
                    total_coeffs[var_name][idx] = total_coeffs[var_name].get(idx, 0.0) + prob * mu

        total_rhs += scenario_rhs

    if not total_coeffs:
        return None
    return {
        'rhs': total_rhs,
        'coeffs': total_coeffs,
        'type': 'optimality',
        'sub_obj': total_rhs,
        'sense': '>=',
    }
