# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Solver-neutral Lagrangian relaxation framework.

The user supplies the relaxed solver and multiplier update/subgradient oracle.
This keeps the framework applicable to arbitrary exact models without guessing
which constraints should be dualized.
"""

from dataclasses import dataclass, field
import time

import numpy as np

from ..base.logging import make_logger
from ..base.result import DecompositionResult
from ....helpers.containers import as_var_group


class LagrangianError(Exception):
    """Raised when Lagrangian callbacks return invalid data."""


def _component_count(captured, n_constraints):
    """Number of connected components in the variable-constraint graph.

    Each constraint unions its first term with every other term (a
    star), which yields exactly the same components as the full
    pairwise adjacency built by ``_detect_dw_structure`` but in
    O(total terms) instead of O(sum of terms squared).
    """
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for cidx in range(n_constraints):
        terms = captured.constraints[cidx][0].expression.terms
        first = None
        for key in terms.keys():
            if key not in parent:
                parent[key] = key
            if first is None:
                first = key
            else:
                rf = find(first)
                rk = find(key)
                if rf != rk:
                    parent[rk] = rf
    if not parent:
        return 0
    roots = set()
    for var_key in parent:
        roots.add(find(var_key))
    return len(roots)


def _select_relaxable(captured, n_constraints):
    """Choose which constraints to dualize for the automatic builders.

    Priority order:

    1. Articulation ("linking") rows from the block detector.  Relaxing
       exactly the rows that couple otherwise-separable blocks keeps the
       hard core of the model — the dual starts near the true optimum
       instead of at the trivial box bound — while the subproblem keeps
       its block structure.  When the model is *already* separable the
       linking set is empty: nothing needs dualizing and the relaxation
       is the model itself, which converges with the true bound in one
       iteration.  Separability is checked first with a linear
       union-find pass so already-separable models never pay for full
       block detection.
    2. The legacy heuristic (all-integer / mixed-kind / multi-term
       rows) when block detection cannot split a densely-coupled graph.
       This keeps a valid — if weaker — bound as a fallback.

    The legacy heuristic on its own classified *every* constraint with
    two or more terms as relaxable (635/635 on a coupled MILP), leaving
    a bounds-only subproblem whose Lagrangian dual maxes out near zero.
    """
    try:
        if _component_count(captured, n_constraints) > 1:
            return []  # already separable: nothing to dualize
        from ..column_generation.automatic import _detect_dw_structure
        _, linking, _, _, _ = _detect_dw_structure(
            captured, n_constraints)
        if linking:
            return list(linking)
        return []
    except Exception:
        pass

    relaxable_constraints = []
    linking_constraints = []
    for idx, (constraint, label) in enumerate(captured.constraints):
        var_types = set(key[0] for key in constraint.expression.terms)
        all_integer = all(t in ('ivar', 'bvar') for t in var_types)
        has_integer = any(t in ('ivar', 'bvar') for t in var_types)
        n_indices = len(constraint.expression.terms)
        if all_integer and len(var_types) > 0:
            linking_constraints.append((idx, constraint, label))
        elif has_integer and len(var_types) > 1:
            linking_constraints.append((idx, constraint, label))
        elif n_indices > 1:
            linking_constraints.append((idx, constraint, label))

    if not linking_constraints:
        return list(range(n_constraints))
    return [idx for idx, _, _ in linking_constraints]


@dataclass
class LagrangianResult(DecompositionResult):
    """Result from Lagrangian relaxation.

    Inherits ``status``, ``objective``, ``variables``, ``iterations``,
    ``runtime_total``, and ``history`` from ``DecompositionResult``.
    """
    solution: object = None
    multipliers: object = None
    lower_bound: float = float('-inf')
    upper_bound: float = float('inf')


class LagrangianRelaxation:
    """Manual and callback-driven Lagrangian relaxation.

    ``relaxed_solver(multipliers)`` returns ``objective`` and optionally
    ``solution``, ``bound``, and ``subgradient``. ``multiplier_update`` receives
    ``(multipliers, subgradient, iteration, relaxed_result)`` and returns the
    next multipliers. ``upper_bound`` may optionally map a solution to a primal
    feasible upper bound.

    Supports multiple step-size strategies:
    - 'subgradient': Classic subgradient method with fixed step size
    - 'polyak': Polyak's step size using knowledge of optimal value
    - 'adaptive': Adaptive step size based on progress
    - 'volume': Volume algorithm (combines subgradient with averaging)
    """

    def __init__(self, relaxed_solver=None, multiplier_update=None,
                 initial_multipliers=None, upper_bound=None, direction='min',
                 multiplier_senses=None):
        """Initialise Lagrangian relaxation.

        Parameters
        ----------
        relaxed_solver : callable, optional
            ``(multipliers) -> dict`` with ``objective`` and optionally
            ``solution``, ``bound``, ``subgradient``.
        multiplier_update : callable, optional
            ``(multipliers, subgradient, iteration, relaxed) -> multipliers``.
        initial_multipliers : array-like, optional
            Starting multiplier values.
        upper_bound : callable, optional
            ``(solution) -> float`` mapping to a primal upper bound.
        direction : str
            ``'min'`` or ``'max'``.
        multiplier_senses : dict, optional
            ``{multiplier_key: '<=' | '>=' | '=='}`` original constraint
            senses.  The internal step strategies (polyak/adaptive/
            volume) flip the subgradient direction for ``>=`` and ``==``
            constraints, matching ``multiplier_update``'s convention.
            Without it the internal steps treat every constraint as
            ``<=``, pushing ``>=`` multipliers the wrong way.
        """
        self.relaxed_solver = relaxed_solver
        self.multiplier_update = multiplier_update
        self.multipliers = initial_multipliers
        self.upper_bound = upper_bound
        self.direction = direction
        self.multiplier_senses = multiplier_senses

    def solve(self, relaxed_solver=None, multiplier_update=None,
              max_iterations=100, tolerance=1e-6, show_log=False,
              step_strategy='adaptive', polyak_estimate=None,
              volume_weight=0.5, multiplier_senses=None):
        """Run Lagrangian relaxation with advanced step-size strategies.

        Parameters
        ----------
        step_strategy : str
            'subgradient', 'polyak', 'adaptive', or 'volume'
        polyak_estimate : float, optional
            Estimated optimal objective value (required for 'polyak')
        volume_weight : float
            Weight for volume algorithm averaging (0-1)
        multiplier_senses : dict, optional
            ``{key: '<=' | '>=' | '=='}`` — overrides the instance-level
            senses for the internal step updates.
        """
        relaxed_solver = relaxed_solver or self.relaxed_solver
        multiplier_update = multiplier_update or self.multiplier_update
        if not callable(relaxed_solver) or not callable(multiplier_update):
            raise LagrangianError(
                "relaxed_solver and multiplier_update callbacks are required")
        if multiplier_senses is None:
            multiplier_senses = getattr(self, 'multiplier_senses', None)

        log = make_logger('Lagrangian', show_log)

        _t_start = time.perf_counter()
        lower_bound = float('-inf')
        upper_bound = float('inf')
        best_solution = None
        history = []
        multipliers = self.multipliers
        is_min = self.direction == 'min'

        best_lower = float('-inf')
        best_upper = float('inf')
        stagnation_count = 0
        prev_bound = None
        adaptive_step = None
        volume_avg_solution = None
        volume_avg_count = 0
        objective_values = []
        prev_multipliers = None
        step_scale = 1.0

        for iteration in range(1, max_iterations + 1):
            relaxed = relaxed_solver(multipliers)
            if not isinstance(relaxed, dict) or 'objective' not in relaxed:
                raise LagrangianError(
                    "relaxed_solver must return a dict containing objective")
            try:
                objective = float(relaxed['objective'])
            except (TypeError, ValueError):
                objective = float('nan')

            if not np.isfinite(objective):
                # The relaxed subproblem failed — typically the
                # multiplier overshoot flipped objective signs and the
                # (partially) relaxed model turned unbounded/infeasible.
                # Previously this non-finite value was folded straight
                # into lower_bound (max(-inf, inf) = inf), poisoning the
                # bound, the gap (inf-inf = nan) and the final objective
                # from this iteration onward.  Keep the last finite
                # bound, revert to the last productive multipliers and
                # back the step off; the history records the failure
                # honestly without contaminating the bounds.
                step_scale *= 0.5
                if prev_multipliers is not None:
                    multipliers = dict(prev_multipliers)
                history.append({
                    'iteration': iteration,
                    'objective': None,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'gap': upper_bound - lower_bound,
                    'step_size': None,
                    'failed': True,
                })
                if show_log:
                    log(f"iteration={iteration} relaxed solve failed "
                        f"(step_scale={step_scale:.6g}); "
                        f"keeping LB={lower_bound}")
                continue

            # Last known-good multipliers: on a future failed solve we
            # revert to these instead of iterating on a poisoned state.
            prev_multipliers = dict(multipliers)
            if step_scale < 1.0:
                step_scale = min(1.0, step_scale * 1.5)
            objective_values.append(objective)

            if is_min:
                try:
                    bound_val = float(relaxed.get('bound', objective))
                except (TypeError, ValueError):
                    bound_val = float('nan')
                if np.isfinite(bound_val):
                    lower_bound = max(lower_bound, bound_val)
            else:
                try:
                    bound_val = float(relaxed.get('bound', objective))
                except (TypeError, ValueError):
                    bound_val = float('nan')
                if np.isfinite(bound_val):
                    upper_bound = min(upper_bound, bound_val)

            solution = relaxed.get('solution')
            primal_bound = None
            if self.upper_bound is not None and solution is not None:
                try:
                    candidate = float(self.upper_bound(solution))
                    primal_bound = candidate
                    if is_min:
                        if candidate < upper_bound:
                            upper_bound = candidate
                            best_solution = solution
                    else:
                        if candidate > lower_bound:
                            lower_bound = candidate
                            best_solution = solution
                except (TypeError, ValueError):
                    pass

            if volume_weight > 0 and solution is not None:
                if volume_avg_solution is None:
                    volume_avg_solution = {}
                    for k, v in solution.items():
                        if isinstance(v, (int, float)):
                            volume_avg_solution[k] = v
                        elif isinstance(v, np.ndarray):
                            volume_avg_solution[k] = v.copy()
                        elif hasattr(v, '__len__') and not isinstance(v, str):
                            try:
                                volume_avg_solution[k] = np.array(v, dtype=float)
                            except (TypeError, ValueError):
                                volume_avg_solution[k] = v
                        else:
                            try:
                                volume_avg_solution[k] = float(v)
                            except (TypeError, ValueError):
                                volume_avg_solution[k] = v
                    volume_avg_count = 1
                else:
                    volume_avg_count += 1
                    alpha = volume_weight / volume_avg_count
                    for k, v in solution.items():
                        if k not in volume_avg_solution:
                            continue
                        old = volume_avg_solution[k]
                        if isinstance(v, (int, float)) and isinstance(old, (int, float)):
                            volume_avg_solution[k] = old + alpha * (v - old)
                        elif isinstance(v, np.ndarray) and isinstance(old, np.ndarray) and v.shape == old.shape:
                            volume_avg_solution[k] = old + alpha * (v - old)

            subgradient = relaxed.get('subgradient')
            if subgradient is None:
                raise LagrangianError(
                    "relaxed_solver must return subgradient for multiplier updates")

            # Sense-adjusted, boundary-clamped effective step directions.
            # The step norm is taken over these — NOT over the raw
            # subgradients — so huge satisfied-constraint values sitting
            # at lambda=0 (frozen: they contribute no motion) cannot
            # throttle every real update down to ~0.  For a model whose
            # constraints are all relaxed, the raw norm was ~1.9e7 while
            # the effective direction norm was 14, freezing the dual
            # ascent completely.
            direction_sign = 1.0 if is_min else -1.0
            step_directions = {}
            for key, lam in multipliers.items():
                g = subgradient.get(key, 0.0)
                # L uses +lambda*g for '<=' and -lambda*g for '>=' /
                # '==', so the ascent direction flips for the latter.
                _sign = direction_sign
                if multiplier_senses is not None:
                    if multiplier_senses.get(key, '<=') != '<=':
                        _sign = -direction_sign
                d = _sign * g
                if lam <= 0.0 and d < 0.0:
                    continue  # clamped at the boundary: no motion
                step_directions[key] = d
            norm_sq = sum(d * d for d in step_directions.values())

            if step_strategy == 'polyak' and polyak_estimate is not None:
                if norm_sq > 1e-12:
                    if is_min:
                        gap = polyak_estimate - objective
                    else:
                        gap = objective - polyak_estimate
                    # The floor is norm-normalized too: a fixed 0.01
                    # step against a 1e6 subgradient moves lambda by
                    # 1e4 in one shot and flips objective signs.
                    adaptive_step = max(
                        0.01 / (norm_sq ** 0.5 + 1e-12),
                        gap / (norm_sq + 1e-12))
                else:
                    adaptive_step = 1.0
            elif step_strategy == 'adaptive':
                if norm_sq > 1e-12:
                    current_bound = lower_bound if is_min else upper_bound
                    if prev_bound is not None:
                        bound_change = abs(current_bound - prev_bound)
                        if bound_change < 1e-8:
                            stagnation_count += 1
                        else:
                            stagnation_count = 0

                    # All fallback steps are divided by the direction
                    # norm — matching multiplier_update's step/norm
                    # convention — so lambda moves by O(step * d_i/||d||)
                    # instead of O(step * d_i).  Unnormalized, a 1e6
                    # subgradient made lambda ~1e6 in one iteration,
                    # turned the penalized objective unbounded and
                    # killed the dual bound with inf.
                    _gn = norm_sq ** 0.5 + 1e-12
                    if stagnation_count >= 5:
                        adaptive_step = max(0.1, 1.0 / (iteration ** 0.5)) / _gn
                    elif stagnation_count >= 3:
                        adaptive_step = max(0.5, 2.0 / (iteration ** 0.5)) / _gn
                    else:
                        feasible_gap = upper_bound - lower_bound
                        if feasible_gap > 0 and feasible_gap < float('inf'):
                            adaptive_step = feasible_gap / (norm_sq + 1e-12)
                        else:
                            adaptive_step = 1.0 / ((iteration ** 0.5) * _gn)
                    prev_bound = current_bound
                else:
                    adaptive_step = 1.0
            elif step_strategy == 'volume':
                if norm_sq > 1e-12:
                    feasible_gap = upper_bound - lower_bound
                    if feasible_gap > 0 and feasible_gap < float('inf'):
                        adaptive_step = feasible_gap / (norm_sq + 1e-12)
                    else:
                        adaptive_step = 1.0 / (
                            (iteration ** 0.5) * (norm_sq ** 0.5 + 1e-12))
                else:
                    adaptive_step = 1.0
            else:
                adaptive_step = None

            if adaptive_step is not None:
                if step_scale != 1.0:
                    adaptive_step = adaptive_step * step_scale
                new_multipliers = dict(multipliers)
                for key, d in step_directions.items():
                    new_multipliers[key] = max(
                        0.0, multipliers.get(key, 0.0) + adaptive_step * d)
                multipliers = new_multipliers
            else:
                multipliers = multiplier_update(
                    multipliers, subgradient, iteration, relaxed)

            gap = upper_bound - lower_bound
            history.append({
                'iteration': iteration,
                'objective': objective,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'gap': gap,
                'step_size': adaptive_step,
            })
            if show_log:
                if adaptive_step:
                    log(
                        f"iteration={iteration} "
                        f"LB={lower_bound:.6f} UB={upper_bound:.6f} "
                        f"gap={gap:.6f} step={adaptive_step:.4f}")
                else:
                    log(
                        f"iteration={iteration} "
                        f"LB={lower_bound:.6f} UB={upper_bound:.6f} "
                        f"gap={gap:.6f}")
            if gap >= 0 and gap <= tolerance:
                best_obj = upper_bound if is_min else lower_bound
                return LagrangianResult(
                    objective=best_obj,
                    solution=best_solution or solution,
                    multipliers=multipliers,
                    iterations=iteration,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    status='optimal',
                    history=history,
                    runtime_total=time.perf_counter() - _t_start,
                )

            best_lower = max(best_lower, lower_bound)
            best_upper = min(best_upper, upper_bound)

        if volume_avg_solution is not None and volume_avg_count > 0:
            best_solution = volume_avg_solution

        best_obj = best_upper if is_min else best_lower
        if best_obj in (float('-inf'), float('inf')):
            best_obj = upper_bound if is_min else lower_bound

        return LagrangianResult(
            objective=best_obj,
            solution=best_solution,
            multipliers=multipliers,
            iterations=max_iterations,
            lower_bound=best_lower,
            upper_bound=best_upper,
            status='max_iterations',
            history=history,
            runtime_total=time.perf_counter() - _t_start,
        )

    @classmethod
    def automatic(cls, relaxed_solver, multiplier_update, **kwargs):
        """Run the callback-driven loop without manual iteration code."""
        upper_bound = kwargs.pop('upper_bound', None)
        direction = kwargs.pop('direction', 'min')
        return cls(relaxed_solver, multiplier_update, upper_bound=upper_bound,
                   direction=direction).solve(**kwargs)

    # ------------------------------------------------------------------
    # Automatic callback factory for boost=True
    # ------------------------------------------------------------------

    @classmethod
    def from_automatic(cls, captured, em, interface, solver, directions,
                       method='exact',
                       time_limit=None, cpu_threads=None,
                       absolute_gap=None, relative_gap=None,
                       step_size=1.0):
        """Build Lagrangian callbacks from an _AutomaticCaptureModel.

        Optimizations: constraint term structures pre-computed, metadata cached.
        Returns (relaxed_solver, multiplier_update, initial_multipliers, relaxable).
        """
        _if = interface if isinstance(interface, str) else (
            interface[0] if isinstance(interface, (list, tuple)) else 'highs')
        _sv = solver if isinstance(solver, str) else (
            solver[0] if isinstance(solver, (list, tuple)) else 'highs')
        _model = cls._import_model()

        relaxable_constraints = _select_relaxable(
            captured, len(captured.constraints))
        relaxed_indices = set(relaxable_constraints)

        n_constraints = len(captured.constraints)
        constraint_senses = [c.sense for c, _ in captured.constraints]
        constraint_labels = [l for _, l in captured.constraints]
        relaxable_senses = {ridx: captured.constraints[ridx][0].sense
                            for ridx in relaxable_constraints}

        obj_expression, obj_direction, obj_label = captured.objectives[0]
        obj_terms = dict(obj_expression.terms)
        obj_constant = obj_expression.constant

        # obj() may carry no explicit direction: fall back to the search
        # directions rather than assuming one
        effective_direction = obj_direction or (
            directions[0] if directions else 'min')

        constraint_term_structs = []
        for cidx in range(n_constraints):
            expr = captured.constraints[cidx][0].expression
            constraint_term_structs.append(
                (expr.constant, list(expr.terms.items())))

        def relaxed_solver(multipliers):
            _m = _model(method=method, name='_lag_relaxed', interface=_if)
            native = cls._build_native_from_captured(_m, captured)

            for cidx in range(n_constraints):
                if cidx in relaxed_indices:
                    continue
                label = constraint_labels[cidx]
                constant, term_items = constraint_term_structs[cidx]
                expression = constant
                for key, coefficient in term_items:
                    expression = expression + coefficient * native[key]
                sense = constraint_senses[cidx]
                if sense == '<=':
                    _m.con(expression <= 0, name=label)
                elif sense == '>=':
                    _m.con(expression >= 0, name=label)
                else:
                    _m.con(expression == 0, name=label)

            translated_obj = obj_constant
            for key, coefficient in obj_terms.items():
                translated_obj = translated_obj + coefficient * native[key]

            for ridx in relaxable_constraints:
                lam = multipliers.get(ridx, 0.0)
                if lam == 0.0:
                    continue
                sign = 1.0 if relaxable_senses[ridx] == '<=' else -1.0
                constant, term_items = constraint_term_structs[ridx]
                constraint_val = constant
                for key, coefficient in term_items:
                    constraint_val = constraint_val + coefficient * native[key]
                translated_obj = translated_obj + sign * lam * constraint_val

            if isinstance(translated_obj, (int, float)):
                dummy = _m.fvar('_lag_dummy', bound=[0, 0])
                _m.obj(translated_obj + 0 * dummy, direction=effective_direction, label=obj_label)
            else:
                _m.obj(translated_obj, direction=effective_direction, label=obj_label)

            _m.sol(
                directions=directions, solver=_sv,
                time_limit=time_limit, cpu_threads=cpu_threads,
                absolute_gap=absolute_gap, relative_gap=relative_gap,
            )

            objective = float(_m.get_obj())

            solved_vals = {}
            for (kind, name), spec in captured.variables.items():
                try:
                    val = _m.get_numpy_var(name)
                except Exception:
                    continue
                if isinstance(val, dict):
                    solved_vals.update({(kind, name, idx): v
                                        for idx, v in val.items()})
                elif isinstance(val, np.ndarray):
                    for idx in spec['values']:
                        try:
                            # get_numpy_var returns arrays shaped like the
                            # captured dimension, so index with the full
                            # captured key (int or tuple), not idx[0].
                            resolved = val[idx]
                            if np.ndim(resolved) != 0:
                                continue
                            solved_vals[(kind, name, idx)] = float(resolved)
                        except (IndexError, TypeError, ValueError):
                            continue
                else:
                    for idx in spec['values']:
                        solved_vals[(kind, name, idx)] = float(val)

            subgradient = {}
            for ridx in relaxable_constraints:
                constant, term_items = constraint_term_structs[ridx]
                violation = constant
                for key, coefficient in term_items:
                    val = solved_vals.get(key, 0.0)
                    violation = violation + coefficient * val
                try:
                    subgradient[ridx] = float(violation)
                except (TypeError, ValueError):
                    subgradient[ridx] = 0.0

            solution = {}
            for (kind, name) in captured.variables:
                try:
                    solution[name] = _m.get_numpy_var(name)
                except Exception:
                    pass

            return {
                'objective': objective,
                'subgradient': subgradient,
                'solution': solution,
            }

        def multiplier_update(multipliers, subgradient, iteration, relaxed_result):
            direction_sign = -1.0 if effective_direction == 'max' else 1.0
            # Norm over effective (sense-adjusted, boundary-clamped)
            # directions: a raw norm is dominated by huge
            # satisfied-constraint values sitting frozen at lambda=0,
            # which throttled every real step down to ~0.
            norm_sq = 0.0
            for ridx in relaxable_constraints:
                lam = multipliers.get(ridx, 0.0)
                g = subgradient.get(ridx, 0.0)
                d = g if relaxable_senses[ridx] == '<=' else -g
                if lam <= 0.0 and d < 0.0:
                    continue
                norm_sq += d * d
            if norm_sq < 1e-12:
                return multipliers
            alpha = step_size / (norm_sq ** 0.5 + 1e-12)
            new_multipliers = {}
            for ridx in relaxable_constraints:
                lam = multipliers.get(ridx, 0.0)
                g = subgradient.get(ridx, 0.0)
                sense = relaxable_senses[ridx]
                if sense == '<=':
                    new_multipliers[ridx] = max(0.0, lam + direction_sign * alpha * g)
                else:
                    new_multipliers[ridx] = max(0.0, lam - direction_sign * alpha * g)
            return new_multipliers

        initial_multipliers = {ridx: 0.0 for ridx in relaxable_constraints}

        return relaxed_solver, multiplier_update, initial_multipliers, relaxable_constraints

    @staticmethod
    def _import_model():
        from feloopy.feloopy import model as _model_cls
        return _model_cls

    @staticmethod
    def _build_native(m, captured):
        import numpy as _np
        native = {}
        for (kind, name), spec in captured.variables.items():
            val = m.get_numpy_var(name)
            for index, symbolic in spec['values'].items():
                if index is None:
                    native[symbolic.key] = val
                elif isinstance(val, dict):
                    native[symbolic.key] = val[index]
                elif isinstance(val, _np.ndarray):
                    idx = index[0] if isinstance(index, tuple) else index
                    native[symbolic.key] = val.flat[idx] if val.ndim > 1 else val[idx]
                else:
                    native[symbolic.key] = val
        return native

    @staticmethod
    def _build_native_from_captured(m, captured):
        native = {}
        for (kind, name), spec in captured.variables.items():
            creator = getattr(m, kind)
            value = creator(name=name, dim=spec['dim'], bound=spec['bound'])
            if isinstance(value, dict):
                values = value
            else:
                # dict / dict-like container (pyomo IndexedVar, COPT
                # tupledict) / scalar — as_var_group normalizes all three
                # with guarded attribute access.
                values = as_var_group(value)
            for index, symbolic in spec['values'].items():
                native[symbolic.key] = values[index]
        return native

    # ------------------------------------------------------------------
    # Incremental Lagrangian for supported interfaces
    # ------------------------------------------------------------------

    @classmethod
    def from_automatic_incremental(cls, captured, em, interface, solver,
                                   directions, method='exact',
                                   time_limit=None, cpu_threads=None,
                                   absolute_gap=None, relative_gap=None,
                                   step_size=1.0):
        """Build incremental Lagrangian callbacks.

        Uses IncrementalModel to build the relaxed problem once and
        update objective coefficients each iteration, avoiding
        per-iteration model rebuilds.

        Returns (relaxed_solver, multiplier_update, initial_multipliers,
                 relaxable).
        """
        from feloopy.classes.incremental import IncrementalModel
        from feloopy.feloopy import model as _model_cls

        _if = interface if isinstance(interface, str) else (
            interface[0] if isinstance(interface, (list, tuple)) else 'highs')
        _sv = solver if isinstance(solver, str) else (
            solver[0] if isinstance(solver, (list, tuple)) else 'highs')

        relaxable_constraints = _select_relaxable(
            captured, len(captured.constraints))
        relaxed_indices = set(relaxable_constraints)

        n_constraints = len(captured.constraints)
        constraint_senses = [c.sense for c, _ in captured.constraints]
        constraint_labels = [l for _, l in captured.constraints]
        relaxable_senses = {ridx: captured.constraints[ridx][0].sense
                            for ridx in relaxable_constraints}

        obj_expression, obj_direction, obj_label = captured.objectives[0]
        obj_terms = dict(obj_expression.terms)
        obj_constant = obj_expression.constant

        # obj() may carry no explicit direction: fall back to the search
        # directions rather than assuming one (treating None as 'max'
        # inverted every relaxed objective)
        effective_direction = obj_direction or (
            directions[0] if directions else 'min')

        # Pre-compute per-constraint coefficient vectors
        var_keys = list(captured.variables.keys())
        key_to_col = {}
        col = 0
        for (kind, name), spec in captured.variables.items():
            for index in spec['values']:
                key_to_col[(kind, name, index)] = col
                col += 1
        n_cols = col

        constraint_coeff_matrix = []
        for cidx in range(n_constraints):
            expr = captured.constraints[cidx][0].expression
            coeffs = {}
            for key, coefficient in expr.terms.items():
                coeffs[key] = coefficient
            constraint_coeff_matrix.append(
                (constraint_senses[cidx], constraint_labels[cidx],
                 expr.constant, coeffs))

        # Build base model once
        _m = _model_cls(method=method, name='_lag_relaxed', interface=_if)
        native = cls._build_native_from_captured(_m, captured)

        # Add non-relaxable constraints
        for cidx in range(n_constraints):
            if cidx in relaxed_indices:
                continue
            sense, label, constant, coeffs = constraint_coeff_matrix[cidx]
            expression = constant
            for key, coefficient in coeffs.items():
                expression = expression + coefficient * native[key]
            if sense == '<=':
                _m.con(expression <= 0, name=label)
            elif sense == '>=':
                _m.con(expression >= 0, name=label)
            else:
                _m.con(expression == 0, name=label)

        # Set initial objective
        translated_obj = obj_constant
        for key, coefficient in obj_terms.items():
            translated_obj = translated_obj + coefficient * native[key]
        if isinstance(translated_obj, (int, float)):
            dummy = _m.fvar('_lag_dummy', bound=[0, 0])
            _m.obj(translated_obj + 0 * dummy, direction=effective_direction,
                   label=obj_label)
        else:
            _m.obj(translated_obj, direction=effective_direction, label=obj_label)

        # Build IncrementalModel
        inc = IncrementalModel(
            _m, directions=directions, obj_index=0,
            solver_name=_sv, solver_options={},
            time_limit=time_limit, thread_count=cpu_threads,
            absolute_gap=absolute_gap, relative_gap=relative_gap,
            log=False)
        inc.build()

        # Cache the base column costs
        m_native = inc.native_model
        if _if == 'highs':
            base_costs = list(m_native.getLp().col_cost_)
        elif _if == 'gurobi':
            base_costs = [v.Obj for v in m_native.getVars()]
        else:
            base_costs = [0.0] * n_cols

        # Invert key->col once: lookups during iterations are then O(1)
        # instead of rescanning key_to_col for every column.
        col_to_key = {col_idx: key for key, col_idx in key_to_col.items()}

        # Map relaxable constraints to (constant, sparse column
        # coefficients).  The sparse form keeps each relaxed solve linear
        # in the matrix nnz — the previous dense n_cols walk per
        # constraint, combined with a key_to_col rescan for every column,
        # made the subgradient O(n_relaxable * n_cols**2) per iteration
        # and hung on medium-sized models.
        relax_coeff_sparse = {}
        for ridx in relaxable_constraints:
            sense, label, constant, coeffs = constraint_coeff_matrix[ridx]
            sign = 1.0 if sense == '<=' else -1.0
            terms = []
            for key, coefficient in coeffs.items():
                col_idx = key_to_col.get(key)
                if col_idx is None:
                    continue
                value = sign * coefficient
                if value != 0.0:
                    terms.append((col_idx, value))
            relax_coeff_sparse[ridx] = (sign * constant, terms)

        can_update_incr = _if in ('highs', 'gurobi', 'cplex')

        def relaxed_solver(multipliers):
            if can_update_incr:
                new_costs = list(base_costs)
                for ridx in relaxable_constraints:
                    lam = multipliers.get(ridx, 0.0)
                    if lam == 0.0:
                        continue
                    _, terms = relax_coeff_sparse[ridx]
                    for j, coeff in terms:
                        new_costs[j] += lam * coeff
                inc.update_objective_coefficients(
                    {j: new_costs[j] for j in range(n_cols)})
                inc.set_objective_direction(
                    'min' if effective_direction == 'min' else 'max')
                inc.solve()

                objective = inc.get_objective_value()
                constant_offset = 0.0
                for ridx in relaxable_constraints:
                    lam = multipliers.get(ridx, 0.0)
                    if lam == 0.0:
                        continue
                    constant, _ = relax_coeff_sparse[ridx]
                    constant_offset += lam * constant
                if constant_offset != 0.0:
                    objective = (objective or 0.0) + constant_offset
            else:
                _m2 = _model_cls(method=method, name='_lag_relaxed',
                                 interface=_if)
                native2 = cls._build_native_from_captured(_m2, captured)
                for cidx in range(n_constraints):
                    if cidx in relaxed_indices:
                        continue
                    sense, label, constant, coeffs = constraint_coeff_matrix[cidx]
                    expression = constant
                    for key, coefficient in coeffs.items():
                        expression = expression + coefficient * native2[key]
                    if sense == '<=':
                        _m2.con(expression <= 0, name=label)
                    elif sense == '>=':
                        _m2.con(expression >= 0, name=label)
                    else:
                        _m2.con(expression == 0, name=label)
                translated_obj = obj_constant
                for key, coefficient in obj_terms.items():
                    translated_obj = translated_obj + coefficient * native2[key]
                for ridx in relaxable_constraints:
                    lam = multipliers.get(ridx, 0.0)
                    if lam == 0.0:
                        continue
                    sign = 1.0 if relaxable_senses[ridx] == '<=' else -1.0
                    constant, coeffs = constraint_coeff_matrix[ridx]
                    constraint_val = constant
                    for key, coefficient in coeffs.items():
                        constraint_val = constraint_val + coefficient * native2[key]
                    translated_obj = translated_obj + sign * lam * constraint_val
                if isinstance(translated_obj, (int, float)):
                    dummy = _m2.fvar('_lag_dummy', bound=[0, 0])
                    _m2.obj(translated_obj + 0 * dummy, direction=effective_direction,
                            label=obj_label)
                else:
                    _m2.obj(translated_obj, direction=effective_direction,
                            label=obj_label)
                _m2.sol(directions=directions, solver=_sv,
                        time_limit=time_limit, cpu_threads=cpu_threads,
                        absolute_gap=absolute_gap, relative_gap=relative_gap)
                objective = float(_m2.get_obj())

            # Compute subgradient: walk only the nonzero coefficients of
            # each relaxed row (inverse col->key map, O(1) per entry)
            subgradient = {}
            all_vals = inc.get_all_variable_values() if can_update_incr else {}
            for ridx in relaxable_constraints:
                constant, terms = relax_coeff_sparse[ridx]
                violation = constant
                for j, coeff in terms:
                    key = col_to_key.get(j)
                    if key is None:
                        continue
                    kind, name, index = key
                    val = all_vals.get(name)
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        violation += coeff * val.get(index, 0.0)
                    elif index is None:
                        violation += coeff * float(val)
                    elif isinstance(val, (list, np.ndarray)):
                        idx = index[0] if isinstance(index, tuple) else index
                        if idx < len(val):
                            violation += coeff * val[idx]
                try:
                    subgradient[ridx] = float(violation)
                except (TypeError, ValueError):
                    subgradient[ridx] = 0.0

            solution = {}
            if can_update_incr:
                for (kind, name), spec in captured.variables.items():
                    if name in all_vals:
                        solution[name] = all_vals[name]
            else:
                for (kind, name) in captured.variables:
                    try:
                        solution[name] = _m2.get_numpy_var(name)
                    except Exception:
                        pass

            return {
                'objective': objective,
                'subgradient': subgradient,
                'solution': solution,
            }

        def multiplier_update(multipliers, subgradient, iteration, relaxed_result):
            direction_sign = -1.0 if effective_direction == 'max' else 1.0
            # Norm over effective (sense-adjusted, boundary-clamped)
            # directions: a raw norm is dominated by huge
            # satisfied-constraint values sitting frozen at lambda=0,
            # which throttled every real step down to ~0.
            norm_sq = 0.0
            for ridx in relaxable_constraints:
                lam = multipliers.get(ridx, 0.0)
                g = subgradient.get(ridx, 0.0)
                d = g if relaxable_senses[ridx] == '<=' else -g
                if lam <= 0.0 and d < 0.0:
                    continue
                norm_sq += d * d
            if norm_sq < 1e-12:
                return multipliers
            alpha = step_size / (norm_sq ** 0.5 + 1e-12)
            new_multipliers = {}
            for ridx in relaxable_constraints:
                lam = multipliers.get(ridx, 0.0)
                g = subgradient.get(ridx, 0.0)
                sense = relaxable_senses[ridx]
                if sense == '<=':
                    new_multipliers[ridx] = max(0.0, lam + direction_sign * alpha * g)
                else:
                    new_multipliers[ridx] = max(0.0, lam - direction_sign * alpha * g)
            return new_multipliers

        initial_multipliers = {ridx: 0.0 for ridx in relaxable_constraints}

        return relaxed_solver, multiplier_update, initial_multipliers, relaxable_constraints
