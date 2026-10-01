"""Core column generation engine.

Provides the solver-neutral iteration loop with dual stabilization,
anti-cycling strategies, column pool management, and incremental mode.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time

from ..base.logging import make_logger
from ..base.result import DecompositionResult


class ColumnGenerationError(Exception):
    """Raised when column-generation callbacks return invalid data."""


@dataclass
class ColumnGenerationResult(DecompositionResult):
    """Result from column generation.

    Inherits ``status``, ``objective``, ``variables``, ``iterations``,
    ``runtime_total``, and ``history`` from ``DecompositionResult``.
    """
    columns: list = field(default_factory=list)
    n_columns_added: int = 0
    lower_bound: float = float('-inf')
    upper_bound: float = float('inf')


class ColumnGeneration:
    """Manual and callback-driven column generation.

    ``master_solver(columns)`` must return a mapping containing ``objective``,
    ``duals``, and optionally ``solution``. ``pricing_oracle(duals)`` returns a
    column object, a list of columns, or ``None`` when no improving column
    exists. A column may be any object understood by ``master_solver``.

    Enhanced with dual stabilization, anti-cycling strategies, and
    incremental column addition.
    """

    def __init__(self, master_solver=None, pricing_oracle=None):
        """Initialise column generation engine.

        Parameters
        ----------
        master_solver : callable, optional
            ``(columns) -> dict`` with keys ``objective``, ``duals``.
        pricing_oracle : callable, optional
            ``(duals) -> column | [columns] | None``.
        """
        self.master_solver = master_solver
        self.pricing_oracle = pricing_oracle
        self.columns = []
        self._column_pool = []
        self._pool_max_size = 200
        self._stabilized_duals = None
        self._dual_averaging_weight = 0.3

    def add_column(self, column):
        self.columns.append(column)
        return self

    def add_columns(self, columns):
        self.columns.extend(columns)
        return self

    def _stabilize_duals(self, duals, iteration):
        if self._stabilized_duals is None:
            self._stabilized_duals = dict(duals)
            return self._stabilized_duals

        if iteration <= 3:
            weight = 0.5
        elif iteration <= 10:
            weight = 0.3
        else:
            weight = 0.2

        stabilized = {}
        for key, value in duals.items():
            if key in self._stabilized_duals:
                old_val = self._stabilized_duals[key]
                if isinstance(value, (int, float)) and isinstance(old_val, (int, float)):
                    stabilized[key] = weight * value + (1 - weight) * old_val
                else:
                    stabilized[key] = value
            else:
                stabilized[key] = value

        self._stabilized_duals = stabilized
        return stabilized

    def _add_to_column_pool(self, column, reduced_cost):
        self._column_pool.append({
            'column': column,
            'reduced_cost': reduced_cost,
            'age': 0,
        })

        if len(self._column_pool) > self._pool_max_size:
            self._column_pool.sort(key=lambda x: abs(x['reduced_cost']), reverse=True)
            self._column_pool = self._column_pool[:self._pool_max_size // 2]

    def _check_cycling(self, history, window=10):
        if len(history) < window:
            return False

        recent = [h['objective'] for h in history[-window:]]
        if len(recent) < 3:
            return False

        signs = []
        for i in range(1, len(recent)):
            diff = recent[i] - recent[i-1]
            if abs(diff) > 1e-10:
                signs.append(1 if diff > 0 else -1)

        if len(signs) >= 4:
            sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
            if sign_changes >= len(signs) * 0.6:
                return True

        return False

    @staticmethod
    def _run_dual_pricing(pricing_oracle, sub_pricing, duals):
        """Evaluate the pricing oracles for one CG iteration.

        DCG-style runs provide two independent pricing problems that
        only depend on the same dual vector, so they are solved
        concurrently (two workers).  Any failure falls back to the
        serial path, which re-raises real pricing errors unchanged;
        single-oracle runs (plain CG/CCG/DW) stay fully serial.
        """
        if not callable(sub_pricing):
            return pricing_oracle(duals), None
        if not (getattr(pricing_oracle, 'feloopy_parallel_safe', False)
                and getattr(sub_pricing, 'feloopy_parallel_safe', False)):
            # One of the solve paths touches global sys.stdout/stderr
            # (gurobi redirects, pyomo tee) — concurrent solves would
            # race those redirects, so stay serial.
            return pricing_oracle(duals), sub_pricing(duals)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_priced = executor.submit(pricing_oracle, duals)
                fut_sub = executor.submit(sub_pricing, duals)
                priced = fut_priced.result()
                sub_priced = fut_sub.result()
        except Exception:
            priced = pricing_oracle(duals)
            sub_priced = sub_pricing(duals)
        return priced, sub_priced

    def solve(self, master_solver=None, pricing_oracle=None, max_iterations=100,
              tolerance=1e-6, show_log=False, initial_columns=None,
              use_stabilization=True, sub_pricing=None,
              incremental_master=None, add_column_fn=None,
              modified_master_solver=None):
        """Run column generation loop.

        Two modes:
        1. Callback mode (default): master_solver(columns) rebuilds model each iteration.
        2. Incremental mode: incremental_master is an IncrementalModel that is
           built once, and add_column_fn(master_inc, col) adds columns to it.

        If modified_master_solver is provided and the regular master is infeasible,
        the modified master is used to obtain duals for pricing (Phase I approach).
        """
        log = make_logger('ColumnGeneration', show_log)

        _t_start = time.perf_counter()
        master_solver = master_solver or self.master_solver
        pricing_oracle = pricing_oracle or self.pricing_oracle
        if not callable(pricing_oracle):
            raise ColumnGenerationError("pricing_oracle callback is required")
        if incremental_master is None and not callable(master_solver):
            raise ColumnGenerationError(
                "master_solver callback is required in callback mode")
        if initial_columns is not None:
            self.columns = list(initial_columns)

        history = []
        cycling_detected = False
        no_improvement_count = 0
        prev_objective = None
        best_objective = float('inf')
        best_lower_bound = float('-inf')

        if incremental_master is not None:
            return self._solve_incremental(
                incremental_master, add_column_fn, pricing_oracle,
                max_iterations, tolerance, show_log, use_stabilization,
                sub_pricing, history)

        master = None
        for iteration in range(1, max_iterations + 1):
            master = master_solver(list(self.columns))
            if not isinstance(master, dict) or 'objective' not in master:
                raise ColumnGenerationError(
                    "master_solver must return a dict containing objective")

            current_obj = master['objective']
            master_feasible = current_obj is not None and \
                current_obj != float('inf') and current_obj != float('-inf')

            if not master_feasible:
                if modified_master_solver is not None:
                    log(f"iteration={iteration} master infeasible, "
                        f"using modified master")
                    master = modified_master_solver(list(self.columns))
                    if isinstance(master, dict) and 'objective' in master:
                        current_obj = master['objective']
                        master_feasible = current_obj is not None and \
                            current_obj != float('inf') and \
                            current_obj != float('-inf')
                if not master_feasible:
                    log(f"iteration={iteration} master infeasible, "
                        f"no modified master available, skipping")
                    continue

            if prev_objective is not None:
                obj_change = abs(current_obj - prev_objective)
                if obj_change < tolerance * 0.1:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            prev_objective = current_obj

            if current_obj < best_objective:
                best_objective = current_obj

            if no_improvement_count >= 5:
                log(f"Stopping: no improvement for {no_improvement_count} iterations")
                # Pricing never ran for this iteration, so record the
                # stopping state explicitly — otherwise the final
                # objective would be missing from the history.
                history.append({
                    'iteration': iteration,
                    'objective': current_obj,
                    'lower_bound': best_lower_bound,
                    'upper_bound': current_obj,
                    'columns': len(self.columns),
                    'reduced_costs': master.get('reduced_costs'),
                    'new_columns': 0,
                })
                break

            if self._check_cycling(history):
                cycling_detected = True
                log(f"Cycling detected at iteration {iteration}, "
                    f"applying anti-cycling strategy")
                if len(self.columns) > 5:
                    half = len(self.columns) // 2
                    self.columns = self.columns[half:]

            duals = master.get('duals', {})
            if use_stabilization and iteration > 1:
                pricing_duals = self._stabilize_duals(duals, iteration)
            else:
                pricing_duals = duals

            priced, sub_priced = self._run_dual_pricing(
                pricing_oracle, sub_pricing, pricing_duals)
            new_columns = []
            iter_lower_bound = None
            if priced is not None:
                new_columns = list(priced) if isinstance(priced, (list, tuple)) else [priced]
                for col in new_columns:
                    if isinstance(col, dict) and '_lower_bound' in col:
                        iter_lower_bound = col.pop('_lower_bound')
                    if isinstance(col, dict) and 'reduced_cost' in col:
                        self._add_to_column_pool(col, col['reduced_cost'])

            if sub_priced is not None:
                sub_cols = list(sub_priced) if isinstance(sub_priced, (list, tuple)) else [sub_priced]
                new_columns.extend(sub_cols)
                for col in sub_cols:
                    if isinstance(col, dict) and 'reduced_cost' in col:
                        self._add_to_column_pool(col, col['reduced_cost'])

            if not new_columns:
                converged_lb = (master['objective'] if best_lower_bound == float('-inf')
                                else best_lower_bound)
                # Record the convergent iteration too: it is the entry
                # that carries the proven bounds (LB = UB at the CG
                # fixed point), and fast runs that never add a column
                # would otherwise return a completely empty history.
                history.append({
                    'iteration': iteration,
                    'objective': master['objective'],
                    'lower_bound': converged_lb,
                    'upper_bound': master['objective'],
                    'columns': len(self.columns),
                    'reduced_costs': master.get('reduced_costs'),
                    'new_columns': 0,
                })
                return ColumnGenerationResult(
                    status='optimal',
                    objective=master['objective'],
                    variables=master.get('solution', {}),
                    iterations=iteration,
                    runtime_total=time.perf_counter() - _t_start,
                    history=history,
                    columns=list(self.columns),
                    n_columns_added=len(self.columns),
                    lower_bound=converged_lb,
                    upper_bound=master['objective'],
                )

            self.columns.extend(new_columns)
            if iter_lower_bound is not None and iter_lower_bound > best_lower_bound:
                best_lower_bound = iter_lower_bound
            history.append({
                'iteration': iteration,
                'objective': master['objective'],
                'lower_bound': best_lower_bound,
                # The restricted master's value is the incumbent
                # (primal-side) bound for this iteration; without it
                # get_convergence() could never build ub_history.
                'upper_bound': master['objective'],
                'columns': len(self.columns),
                'reduced_costs': master.get('reduced_costs'),
                'new_columns': len(new_columns),
            })
            log(f"iteration={iteration} objective={master['objective']:.6f} "
                f"columns={len(self.columns)} new={len(new_columns)}")

        final = master_solver(list(self.columns))
        return ColumnGenerationResult(
            status='max_iterations' if not cycling_detected else 'cycling_detected',
            objective=final.get('objective'),
            variables=final.get('solution', {}),
            iterations=max_iterations,
            runtime_total=time.perf_counter() - _t_start,
            history=history,
            columns=list(self.columns),
            n_columns_added=len(self.columns),
            lower_bound=best_lower_bound,
            upper_bound=best_objective,
        )

    def _solve_incremental(self, master_inc, add_column_fn, pricing_oracle,
                           max_iterations, tolerance, show_log,
                           use_stabilization, sub_pricing, history):
        """Solve using an IncrementalModel that supports add_column()."""
        log = make_logger('ColumnGeneration', show_log)

        _t_start = time.perf_counter()
        cycling_detected = False
        no_improvement_count = 0
        prev_objective = None
        best_objective = float('inf')
        best_lower_bound = float('-inf')

        for iteration in range(1, max_iterations + 1):
            status, _ = master_inc.solve()
            obj_val = master_inc.get_objective_value()
            if obj_val is None:
                obj_val = float('inf')

            current_obj = float(obj_val)
            if prev_objective is not None:
                obj_change = abs(current_obj - prev_objective)
                if obj_change < tolerance * 0.1:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            prev_objective = current_obj

            if current_obj < best_objective:
                best_objective = current_obj

            if no_improvement_count >= 5:
                log(f"Stopping: no improvement for {no_improvement_count} iterations")
                # Pricing never ran for this iteration, so record the
                # stopping state explicitly — otherwise the final
                # objective would be missing from the history.
                history.append({
                    'iteration': iteration,
                    'objective': current_obj,
                    'lower_bound': best_lower_bound,
                    'upper_bound': current_obj,
                    'columns': len(self.columns),
                    'new_columns': 0,
                })
                break

            if self._check_cycling(history):
                cycling_detected = True
                log(f"Cycling detected at iteration {iteration}")

            duals_list = master_inc.get_row_duals()
            row_count = master_inc.get_num_rows()
            if duals_list is None:
                log(f"WARNING: duals unavailable "
                    f"(solver status may not be optimal); "
                    f"falling back to zero duals")
                duals_list = [0.0] * row_count
            elif len(duals_list) < row_count:
                duals_list = duals_list + [0.0] * (row_count - len(duals_list))
            duals = {i: duals_list[i] for i in range(min(row_count, len(duals_list)))
                     if abs(duals_list[i]) > 1e-12}

            if use_stabilization and iteration > 1:
                pricing_duals = self._stabilize_duals(duals, iteration)
            else:
                pricing_duals = duals

            priced, sub_priced = self._run_dual_pricing(
                pricing_oracle, sub_pricing, pricing_duals)
            new_columns = []
            if priced is not None:
                new_columns = list(priced) if isinstance(priced, (list, tuple)) else [priced]
                for col in new_columns:
                    if isinstance(col, dict) and 'reduced_cost' in col:
                        self._add_to_column_pool(col, col['reduced_cost'])

            if sub_priced is not None:
                sub_cols = list(sub_priced) if isinstance(sub_priced, (list, tuple)) else [sub_priced]
                new_columns.extend(sub_cols)
                for col in sub_cols:
                    if isinstance(col, dict) and 'reduced_cost' in col:
                        self._add_to_column_pool(col, col['reduced_cost'])

            # Pricing may carry a dual-based bound (same convention as
            # the callback path): pop it off before the column is
            # stored/added and keep the best one seen so far.
            for col in new_columns:
                if isinstance(col, dict) and '_lower_bound' in col:
                    lb_val = col.pop('_lower_bound')
                    if lb_val > best_lower_bound:
                        best_lower_bound = lb_val

            if not new_columns:
                solution = master_inc.get_all_variable_values()
                # Record the convergent iteration too: it is the entry
                # that carries the proven bounds (LB = UB at the CG
                # fixed point), and fast runs that never add a column
                # would otherwise return a completely empty history.
                history.append({
                    'iteration': iteration,
                    'objective': current_obj,
                    'lower_bound': current_obj,
                    'upper_bound': current_obj,
                    'columns': len(self.columns),
                    'new_columns': 0,
                })
                return ColumnGenerationResult(
                    objective=current_obj,
                    variables=solution,
                    columns=list(self.columns),
                    iterations=iteration,
                    status='optimal',
                    history=history,
                    runtime_total=time.perf_counter() - _t_start,
                    n_columns_added=len(self.columns),
                    # Pricing found no improving column: the restricted
                    # master's objective is the proven CG optimum (same
                    # convention as the callback-mode converged return).
                    lower_bound=current_obj,
                    upper_bound=current_obj,
                )

            for col in new_columns:
                if callable(add_column_fn):
                    add_column_fn(master_inc, col)
                self.columns.append(col)

            history.append({
                'iteration': iteration,
                'objective': current_obj,
                'lower_bound': best_lower_bound,
                # The restricted master's value is the incumbent
                # (primal-side) bound for this iteration; without it
                # get_convergence() could never build ub_history.
                'upper_bound': current_obj,
                'columns': len(self.columns),
                'new_columns': len(new_columns),
            })
            log(f"iteration={iteration} objective={current_obj:.6f} "
                f"columns={len(self.columns)} new={len(new_columns)}")

        solution = master_inc.get_all_variable_values()
        return ColumnGenerationResult(
            objective=best_objective,
            variables=solution,
            columns=list(self.columns),
            iterations=max_iterations,
            status='max_iterations' if not cycling_detected else 'cycling_detected',
            history=history,
            runtime_total=time.perf_counter() - _t_start,
            n_columns_added=len(self.columns),
            lower_bound=best_lower_bound,
            upper_bound=best_objective,
        )

    @classmethod
    def automatic(cls, master_solver, pricing_oracle, **kwargs):
        return cls(master_solver, pricing_oracle).solve(**kwargs)
