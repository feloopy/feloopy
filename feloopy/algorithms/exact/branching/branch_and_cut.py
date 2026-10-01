# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Branch-and-Cut: Branch-and-Bound with lazy cut separation.

Composes :class:`BranchAndBound` with separation oracles.  Cuts are
generated at integer-feasible nodes and injected back as lazy or
regular constraints depending on the solver's callback support.

Manual API::

    bac = BranchAndCut()
    bac.add_separation_oracle(my_oracle)
    result = bac.solve(model_fn, interface='gurobi', show_log=True)

Automatic API::

    bac = BranchAndCut.from_automatic(model_fn, directions, interface,
                                       solver, show_log=True)
    result = bac.solve(interface='gurobi')
"""

import time
from typing import List, Dict, Any, Optional

from .core import BranchAndBound
from .enums import BranchingStatus, NodeStrategy, SelectionStrategy
from .result import BranchAndCutResult
from .callbacks import BranchingCallback, SeparationOracle
from .cuts import Cut, CutPool, LazyConstraint, CutType, SeparationOracles
from ..base.logging import make_logger

_PREFIXES = ('pvar', 'ivar', 'bvar', 'fvar')


def _find_var_col(var_obj):
    """Return the solver column index for a feloopy variable."""
    if isinstance(var_obj, dict):
        for v in var_obj.values():
            return v.index if hasattr(v, 'index') else v._col
    return var_obj.index if hasattr(var_obj, 'index') else var_obj._col


def _find_gurobi_var(var_obj):
    """Return the Gurobi Var object (first element for indexed)."""
    if isinstance(var_obj, dict):
        return list(var_obj.values())[0]
    return var_obj


class BranchAndCut:
    """Branch-and-Cut: Branch-and-Bound with lazy cut separation.

    Composes BranchAndBound with separation oracles.  Cuts are
    generated at integer-feasible nodes and added back as lazy
    constraints or regular constraints.
    """

    def __init__(self):
        """Initialise a Branch-and-Cut solver.

        Configure with ``add_separation_oracle`` or ``add_lazy_constraint``,
        then call ``solve()``.
        """
        self._bab = BranchAndBound()
        self._callback = BranchingCallback()
        self._oracles: List[SeparationOracle] = []
        self._automatic_config: Optional[Dict[str, Any]] = None

    def add_lazy_constraint(self, lc: LazyConstraint):
        """Register a :class:`LazyConstraint` with the internal callback."""
        self._callback.add_lazy_constraint(lc)
        return self

    def add_separation_oracle(self, oracle, priority: int = 0,
                              max_cuts_per_round: int = 100):
        """Wrap a separation oracle in a :class:`LazyConstraint` and register."""
        if isinstance(oracle, SeparationOracle):
            lc = LazyConstraint(
                name=type(oracle).__name__, separation=oracle,
                priority=priority, max_cuts_per_round=max_cuts_per_round,
                is_valid=getattr(oracle, 'is_valid', None))
            self._oracles.append(oracle)
        elif callable(oracle):
            lc = LazyConstraint(
                name=getattr(oracle, '__name__', 'custom_oracle'),
                separation=oracle, priority=priority,
                max_cuts_per_round=max_cuts_per_round)
        else:
            raise TypeError(
                f"oracle must be SeparationOracle or callable, "
                f"got {type(oracle)}")
        self._callback.add_lazy_constraint(lc)
        return self

    def set_node_strategy(self, strategy: NodeStrategy):
        """Delegate node exploration order to the internal B&B."""
        self._bab.set_node_strategy(strategy)
        return self

    def set_selection_strategy(self, strategy: SelectionStrategy):
        """Delegate variable selection to the internal B&B."""
        self._bab.set_selection_strategy(strategy)
        return self

    @classmethod
    def from_automatic(cls, model_fn, directions, interface='highs',
                       solver=None, show_log=False):
        """Auto-configure a :class:`BranchAndCut` from a model factory."""
        bac = cls()
        bac._automatic_config = {
            'model_fn': model_fn,
            'directions': directions or ['min'],
            'interface': interface,
            'solver': solver or interface,
            'show_log': show_log,
        }
        return bac

    def solve(self, model_fn=None, interface='highs', solver=None,
              directions=None, max_nodes=1000, max_time=None,
              tolerance=1e-6, show_log=False, save_vars=True, **kwargs):
        """Run Branch-and-Cut and return a :class:`BranchAndCutResult`."""
        log = make_logger('BranchAndCut', show_log)
        t_start = time.perf_counter()

        if model_fn is None and self._automatic_config is not None:
            cfg = self._automatic_config
            model_fn, directions = cfg['model_fn'], directions or cfg['directions']
            interface, solver = cfg['interface'], cfg['solver'] or cfg['interface']
            show_log = cfg['show_log']
            log = make_logger('BranchAndCut', show_log)
        elif model_fn is None:
            raise ValueError("model_fn required when not using from_automatic()")
        if directions is None:
            directions = ['min']
        if solver is None:
            solver = interface

        log(f"Starting Branch-and-Cut  |  interface={interface}  "
            f"solver={solver}  sense={directions[0]}")

        if interface == 'gurobi':
            result = self._solve_gurobi(model_fn, solver, directions,
                                        max_nodes, max_time, tolerance,
                                        show_log, save_vars, **kwargs)
        else:
            result = self._solve_generic(model_fn, interface, solver,
                                         directions, max_nodes, max_time,
                                         tolerance, show_log, save_vars,
                                         **kwargs)

        result.runtime_total = time.perf_counter() - t_start
        cb_stats = self._callback.get_statistics()
        result.n_cuts_added = cb_stats.get('total_cuts', 0)
        result.cut_generation_time = cb_stats.get('separation_time', 0.0)
        log(f"Finished  |  status={result.status}  "
            f"objective={result.objective:.6f}  "
            f"cuts={result.n_cuts_added}  "
            f"nodes={result.n_nodes_explored}  "
            f"time={result.runtime_total:.2f}s")
        return result

    def _solve_gurobi(self, model_fn, solver, directions,
                      max_nodes, max_time, tolerance, show_log,
                      save_vars, **kwargs):
        """Solve using Gurobi's native lazy-constraint callback."""
        log = make_logger('BranchAndCut', show_log)

        from feloopy.feloopy import model as _feloopy_model

        m = _feloopy_model(interface='gurobi', validate=False)
        m = model_fn(m)
        m.features['directions'] = directions
        m.features['objective_being_optimized'] = 0
        m.features['solver_name'] = solver

        native = m.model
        if native is None:
            log("WARNING: native model unavailable; falling back")
            return self._solve_generic(model_fn, 'gurobi', solver,
                                       directions, max_nodes, max_time,
                                       tolerance, show_log, save_vars,
                                       **kwargs)

        native.setParam('LazyConstraints', 1)
        native.setParam('OutputFlag', 0 if not show_log else 1)
        if max_time is not None:
            native.setParam('TimeLimit', max_time)
        if tolerance is not None:
            native.setParam('MIPGap', tolerance)
        if kwargs.get('cpu_threads'):
            try:
                native.setParam('Threads', int(kwargs['cpu_threads']))
            except Exception:
                pass
        for k, v in kwargs.get('options', {}).items():
            try:
                native.setParam(k, v)
            except Exception:
                pass

        # Constraints and the objective are stored in features and are
        # materialised by the solution generator on the normal solve
        # path; this native callback path calls optimize() directly, so
        # apply them here first — otherwise Gurobi searches an empty
        # model with the default objective and reports ObjVal=0.0.
        objectives = m.features.get('objectives', [])
        if objectives:
            import gurobipy as gurobi_interface
            obj_id = m.features.get('objective_being_optimized', 0)
            obj = (objectives[obj_id] if obj_id < len(objectives)
                   else objectives[0])
            direction = (directions[obj_id] if obj_id < len(directions)
                         else 'min')
            native.setObjective(
                obj,
                gurobi_interface.GRB.MAXIMIZE if direction == 'max'
                else gurobi_interface.GRB.MINIMIZE)
        constraint_labels = m.features.get('constraint_labels') or []
        for i, constraint in enumerate(m.features.get('constraints', [])):
            label = (constraint_labels[i]
                     if i < len(constraint_labels) else None)
            if label is not None:
                native.addConstr(constraint, name=str(label))
            else:
                native.addConstr(constraint)
        native.update()

        log("Solving MIP with lazy constraint callbacks...")
        native.optimize(self._callback.gurobi_callback)
        return self._build_result_from_gurobi(native, m, directions, save_vars)

    def _build_result_from_gurobi(self, native, feloopy_model,
                                  directions, save_vars):
        """Build a BranchAndCutResult from a solved Gurobi model."""
        result = BranchAndCutResult()
        try:
            result.objective = native.getAttr('ObjVal')
        except Exception:
            result.objective = (float('inf') if directions[0] == 'min'
                                else float('-inf'))
        try:
            import gurobipy as gurobi
            status_map = {
                gurobi.GRB.OPTIMAL: BranchingStatus.OPTIMAL,
                gurobi.GRB.SUBOPTIMAL: BranchingStatus.FEASIBLE,
                gurobi.GRB.INFEASIBLE: BranchingStatus.INFEASIBLE,
                gurobi.GRB.UNBOUNDED: BranchingStatus.UNBOUNDED,
                gurobi.GRB.TIME_LIMIT: BranchingStatus.MAX_TIME,
            }
            result.status = status_map.get(
                native.getAttr('Status'), BranchingStatus.FEASIBLE)
        except (ImportError, Exception):
            result.status = BranchingStatus.FEASIBLE
        try:
            result.n_nodes_explored = native.getAttr('NodeCount')
        except Exception:
            result.n_nodes_explored = 0
        try:
            result.gap = native.getAttr('MIPGap')
        except Exception:
            result.gap = float('inf')
        if save_vars:
            result.variables = self._extract_vars_from_native(native)
        return result

    def _extract_vars_from_native(self, native):
        """Pull variable values from a solved native solver model."""
        variables = {}
        try:
            for v in native.getVars():
                name = v.VarName
                for pfx in _PREFIXES:
                    if name.startswith(pfx + '_'):
                        var_name = name[len(pfx) + 1:]
                        if var_name not in variables:
                            variables[var_name] = {}
                        try:
                            variables[var_name][pfx] = v.X
                        except Exception:
                            variables[var_name][pfx] = None
                        break
        except Exception:
            pass
        flat = {}
        for var_name, pvars in variables.items():
            flat[var_name] = (list(pvars.values())[0]
                              if len(pvars) == 1 else pvars)
        return flat

    def _solve_generic(self, model_fn, interface, solver, directions,
                       max_nodes, max_time, tolerance, show_log,
                       save_vars, **kwargs):
        """Polling-based B&C for solvers without native lazy callbacks."""
        log = make_logger('BranchAndCut', show_log)

        from ..base.model import create_model, solve_model, get_obj, get_status
        from feloopy.generators import solution_generator

        solver_opts = dict(
            interface_name=interface,
            solver_name=solver,
            debug_mode=False,
            write_model_file=False,
            save_solver_log=False,
            time_limit=max_time,
            absolute_gap=tolerance,
            relative_gap=tolerance,
            thread_count=kwargs.get('cpu_threads'),
            email_address=None,
            max_iterations=None,
            log=show_log,
            solver_options={},
        )
        m = create_model(model_fn, directions, 0, solver_opts)

        cut_rounds = kwargs.get('cut_rounds', 20)
        total_cuts = 0
        cut_time = 0.0
        initial_solution = None
        initial_obj = None

        log("Polling-based cut generation (pre-B&B)...")
        for rnd in range(cut_rounds):
            t_cut = time.perf_counter()
            solve_model(m, solution_generator)
            cut_time += time.perf_counter() - t_cut

            if not m.healthy():
                log(f"  Round {rnd + 1}: infeasible or unsolvable")
                break

            # The polling solve runs the full MIP — keep its incumbent
            # as the B&B seed.  _extract_solution below drops array
            # variables (float() on an ndarray fails), which used to
            # throw the found solution away and left the tree with no
            # upper bound (objective=inf at MAX_TIME).
            try:
                cand = self._bab._extract_var_values(m)
                cand_obj = get_obj(m)
                if cand and cand_obj is not None and (
                        self._bab._is_integer_feasible(m, cand, tolerance)):
                    initial_solution = cand
                    initial_obj = float(cand_obj)
                    log(f"  Round {rnd + 1}: incumbent from MIP solve "
                        f"obj={initial_obj:.6f}")
            except Exception:
                pass

            solution = self._extract_solution(m)
            if not solution:
                log(f"  Round {rnd + 1}: no solution extracted")
                break

            new_cuts = self._callback.generic_poll(solution)
            if not new_cuts:
                log(f"  Round {rnd + 1}: no new cuts")
                break

            for cut in new_cuts:
                self._add_cut_to_model(m, cut, interface)
                total_cuts += 1
            log(f"  Round {rnd + 1}: +{len(new_cuts)} cuts "
                f"(total={total_cuts})")

        log(f"Cut generation done: {total_cuts} cuts in {cut_time:.2f}s")

        # When no cuts were added, the polling solution is valid for
        # the original model; if the solver proved optimality, the
        # Python B&B tree (whose LP bounds are weaker) can only spend
        # the whole budget re-proving it.  Return the solver's own
        # proof instead — the same contract as the native Gurobi
        # callback path, which returns native.optimize()'s result.
        if (total_cuts == 0 and initial_solution is not None
                and get_status(m) == 'optimal'):
            log("Pre-B&B MIP solve proved optimality; "
                "returning the solver's proof")
            result = BranchAndCutResult()
            result.status = BranchingStatus.OPTIMAL
            result.objective = initial_obj
            result.variables = initial_solution
            result.best_bound = initial_obj
            result.gap = 0.0
            result.n_nodes_explored = 1
            result.solution_count = 1
            result.iterations = 1
            result.separation_time = cut_time
            return result

        log("Proceeding to Branch-and-Bound...")

        self._callback._cuts_added_total = total_cuts
        self._callback._cuts_added_per_iteration = [total_cuts]

        bab_result = self._bab.solve(
            lambda mod: m, interface=interface, solver=solver,
            directions=directions, max_nodes=max_nodes,
            max_time=max_time, tolerance=tolerance,
            show_log=show_log, save_vars=save_vars,
            initial_solution=initial_solution, initial_obj=initial_obj,
            **kwargs)

        result = BranchAndCutResult()
        for attr in ('status', 'objective', 'variables', 'iterations',
                     'history', 'n_nodes_explored', 'n_pruned',
                     'n_infeasible', 'best_bound', 'gap',
                     'solution_count'):
            setattr(result, attr, getattr(bab_result, attr))
        result.separation_time = cut_time
        return result

    def _extract_solution(self, feloopy_model):
        """Extract a {var_name: value} dict from a solved feloopy model."""
        solution = {}
        try:
            variables = feloopy_model.features.get('variables', {})
            for (prefix, var_name), var_obj in variables.items():
                if prefix not in _PREFIXES:
                    continue
                try:
                    solution[var_name] = float(
                        feloopy_model.get_numpy_var(var_name))
                except Exception:
                    if hasattr(var_obj, 'X'):
                        solution[var_name] = var_obj.X
        except Exception:
            pass
        return solution

    def _add_cut_to_model(self, m, cut: Cut, interface: str):
        """Add a Cut as a regular constraint to the feloopy model."""
        variables = m.features.get('variables', {})
        if interface == 'highs':
            self._add_cut_highs(m, cut, variables)
        elif interface == 'gurobi':
            self._add_cut_gurobi(m, cut, variables)
        elif interface == 'cplex':
            self._add_cut_cplex(m, cut, variables)

    def _add_cut_highs(self, m, cut, variables):
        """Add a cut row to a HiGHS model."""
        import numpy as np
        indices, values = [], []
        for var_name, coeff in cut.coefficients.items():
            for pfx in _PREFIXES:
                var_obj = variables.get((pfx, var_name))
                if var_obj is None:
                    continue
                if isinstance(var_obj, dict):
                    for v in var_obj.values():
                        indices.append(_find_var_col(v))
                        values.append(coeff)
                else:
                    indices.append(_find_var_col(var_obj))
                    values.append(coeff)
                break
        if not indices:
            return
        rl = cut.rhs if cut.sense == '>=' else -1e20
        ru = cut.rhs if cut.sense in ('<=', '==') else 1e20
        if cut.sense == '<=':
            rl, ru = -1e20, cut.rhs
        m.model.addRows(
            1, np.array([rl]), np.array([ru]),
            np.array([len(indices)]), np.array(indices),
            np.array(values))

    def _add_cut_gurobi(self, m, cut, variables):
        """Add a cut constraint to a Gurobi model."""
        try:
            import gurobipy as gurobi
            expr = gurobi.LinExpr()
            for var_name, coeff in cut.coefficients.items():
                for pfx in _PREFIXES:
                    var_obj = variables.get((pfx, var_name))
                    if var_obj is None:
                        continue
                    expr += coeff * _find_gurobi_var(var_obj)
                    break
            if cut.sense == '<=':
                m.model.addConstr(expr <= cut.rhs)
            elif cut.sense == '>=':
                m.model.addConstr(expr >= cut.rhs)
            else:
                m.model.addConstr(expr == cut.rhs)
            m.model.update()
        except ImportError:
            pass

    def _add_cut_cplex(self, m, cut, variables):
        """Add a cut constraint to a CPLEX model."""
        try:
            import cplex as cpx
            names, coeffs = [], []
            for var_name, coeff in cut.coefficients.items():
                for pfx in _PREFIXES:
                    var_obj = variables.get((pfx, var_name))
                    if var_obj is None:
                        continue
                    objs = (list(var_obj.values())
                            if isinstance(var_obj, dict) else [var_obj])
                    for v in objs:
                        nm = (v.get_name() if hasattr(v, 'get_name')
                              else str(v))
                        names.append(nm)
                        coeffs.append(float(coeff))
                    break
            if names:
                sense_map = {'<=': 'L', '>=': 'G', '==': 'E'}
                m.model.linear_constraints.add(
                    lin_expr=[cpx.SparsePair(names, coeffs)],
                    senses=[sense_map.get(cut.sense, 'L')],
                    rhs=[float(cut.rhs)])
        except ImportError:
            pass

    def get_statistics(self) -> Dict[str, Any]:
        """Return cut-generation statistics from the last solve."""
        return self._callback.get_statistics()

    def reset(self):
        """Reset all internal state for a fresh solve."""
        self._callback.reset()
        self._oracles.clear()
