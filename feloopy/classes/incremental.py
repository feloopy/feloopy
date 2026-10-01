# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""
Solver-independent incremental model API.

Provides an ``IncrementalModel`` that builds the solver model once and
then supports incremental operations (add/remove constraints, fix/unfix
variables, re-solve) without rebuilding the entire model from scratch.

This eliminates the per-iteration overhead of ``_create_model`` +
``generate_solution`` in decomposition algorithms (Benders, Lagrangian,
Column Generation, etc.).

Supported interfaces: HiGHS, Gurobi, CPLEX, OR-Tools, PuLP, COPT,
SCIP, Xpress, MOSEK, and any exact solver that exposes standard
addRow / addConstr / changeColBounds / run / optimize primitives.
"""

import timeit
import numpy as np

from ..helpers.containers import to_indexed_dict


class IncrementalModel:
    """Solver-independent incremental model.

    Wraps a feloopy ``model`` instance and provides an incremental
    interface for iterative algorithms.  The solver model is built once
    via ``build()``, and subsequent modifications (add constraints,
    fix variables, re-solve) operate directly on the native solver
    model without recreating feloopy objects.

    Examples
    --------
    >>> m = flp.model(interface='highs')
    >>> m = my_model_fn(m)
    >>> inc = m.build_incremental(directions=['min'])
    >>> result, times = inc.build()
    >>> # Add a cut
    >>> inc.add_row(lower=5.0, upper=1e20, indices=[0, 1], values=[1.0, -1.0])
    >>> # Fix a variable
    >>> inc.fix_variable('open', {0: 1.0, 1: 0.0})
    >>> result, times = inc.solve()
    """

    def __init__(self, feloopy_model, directions=None, obj_index=0,
                 solver_name=None, solver_options=None,
                 time_limit=None, thread_count=None,
                 absolute_gap=None, relative_gap=None,
                 log=False, debug=False):
        """
        Parameters
        ----------
        feloopy_model : model
            A feloopy model instance with variables, constraints, and
            objectives already defined.
        directions : list of str
            Optimization directions, e.g. ``['min']``.
        obj_index : int
            Which objective to optimize (default 0).
        solver_name : str, optional
            Solver engine name (defaults to the model's interface).
        solver_options : dict, optional
            Solver-specific options.
        """
        self._m = feloopy_model
        self._features = feloopy_model.features
        self._directions = directions or self._features.get('directions', ['min'])
        self._obj_index = obj_index
        self._interface = self._features.get('interface_name', 'highs')
        self._solver_name = solver_name or self._features.get(
            'solver_name', self._interface)
        self._solver_options = solver_options or {}
        self._time_limit = time_limit
        self._thread_count = thread_count
        self._absolute_gap = absolute_gap
        self._relative_gap = relative_gap
        self._log = log
        self._debug = debug

        self._built = False
        self._var_col_map = {}
        self._original_bounds = {}
        self._initial_row_count = 0
        self._initial_col_count = 0
        self._cut_count = 0
        self._last_result = None
        self._last_times = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Build the solver model from current features and solve once.

        Returns
        -------
        tuple
            ``(solver_result, [t_begin, t_end])``
        """
        self._set_solver_options()
        self._add_all_constraints()
        self._set_objective()
        t0 = timeit.default_timer()
        self._solve_native()
        t1 = timeit.default_timer()
        self._last_result = self._get_native_result()
        self._last_times = [t0, t1]
        self._built = True
        self._capture_state()
        return self._last_result, self._last_times

    # ------------------------------------------------------------------
    # Incremental constraint operations
    # ------------------------------------------------------------------

    def add_row(self, lower, upper, indices, values, label=None):
        """Add a single row (constraint) to the native solver model.

        Parameters
        ----------
        lower : float
            Lower bound of the row.
        upper : float
            Upper bound of the row (use ``1e20`` for no upper bound).
        indices : list of int
            Column indices of non-zero coefficients.
        values : list of float
            Coefficient values.
        label : str, optional
            Row name.
        """
        self._add_row_native(lower, upper, indices, values, label)
        self._cut_count += 1

    def add_rows(self, rows):
        """Add multiple rows at once.

        Parameters
        ----------
        rows : list of tuple
            Each tuple is ``(lower, upper, indices, values, label)``.
        """
        for row in rows:
            if len(row) == 5:
                lower, upper, indices, values, label = row
            elif len(row) == 4:
                lower, upper, indices, values = row
                label = None
            else:
                raise ValueError(
                    "Each row must be (lower, upper, indices, values) "
                    "or (lower, upper, indices, values, label)")
            self.add_row(lower, upper, indices, values, label)

    # ------------------------------------------------------------------
    # Variable fixing (bound-based, O(1) per variable)
    # ------------------------------------------------------------------

    def fix_variable(self, var_name, value):
        """Fix a variable to a specific value via bounds.

        Parameters
        ----------
        var_name : str
            Variable name as defined in the feloopy model.
        value : float or dict
            Value to fix to.  For indexed variables, pass a dict
            ``{index: value}``.
        """
        if var_name not in self._original_bounds:
            self._capture_var_bounds(var_name)

        col_info = self._var_col_map.get(var_name)
        if col_info is None:
            return

        if isinstance(col_info, dict):
            for idx, col_idx in col_info.items():
                try:
                    val = float(value[idx]) if isinstance(value, dict) else float(value)
                    self._set_col_bounds(col_idx, val, val)
                except Exception:
                    pass
        else:
            try:
                val = float(value)
                self._set_col_bounds(col_info, val, val)
            except Exception:
                pass

    def unfix_variable(self, var_name):
        """Restore original bounds for a variable.

        Parameters
        ----------
        var_name : str
            Variable name as defined in the feloopy model.
        """
        orig = self._original_bounds.get(var_name)
        if orig is None:
            return

        col_info = self._var_col_map.get(var_name)
        if col_info is None:
            return

        if isinstance(col_info, dict):
            for idx, col_idx in col_info.items():
                lb, ub = orig.get(idx, (-1e20, 1e20))
                self._set_col_bounds(col_idx, lb, ub)
        else:
            lb, ub = orig if isinstance(orig, tuple) else (-1e20, 1e20)
            self._set_col_bounds(col_info, lb, ub)

    def fix_variables(self, fixes):
        """Fix multiple variables at once.

        Parameters
        ----------
        fixes : dict
            ``{var_name: value}`` mapping.
        """
        for var_name, value in fixes.items():
            self.fix_variable(var_name, value)

    def unfix_variables(self, var_names):
        """Restore bounds for multiple variables.

        Parameters
        ----------
        var_names : list of str
            Variable names to unfix.
        """
        for var_name in var_names:
            self.unfix_variable(var_name)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self):
        """Re-solve the native solver model with current state.

        Returns
        -------
        tuple
            ``(solver_result, [t_begin, t_end])``
        """
        t0 = timeit.default_timer()
        self._solve_native()
        t1 = timeit.default_timer()
        self._last_result = self._get_native_result()
        self._last_times = [t0, t1]
        return self._last_result, self._last_times

    # ------------------------------------------------------------------
    # Incremental objective operations
    # ------------------------------------------------------------------

    def update_objective_coefficients(self, col_costs):
        """Update column costs in the objective without rebuilding.

        Parameters
        ----------
        col_costs : dict
            ``{col_index: cost}`` or ``{var_name: cost}`` mapping.
            If keys are strings (variable names), they are resolved
            via ``var_col_map``.
        """
        m = self._get_native_model()
        iface = self._interface

        for key, cost in col_costs.items():
            if isinstance(key, str):
                col_info = self._var_col_map.get(key)
                if col_info is None:
                    continue
                if isinstance(col_info, dict):
                    for idx, col_idx in col_info.items():
                        if isinstance(cost, dict):
                            c = cost.get(idx, 0.0)
                        else:
                            c = cost
                        self._change_col_cost(col_idx, c)
                else:
                    self._change_col_cost(col_info, cost)
            else:
                self._change_col_cost(key, cost)

    def _change_col_cost(self, col_idx, cost):
        """Set the cost of a single column."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            m.changeColCost(col_idx, cost)
        elif iface == 'gurobi':
            m.getVars()[col_idx].Obj = cost
        elif iface == 'cplex':
            m.cplex.objective.set_linear(col_idx, cost)
        elif iface == 'copt':
            all_vars = m.getVars()
            if col_idx < len(all_vars):
                all_vars[col_idx].Obj = cost
        elif iface == 'pulp':
            vars_list = list(m.variables())
            if col_idx < len(vars_list):
                v = vars_list[col_idx]
                if m.sense == 1:
                    v.obj = cost
                else:
                    v.obj = cost
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                vars_list = list(m.component_data_objects(pyo.Var, active=True))
                if col_idx < len(vars_list):
                    obj = list(m.component_data_objects(pyo.Objective, active=True))
                    if obj:
                        obj[0].expr += cost * vars_list[col_idx]
            except Exception:
                pass
        else:
            try:
                m.changeColCost(col_idx, cost)
            except Exception:
                pass

    def set_objective_direction(self, direction):
        """Change optimization direction (min/max) without rebuilding.

        Parameters
        ----------
        direction : str
            ``'min'`` or ``'max'``.
        """
        self._directions = [direction]

    # ------------------------------------------------------------------
    # Incremental column operations (for Column Generation)
    # ------------------------------------------------------------------

    def add_column(self, cost, indices, values, lb=0.0, ub=1e20,
                   name=None, is_integer=False):
        """Add a single column (variable) to the native solver model.

        Parameters
        ----------
        cost : float
            Objective coefficient.
        indices : list of int
            Row indices with non-zero coefficients.
        values : list of float
            Coefficient values for the new column.
        lb : float
            Lower bound.
        ub : float
            Upper bound.
        name : str, optional
            Column name.
        is_integer : bool
            If True, add as integer variable.

        Returns
        -------
        int
            Column index of the new variable.
        """
        return self._add_col_native(cost, indices, values, lb, ub,
                                    name, is_integer)

    def add_columns(self, columns):
        """Add multiple columns at once.

        Parameters
        ----------
        columns : list of tuple
            Each tuple is ``(cost, indices, values, lb, ub, name, is_integer)``
            or a subset thereof.
        """
        col_indices = []
        for col in columns:
            if len(col) >= 5:
                cost, indices, values, lb, ub = col[:5]
                name = col[5] if len(col) > 5 else None
                is_integer = col[6] if len(col) > 6 else False
            else:
                raise ValueError(
                    "Each column must be (cost, indices, values, lb, ub, "
                    "[name, is_integer])")
            col_idx = self.add_column(cost, indices, values, lb, ub,
                                      name, is_integer)
            col_indices.append(col_idx)
        return col_indices

    def _add_col_native(self, cost, indices, values, lb, ub,
                        name=None, is_integer=False):
        """Add a column to the native solver model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            col_idx = m.getNumCol()
            m.addCol(cost, lb, ub, len(indices),
                     list(indices), list(values))
            if is_integer:
                m.changeColIntegrality(col_idx, 1)  # kInteger = 1
            if name:
                m.passColName(col_idx, name)
            return col_idx

        elif iface == 'gurobi':
            import gurobipy
            vtype = gurobipy.GRB.INTEGER if is_integer else gurobipy.GRB.CONTINUOUS
            col_vars = m.getVars()
            v = m.addVar(lb=lb, ub=ub, obj=cost, vtype=vtype,
                         name=name or f'col_{m.NumVars}')
            m.update()
            # Add existing row coefficients
            if indices and values:
                expr = gurobipy.LinExpr(values, [col_vars[i] for i in indices])
                m.addConstr(expr <= 0)  # placeholder — caller should use add_row
            return m.NumVars - 1

        elif iface == 'cplex':
            m.cplex.variables.add(
                lb=[lb], ub=[ub], obj=[cost],
                types=['I' if is_integer else 'C'],
                names=[name] if name else None)
            return m.cplex.variables.get_num() - 1

        elif iface == 'copt':
            import coptpy
            vtype = coptpy.COPT.INTEGER if is_integer else coptpy.COPT.CONTINUOUS
            v = m.addVar(lb=lb, ub=ub, obj=cost, vtype=vtype,
                         name=name or f'col_{len(m.getVars())}')
            m.update()
            return len(m.getVars()) - 1

        elif iface == 'pulp':
            import pulp
            from ..generators.pulp_compat import make_variable, problem_variables
            vtype = pulp.LpInteger if is_integer else pulp.LpContinuous
            v = make_variable(m, name or f'col_{len(problem_variables(m))}', lb, ub, vtype)
            v.obj = cost
            m += 0 * v
            return len(problem_variables(m)) - 1

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                vname = name or f'col_{sum(1 for _ in m.component_data_objects(pyo.Var, active=True))}'
                vdomain = pyo.Integers if is_integer else pyo.Reals
                v = pyo.Var(bounds=(lb, ub), domain=vdomain)
                m.add_component(vname, v)
                return sum(1 for _ in m.component_data_objects(pyo.Var, active=True)) - 1
            except Exception:
                return -1

        else:
            try:
                col_idx = m.getNumCol()
                m.addCol(cost, lb, ub, len(indices),
                         list(indices), list(values))
                return col_idx
            except Exception:
                return -1

    # ------------------------------------------------------------------
    # Incremental row removal (for Column Generation, Lagrangian)
    # ------------------------------------------------------------------

    def remove_rows(self, row_indices):
        """Remove rows by index (solver-specific).

        Parameters
        ----------
        row_indices : list of int
            Row indices to remove, in descending order.
        """
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            for idx in sorted(row_indices, reverse=True):
                m.deleteRows(1, [idx])

        elif iface == 'gurobi':
            constrs = m.getConstrs()
            for idx in sorted(row_indices, reverse=True):
                if idx < len(constrs):
                    m.remove(constrs[idx])
            m.update()

        elif iface == 'cplex':
            m.cplex.linear_constraints.delete(
                indices=row_indices)

        elif iface == 'copt':
            m.update()
            constrs = m.getConstrs()
            for idx in sorted(row_indices, reverse=True):
                if idx < len(constrs):
                    m.remove(constrs[idx])
            m.update()

        elif iface == 'pulp':
            from ..generators.pulp_compat import PULP_V4
            if PULP_V4:
                raise NotImplementedError(
                    "PuLP 4 cannot remove constraints from a problem, so "
                    "remove_rows() is only available on PuLP 3.x.")
            labels = list(m.constraints.keys())
            for idx in sorted(row_indices, reverse=True):
                if idx < len(labels):
                    del m.constraints[labels[idx]]

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                cons = list(m.component_data_objects(pyo.Constraint, active=True))
                for idx in sorted(row_indices, reverse=True):
                    if idx < len(cons):
                        cons[idx].deactivate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Solution accessors
    # ------------------------------------------------------------------

    def get_objective_value(self):
        """Return the objective value of the last solve."""
        return self._get_objective_native()

    def get_variable_value(self, var_name):
        """Return the value of a variable from the last solve.

        Parameters
        ----------
        var_name : str
            Variable name.

        Returns
        -------
        float or dict
            Scalar value for scalar variables, dict ``{index: value}``
            for indexed variables.
        """
        col_info = self._var_col_map.get(var_name)
        if col_info is None:
            return None

        sol = self._get_solution_values()
        if sol is None:
            return None

        if isinstance(col_info, dict):
            return {idx: sol[col_idx] for idx, col_idx in col_info.items()
                    if col_idx < len(sol)}
        else:
            return sol[col_info] if col_info < len(sol) else None

    def get_all_variable_values(self):
        """Return values of all variables as a dict."""
        sol = self._get_solution_values()
        if sol is None:
            return {}
        result = {}
        for var_name, col_info in self._var_col_map.items():
            if isinstance(col_info, dict):
                result[var_name] = {
                    idx: sol[col_idx]
                    for idx, col_idx in col_info.items()
                    if col_idx < len(sol)
                }
            else:
                result[var_name] = sol[col_info] if col_info < len(sol) else None
        return result

    def get_row_duals(self):
        """Return row dual values from the last solve."""
        return self._get_duals_native()

    def get_reduced_costs(self):
        """Return column reduced costs from the last solve."""
        return self._get_reduced_costs_native()

    def get_num_rows(self):
        """Return the current number of rows (constraints) in the model."""
        return self._get_num_rows_native()

    def get_num_cols(self):
        """Return the current number of columns (variables) in the model."""
        return self._get_num_cols_native()

    @property
    def var_col_map(self):
        """Variable name to column index mapping."""
        return self._var_col_map

    @property
    def interface(self):
        """The solver interface name."""
        return self._interface

    @property
    def native_model(self):
        """The underlying native solver model object."""
        return self._get_native_model()

    # ------------------------------------------------------------------
    # Internal: state capture
    # ------------------------------------------------------------------

    def _capture_state(self):
        """Capture variable-to-column mapping and initial bounds."""
        self._var_col_map = self._build_var_col_map()
        self._initial_row_count = self._get_num_rows_native()
        self._initial_col_count = self._get_num_cols_native()

    def _capture_var_bounds(self, var_name):
        """Capture original bounds for a variable before fixing."""
        col_info = self._var_col_map.get(var_name)
        if col_info is None:
            return
        if isinstance(col_info, dict):
            self._original_bounds[var_name] = {}
            for idx, col_idx in col_info.items():
                lb, ub = self._get_col_bounds(col_idx)
                self._original_bounds[var_name][idx] = (lb, ub)
        else:
            lb, ub = self._get_col_bounds(col_info)
            self._original_bounds[var_name] = (lb, ub)

    # ------------------------------------------------------------------
    # Internal: solver-specific dispatch
    # ------------------------------------------------------------------

    def _set_solver_options(self):
        """Apply solver options to the native model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            m.setOptionValue('output_flag', self._log)
            if self._time_limit:
                m.setOptionValue('time_limit', self._time_limit)
            if self._thread_count:
                m.setOptionValue('threads', self._thread_count)
            if self._absolute_gap:
                m.setOptionValue('mip_abs_gap', self._absolute_gap)
            if self._relative_gap:
                m.setOptionValue('mip_rel_gap', self._relative_gap)
            for k, v in self._solver_options.items():
                m.setOptionValue(k, v)

        elif iface == 'gurobi':
            m.setParam('OutputFlag', 1 if self._log else 0)
            if self._time_limit:
                m.setParam('TimeLimit', self._time_limit)
            if self._thread_count:
                m.setParam('Threads', self._thread_count)
            if self._absolute_gap:
                m.setParam('MIPGap', self._absolute_gap)
            if self._relative_gap:
                m.setParam('MIPGap', self._relative_gap)
            for k, v in self._solver_options.items():
                m.setParam(k, v)

        elif iface == 'cplex':
            if self._time_limit:
                m.parameters.timelimit.set(self._time_limit)
            if self._thread_count:
                m.parameters.threads = self._thread_count
            if self._relative_gap:
                m.parameters.mip.tolerances.mipgap.set(self._relative_gap)
            for k, v in self._solver_options.items():
                setattr(m.parameters, k, v)

        elif iface == 'ortools':
            pass  # OR-Tools options set at Solve() time

        elif iface == 'copt':
            import coptpy
            if self._time_limit:
                m.setParam(coptpy.COPT.Param.TimeLimit, self._time_limit)
            if self._thread_count:
                m.setParam(coptpy.COPT.Param.Threads, self._thread_count)
            for k, v in self._solver_options.items():
                m.setParam(k, v)

        elif iface == 'pulp':
            pass  # PuLP solver options are passed at solve() time

        elif iface == 'pyomo':
            pass  # Pyomo solver options are passed at solve() time

        elif iface == 'scip':
            pass  # SCIP options set differently

        else:
            # Generic fallback: try setting common options
            pass

    def _add_all_constraints(self):
        """Add all constraints from features to the native model."""
        m = self._get_native_model()
        iface = self._interface
        constraints = self._features.get('constraints', [])
        labels = self._features.get('constraint_labels', [])

        if iface == 'highs':
            for constraint, label in zip(constraints, labels):
                try:
                    m.addConstr(constraint, name=label)
                except Exception:
                    try:
                        idxs, vals = constraint.unique_elements()
                        lb = constraint.bounds[0]
                        ub = constraint.bounds[1]
                        row_idx = m.numConstrs
                        m.addRow(lb, ub, len(idxs), list(idxs), list(vals))
                        if label:
                            m.passRowName(row_idx, label)
                    except Exception:
                        pass

        elif iface == 'gurobi':
            for constraint, label in zip(constraints, labels):
                try:
                    # gurobipy rejects name=None ('NoneType' object has
                    # no attribute 'encode'); unnamed constraints would
                    # be silently dropped, leaving a model with no rows.
                    # Mirror the solution generator: only pass a name
                    # when the label is set.
                    if label is not None:
                        m.addConstr(constraint, name=str(label))
                    else:
                        m.addConstr(constraint)
                except Exception:
                    pass
            m.update()

        elif iface == 'cplex':
            if constraints:
                m.add_constraints(constraints, names=labels if labels else None)

        elif iface == 'ortools':
            from ..generators.solution.ortools_solution_generator import (
                _materialize_proxy)
            real_vars = _materialize_proxy(self._features)
            for constraint, label in zip(constraints, labels):
                try:
                    constraint.evaluate(real_vars)
                except Exception:
                    pass

        elif iface == 'copt':
            for constraint, label in zip(constraints, labels):
                try:
                    # Same None-label hazard as gurobi: unnamed
                    # constraints must be added without a name or they
                    # would be silently dropped.
                    if label is not None:
                        m.addConstr(constraint, name=str(label))
                    else:
                        m.addConstr(constraint)
                except Exception:
                    pass

        elif iface == 'pulp':
            for constraint, label in zip(constraints, labels):
                try:
                    m += (constraint, label)
                except Exception:
                    try:
                        m += constraint
                    except Exception:
                        pass

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                if not hasattr(m, '_auto_constraints'):
                    m._auto_constraints = pyo.ConstraintList()
                for constraint, label in zip(constraints, labels):
                    try:
                        m._auto_constraints.add(constraint)
                    except Exception:
                        pass
            except Exception:
                pass

        elif iface == 'xpress':
            for constraint, label in zip(constraints, labels):
                if label is not None:
                    try:
                        constraint.name = str(label)
                    except Exception:
                        pass
                m.addConstraint(constraint)

        else:
            # Generic fallback: try addConstr
            for constraint, label in zip(constraints, labels):
                try:
                    m.addConstr(constraint, name=label)
                except Exception:
                    try:
                        m += constraint
                    except Exception:
                        pass

    def _set_objective(self):
        """Set the objective on the native model."""
        m = self._get_native_model()
        iface = self._interface
        objectives = self._features.get('objectives', [])
        directions = self._features.get('directions', [])
        obj_idx = self._obj_index

        if not objectives:
            return

        obj = objectives[obj_idx] if obj_idx < len(objectives) else objectives[0]
        direction = directions[obj_idx] if obj_idx < len(directions) else 'min'

        if iface == 'highs':
            # Store theta cost before minimize (which resets all column costs)
            theta_cost = None
            theta_idx = None
            if hasattr(self, '_var_col_map') and '_benders_theta' in self._var_col_map:
                theta_idx = self._var_col_map['_benders_theta']
                theta_cost = m.getLp().col_cost_[theta_idx]

            if direction == 'max':
                m.maximize(obj)
            else:
                m.minimize(obj)

            # Restore theta cost after minimize
            if theta_cost is not None:
                m.changeColCost(theta_idx, theta_cost)

            self._cached_obj_expr = obj
            self._cached_direction = direction

        elif iface == 'gurobi':
            import gurobipy
            sense = gurobipy.GRB.MAXIMIZE if direction == 'max' else gurobipy.GRB.MINIMIZE
            m.setObjective(obj, sense)
            m.update()

        elif iface == 'cplex':
            sense = 'max' if direction == 'max' else 'min'
            m.set_objective(sense, obj)

        elif iface == 'ortools':
            pass  # OR-Tools sets objective at Solve() time

        elif iface == 'copt':
            import coptpy
            sense = coptpy.COPT.MAXIMIZE if direction == 'max' else coptpy.COPT.MINIMIZE
            m.setObjective(obj, sense)

        elif iface == 'pulp':
            try:
                m.objective = None
                if direction == 'max':
                    m += -obj
                else:
                    m += obj
            except Exception:
                pass

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                for obj_comp in list(m.component_objects(pyo.Objective, active=True)):
                    m.del_component(obj_comp)
                sense = pyo.minimize if direction == 'min' else pyo.maximize
                m.obj = pyo.Objective(expr=obj, sense=sense)
            except Exception:
                pass

        elif iface == 'xpress':
            try:
                import xpress as xpress_interface
                sense = (xpress_interface.maximize if direction == 'max'
                         else xpress_interface.minimize)
                m.setObjective(obj, sense=sense)
            except Exception:
                try:
                    m.setObjective(obj)
                except Exception:
                    pass

        else:
            # Generic fallback
            try:
                if direction == 'max':
                    m.maximize(obj)
                else:
                    m.minimize(obj)
            except Exception:
                try:
                    m.setObjective(obj)
                except Exception:
                    pass

    def _solve_native(self):
        """Solve the native model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            # Use run() directly — this preserves column costs set via
            # changeColCost (e.g. for theta variable added after build).
            # minimize() would re-create the expression from the feloopy
            # objective and overwrite any new column costs.
            m.run()

        elif iface == 'gurobi':
            m.optimize()

        elif iface == 'cplex':
            m.solve()

        elif iface == 'ortools':
            # OR-Tools: need to re-materialize and call Solve()
            pass

        elif iface == 'copt':
            m.solve()

        elif iface == 'pulp':
            from ..generators.pulp_compat import cbc_solver_class
            solver_name = self._solver_name if self._solver_name else 'cbc'
            solver_cls = cbc_solver_class()
            self._pulp_stats = m.solve(solver_cls(msg=0))

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                solver_name = self._solver_name if self._solver_name else 'glpk'
                solver = pyo.SolverFactory(solver_name)
                solver.solve(m, tee=False)
            except Exception:
                pass

        elif iface == 'xpress':
            m.optimize()

        else:
            try:
                m.run()
            except Exception:
                try:
                    m.optimize()
                except Exception:
                    try:
                        m.solve()
                    except Exception:
                        pass

    def _get_native_model(self):
        """Return the native solver model object."""
        return self._features.get('model_object_before_solve',
                                  self._features.get('model_object',
                                                     getattr(self._m, 'model', None)))

    def _get_native_result(self):
        """Return the result from the last solve."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            return m.getModelStatus()
        elif iface == 'gurobi':
            return m.status
        elif iface == 'cplex':
            try:
                return m.solve_details.status
            except Exception:
                try:
                    return m.cplex.solution.get_status()
                except Exception:
                    return None
        elif iface == 'copt':
            # COPT only exposes a numeric solver status (verified:
            # 1 = optimal, 2 = infeasible).  Map the two cases that
            # matter to text tokens so `_master_status_dead`'s
            # substring check sees them; anything else stays opaque.
            try:
                status = int(m.status)
            except Exception:
                return None
            if status == 1:
                return 'optimal'
            if status == 2:
                return 'infeasible'
            return None
        else:
            return None

    def _get_objective_native(self):
        """Get objective value from the native model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            try:
                _, val = m.getInfoValue('objective_function_value')
                return val
            except Exception:
                return None
        elif iface == 'gurobi':
            try:
                import gurobipy as gp
                if m.status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                                    gp.GRB.LOADED):
                    return None
                return m.objVal
            except Exception:
                return None
        elif iface == 'cplex':
            try:
                return m.objective_value
            except Exception:
                try:
                    return m.cplex.solution.get_objective_value()
                except Exception:
                    return None
        elif iface == 'copt':
            try:
                return m.objval
            except Exception:
                return None
        elif iface == 'pulp':
            try:
                import pulp
                return pulp.value(m.objective)
            except Exception:
                return None
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                return pyo.value(m.obj)
            except Exception:
                return None
        elif iface == 'xpress':
            try:
                return m.attributes.objval
            except Exception:
                return None
        else:
            return None

    def _get_solution_values(self):
        """Get column values from the native solution."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            try:
                return list(m.getSolution().col_value)
            except Exception:
                return None
        elif iface == 'gurobi':
            try:
                return [v.X for v in m.getVars()]
            except Exception:
                return None
        elif iface == 'cplex':
            try:
                return list(m.cplex.solution.get_values())
            except Exception:
                try:
                    return list(m.solution.get_values())
                except Exception:
                    return None
        elif iface == 'copt':
            try:
                return [v.X for v in m.getVars()]
            except Exception:
                return None
        elif iface == 'pulp':
            try:
                return [v.varValue for v in m.variables()]
            except Exception:
                return None
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                return [v.value for v in m.component_data_objects(pyo.Var, active=True)]
            except Exception:
                return None
        elif iface == 'xpress':
            try:
                return list(m.getSolution())
            except Exception:
                return None
        else:
            return None

    def _get_duals_native(self):
        """Get row dual values from the native solution.

        Returns None when the solve status is not optimal or when the model
        contains integer variables (MIP duals are LP-relaxation duals and
        may be unreliable).  Also returns None when the solver does not
        expose dual information.
        """
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            try:
                import highspy
                status = m.getModelStatus()
                if status not in (highspy.HighsModelStatus.kOptimal,
                                  highspy.HighsModelStatus.kWarning):
                    return None
                return list(m.getSolution().row_dual)
            except Exception:
                return None

        elif iface == 'gurobi':
            try:
                import gurobipy as gp
                if m.status != gp.GRB.OPTIMAL:
                    return None
                duals = [c.Pi for c in m.getConstrs()]
                return duals if duals else None
            except Exception:
                return None

        elif iface == 'cplex':
            try:
                if m.cplex.solution.get_status() != 1:  # 1 = optimal
                    return None
                return list(m.cplex.solution.get_dual_values())
            except Exception:
                try:
                    return list(m.solution.get_dual_values())
                except Exception:
                    return None

        elif iface == 'copt':
            try:
                import coptpy
                if m.status != coptpy.COPT.OPTIMAL:
                    return None
                return [c.Pi for c in m.getConstrs()]
            except Exception:
                return None

        elif iface == 'pulp':
            try:
                from ..generators.pulp_compat import PULP_V4
                if PULP_V4:
                    stats = getattr(self, '_pulp_stats', None)
                    if stats is None or stats.status != 1:
                        return None
                    return [c.pi for c in m.constraints()]
                if m.status != 1:  # LpStatusOptimal
                    return None
                return [c.pi for c in m.constraints.values()]
            except Exception:
                return None

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                duals = []
                has_dual_suffix = hasattr(m, 'dual') and m.dual is not None
                if not has_dual_suffix:
                    return None
                for c in m.component_data_objects(pyo.Constraint, active=True):
                    try:
                        duals.append(m.dual[c])
                    except Exception:
                        return None
                return duals if duals else None
            except Exception:
                return None
        elif iface == 'xpress':
            try:
                solstatus = str(m.attributes.solstatus).split('.')[-1].lower()
                if solstatus != 'optimal':
                    return None
                duals = m.getDuals()
                if isinstance(duals, dict):
                    return [duals[k] for k in sorted(duals)]
                return list(duals) if duals else None
            except Exception:
                return None
        else:
            return None

    def _get_reduced_costs_native(self):
        """Get column reduced costs from the native solution.

        Returns None when the solve status is not optimal or when the
        solver does not expose reduced-cost information.
        """
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            try:
                import highspy
                status = m.getModelStatus()
                if status not in (highspy.HighsModelStatus.kOptimal,
                                  highspy.HighsModelStatus.kWarning):
                    return None
                return list(m.getSolution().col_dual)
            except Exception:
                return None
        elif iface == 'gurobi':
            try:
                import gurobipy as gp
                if m.status != gp.GRB.OPTIMAL:
                    return None
                return [v.RC for v in m.getVars()]
            except Exception:
                return None
        elif iface == 'cplex':
            try:
                if m.cplex.solution.get_status() != 1:
                    return None
                return list(m.cplex.solution.advanced.get_reduced_costs())
            except Exception:
                return None
        elif iface == 'copt':
            try:
                import coptpy
                if m.status != coptpy.COPT.OPTIMAL:
                    return None
                return [v.RC for v in m.getVars()]
            except Exception:
                return None
        elif iface == 'pulp':
            try:
                from ..generators.pulp_compat import PULP_V4
                if PULP_V4:
                    stats = getattr(self, '_pulp_stats', None)
                    if stats is None or stats.status != 1:
                        return None
                elif m.status != 1:
                    return None
                return [v.dj for v in m.variables()]
            except Exception:
                return None
        else:
            return None

    def _get_num_rows_native(self):
        """Get the number of rows in the native model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            return m.getNumRow()
        elif iface == 'gurobi':
            return m.numConstrs
        elif iface == 'cplex':
            return m.cplex.linear_constraints.get_num()
        elif iface == 'copt':
            return len(m.getConstrs())
        elif iface == 'pulp':
            from ..generators.pulp_compat import constraints_collection
            return len(constraints_collection(m))
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                return sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
            except Exception:
                return 0
        else:
            return 0

    def _get_num_cols_native(self):
        """Get the number of columns in the native model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            return m.getNumCol()
        elif iface == 'gurobi':
            return m.numVars
        elif iface == 'cplex':
            return m.cplex.variables.get_num()
        elif iface == 'copt':
            return len(m.getVars())
        elif iface == 'pulp':
            from ..generators.pulp_compat import problem_variables
            return len(problem_variables(m))
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                return sum(1 for _ in m.component_data_objects(pyo.Var, active=True))
            except Exception:
                return 0
        else:
            return 0

    def _get_col_bounds(self, col_idx):
        """Get the bounds of a column."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            _, cost, lb, ub, integrality = m.getCol(col_idx)
            return lb, ub
        elif iface == 'gurobi':
            v = m.getVars()[col_idx]
            return v.LB, v.UB
        elif iface == 'cplex':
            lb = m.cplex.variables.get_lower_bounds(col_idx)
            ub = m.cplex.variables.get_upper_bounds(col_idx)
            return lb, ub
        elif iface == 'copt':
            v = m.getVars()[col_idx]
            return v.lb, v.ub
        elif iface == 'pulp':
            vars_list = list(m.variables())
            if col_idx < len(vars_list):
                v = vars_list[col_idx]
                return v.lowBound if v.lowBound is not None else -1e20, v.upBound if v.upBound is not None else 1e20
            return -1e20, 1e20
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                vars_list = list(m.component_data_objects(pyo.Var, active=True))
                if col_idx < len(vars_list):
                    v = vars_list[col_idx]
                    lb = v.lb if v.lb is not None else -1e20
                    ub = v.ub if v.ub is not None else 1e20
                    return lb, ub
            except Exception:
                pass
            return -1e20, 1e20
        else:
            return -1e20, 1e20

    def _set_col_bounds(self, col_idx, lb, ub):
        """Set the bounds of a column."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            m.changeColBounds(col_idx, lb, ub)
        elif iface == 'gurobi':
            v = m.getVars()[col_idx]
            v.LB = lb
            v.UB = ub
            m.update()
        elif iface == 'cplex':
            m.cplex.variables.set_lower_bounds(col_idx, lb)
            m.cplex.variables.set_upper_bounds(col_idx, ub)
        elif iface == 'copt':
            v = m.getVars()[col_idx]
            v.lb = lb
            v.ub = ub
            m.update()
        elif iface == 'pulp':
            vars_list = list(m.variables())
            if col_idx < len(vars_list):
                v = vars_list[col_idx]
                v.lowBound = lb
                v.upBound = ub
        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                vars_list = list(m.component_data_objects(pyo.Var, active=True))
                if col_idx < len(vars_list):
                    v = vars_list[col_idx]
                    v.setlb(lb)
                    v.setub(ub)
            except Exception:
                pass
        else:
            try:
                m.changeColBounds(col_idx, lb, ub)
            except Exception:
                pass

    def _add_row_native(self, lower, upper, indices, values, label=None):
        """Add a row to the native solver model."""
        m = self._get_native_model()
        iface = self._interface

        if iface == 'highs':
            row_idx = m.numConstrs
            m.addRow(lower, upper, len(indices), list(indices), list(values))
            if label:
                m.passRowName(row_idx, label)

        elif iface == 'gurobi':
            import gurobipy
            row_expr = gurobipy.LinExpr(values, [m.getVars()[i] for i in indices])
            m.addConstr(row_expr >= lower, name=label or '')
            if upper < 1e20:
                m.addConstr(row_expr <= upper, name=(label or '') + '_ub')
            m.update()

        elif iface == 'cplex':
            import cplex
            row = cplex.SparsePair(ind=list(indices), val=list(values))
            if lower > -1e20 and upper < 1e20 and abs(lower - upper) < 1e-12:
                sense = 'E'
                rhs = lower
            elif lower > -1e20 and upper >= 1e20:
                sense = 'G'
                rhs = lower
            elif lower <= -1e20 and upper < 1e20:
                sense = 'L'
                rhs = upper
            else:
                sense = 'G'
                rhs = lower
            m.cplex.linear_constraints.add(
                lin_expr=[row],
                senses=[sense],
                rhs=[rhs],
                names=[label] if label else [None])

        elif iface == 'copt':
            import coptpy
            m.update()
            expr = coptpy.LinExpr()
            all_vars = m.getVars()
            # coptpy's LinExpr.addTerms expects parallel coefficient/Var
            # *lists*; passing a scalar pair raises TypeError, which the
            # old blanket `except` swallowed — every term was silently
            # dropped and the row went in as `0 >= rhs`, making any cut
            # master infeasible.  `+=` accepts a scalar product directly,
            # and failures now surface to the decomposition fallback
            # instead of corrupting the model.
            for idx, val in zip(indices, values):
                expr += val * all_vars[idx]
            m.addConstr(expr >= lower, name=label or '')
            if upper < 1e20:
                m.addConstr(expr <= upper, name=(label or '') + '_ub')
            m.update()

        elif iface == 'pulp':
            import pulp
            expr = 0
            vars_list = list(m.variables())
            for idx, val in zip(indices, values):
                if idx < len(vars_list):
                    expr += val * vars_list[idx]
            if lower > -1e20 and upper < 1e20 and abs(lower - upper) < 1e-12:
                m += (expr == lower, label or '')
            elif lower > -1e20 and upper >= 1e20:
                m += (expr >= lower, label or '')
            elif lower <= -1e20 and upper < 1e20:
                m += (expr <= upper, label or '')
            else:
                m += (expr >= lower, label or '')

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                expr = 0
                vars_list = list(m.component_data_objects(pyo.Var, active=True))
                for idx, val in zip(indices, values):
                    if idx < len(vars_list):
                        expr += val * vars_list[idx]
                if not hasattr(m, '_incremental_constraints'):
                    m._incremental_constraints = pyo.ConstraintList()
                if lower > -1e20 and upper < 1e20 and abs(lower - upper) < 1e-12:
                    m._incremental_constraints.add(expr == lower)
                elif lower > -1e20 and upper >= 1e20:
                    m._incremental_constraints.add(expr >= lower)
                elif lower <= -1e20 and upper < 1e20:
                    m._incremental_constraints.add(expr <= upper)
                else:
                    m._incremental_constraints.add(expr >= lower)
            except Exception:
                pass

        else:
            try:
                m.addRow(lower, upper, len(indices), list(indices), list(values))
            except Exception:
                pass

    def _build_var_col_map(self):
        """Build mapping from variable name to column index."""
        mapping = {}
        m = self._get_native_model()
        iface = self._interface
        variables = self._features.get('variables', {})

        # Some interfaces store a dict-like container (pyomo's IndexedVar,
        # COPT's tupledict) rather than a plain dict; unwrap those locally
        # so every element gets a column index instead of being mistaken
        # for a scalar.  to_indexed_dict guards the attribute access —
        # COPT's Var proxies unknown attributes to the solver and raises
        # CoptError (escaping a bare hasattr).
        _variables = {}
        for _key, _value in variables.items():
            if not isinstance(_value, dict):
                _container = to_indexed_dict(_value)
                if _container is not None:
                    _value = _container
            _variables[_key] = _value
        variables = _variables

        if iface == 'highs':
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            col_idx = v.index if hasattr(v, 'index') else v._col
                            mapping[name][idx] = col_idx
                        except Exception:
                            pass
                else:
                    try:
                        col_idx = var_obj.index if hasattr(var_obj, 'index') else var_obj._col
                        mapping[name] = col_idx
                    except Exception:
                        pass

        elif iface == 'gurobi':
            try:
                var_names = [v.VarName for v in m.getVars()]
            except Exception:
                var_names = []
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            vname = v.VarName if hasattr(v, 'VarName') else str(v)
                            if vname in var_names:
                                mapping[name][idx] = var_names.index(vname)
                        except Exception:
                            pass
                else:
                    try:
                        vname = var_obj.VarName if hasattr(var_obj, 'VarName') else str(var_obj)
                        if vname in var_names:
                            mapping[name] = var_names.index(vname)
                    except Exception:
                        pass

        elif iface == 'cplex':
            try:
                var_names = list(m.cplex.variables.get_names())
            except Exception:
                var_names = []
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            vname = str(v)
                            if vname in var_names:
                                mapping[name][idx] = var_names.index(vname)
                        except Exception:
                            pass
                else:
                    try:
                        vname = str(var_obj)
                        if vname in var_names:
                            mapping[name] = var_names.index(vname)
                    except Exception:
                        pass

        elif iface == 'copt':
            try:
                var_names = [v.getName() for v in m.getVars()]
            except Exception:
                var_names = []
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            vname = v.getName() if hasattr(v, 'getName') else str(v)
                            if vname in var_names:
                                mapping[name][idx] = var_names.index(vname)
                        except Exception:
                            pass
                else:
                    try:
                        vname = var_obj.getName() if hasattr(var_obj, 'getName') else str(var_obj)
                        if vname in var_names:
                            mapping[name] = var_names.index(vname)
                    except Exception:
                        pass

        elif iface == 'pulp':
            try:
                var_names = [v.name for v in m.variables()]
            except Exception:
                var_names = []
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            vname = v.name if hasattr(v, 'name') else str(v)
                            if vname in var_names:
                                mapping[name][idx] = var_names.index(vname)
                        except Exception:
                            pass
                else:
                    try:
                        vname = var_obj.name if hasattr(var_obj, 'name') else str(var_obj)
                        if vname in var_names:
                            mapping[name] = var_names.index(vname)
                    except Exception:
                        pass

        elif iface == 'pyomo':
            try:
                import pyomo.environ as pyo
                var_names = [v.name for v in m.component_data_objects(pyo.Var, active=True)]
            except Exception:
                var_names = []
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            vname = v.name if hasattr(v, 'name') else str(v)
                            if vname in var_names:
                                mapping[name][idx] = var_names.index(vname)
                        except Exception:
                            pass
                else:
                    try:
                        vname = var_obj.name if hasattr(var_obj, 'name') else str(var_obj)
                        if vname in var_names:
                            mapping[name] = var_names.index(vname)
                    except Exception:
                        pass

        else:
            # Generic fallback: try index attribute
            for (prefix, name), var_obj in variables.items():
                if isinstance(var_obj, dict):
                    mapping[name] = {}
                    for idx, v in var_obj.items():
                        try:
                            col_idx = v.index if hasattr(v, 'index') else v._col
                            mapping[name][idx] = col_idx
                        except Exception:
                            pass
                else:
                    try:
                        col_idx = var_obj.index if hasattr(var_obj, 'index') else var_obj._col
                        mapping[name] = col_idx
                    except Exception:
                        pass

        return mapping
