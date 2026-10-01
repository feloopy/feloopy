# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Irreducible Infeasible Subsystem (IIS) finder.

Single home for all IIS discovery logic:

- Native solver IIS: HiGHS, Gurobi, CPLEX, Xpress
- Additive/deletion filter over LP data (``_find_iis_additive``)
- NLP/MINLP violation checks: Uno, Pyomo (bonmin/couenne), SCIP
- CP satisfaction checks: OR-Tools CP, CPLEX CP, Picat

The model class keeps thin ``find_iis`` / ``get_iis`` wrappers that
construct an ``IISFinder`` bound to the model instance.

Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

__all__ = ["IISFinder"]


class IISFinder:
    """Find IIS / violated constraints for a solved (or LP-exported) model.

    Parameters
    ----------
    owner:
        A ``model`` instance (or duck-type with ``features``, ``model``,
        ``solution`` attributes).
    """

    def __init__(self, owner):
        self._owner = owner
        self.features = owner.features
        self.model = owner.model
        self.solution = getattr(owner, "solution", None)

    @property
    def constraint_labels(self):
        return list(self.features.get("constraint_labels") or [])

    @property
    def constraints(self):
        return self.features.get("constraints") or []

    @property
    def interface(self):
        return self.features.get("interface_name", "")

    @staticmethod
    def _label(labels, i, n=None):
        if i < len(labels) and labels[i] is not None:
            return labels[i]
        return f"con_{i}"

    @classmethod
    def _all_labels(cls, labels, n_constraints):
        return [cls._label(labels, i) for i in range(n_constraints)]

    def _make_iis_test_model(self):
        """Build a fresh HiGHS model used by additive/LP IIS probes."""
        from ..feloopy import model
        return model(name="iis_test", method="exact", interface="highs")


    def get_iis(self):

        interface = self.features.get('interface_name', '')

        try:
            import highspy
            if isinstance(self.model, highspy.Highs):
                status, iis_result = self.model.getIis()
                if iis_result.valid_ and iis_result.row_index_:
                    output = ""
                    lp = self.model.getLp()
                    for idx in iis_result.row_index_:
                        if idx < lp.num_row_:
                            name = lp.row_names_[idx] if lp.row_names_ and idx < len(lp.row_names_) else str(idx)
                            output += f"con: {name}\n"
                    if output:
                        return "Infeasibility is caused by:\n" + output.rstrip("\n")
        except Exception:
            pass

        if 'lp_data' not in self.features and interface not in ['highs']:
            iis_labels = self.find_iis()
            if iis_labels:
                output = ""
                for label in iis_labels:
                    output += f"con: {label}\n"
                return "Infeasibility is caused by:\n" + output.rstrip("\n")
            else:
                return "Could not identify critical constraints."

        import highspy as highs_interface

        lp = self.features['lp_data']
        n_cols = lp['n_cols']
        n_rows = lp['n_rows']

        if n_rows == 0 or n_cols == 0:
            return "No constraints or variables in the model."

        direction = self.features['directions'][self.features['objective_being_optimized']]

        col_lower = lp['col_lower']
        col_upper = lp['col_upper']
        col_cost = lp['col_cost']
        row_lower = lp['row_lower']
        row_upper = lp['row_upper']
        integrality = lp['integrality']
        col_names = lp['col_names'] if lp['col_names'] else [f"x{i}" for i in range(n_cols)]
        row_names = lp['row_names'] if lp['row_names'] else [f"row_{i}" for i in range(n_rows)]
        A_col_pointers = lp['A_col_pointers']
        A_row_indices = lp['A_row_indices']
        A_values = lp['A_values']

        output = ""
        for excluded_row in range(n_rows):
            m = self._make_iis_test_model()

            new_vars = []
            for j in range(n_cols):
                lb = col_lower[j] if col_lower[j] > -1e20 else None
                ub = col_upper[j] if col_upper[j] < 1e20 else None
                if integrality[j] == highs_interface.HighsVarType.kInteger:
                    v = m.ivar(name=col_names[j], bound=[lb, ub])
                elif ub is not None and ub <= 1 and lb == 0:
                    v = m.bvar(name=col_names[j], bound=[lb, ub])
                else:
                    v = m.pvar(name=col_names[j], bound=[lb, ub])
                new_vars.append(v)

            for row in range(n_rows):
                if row == excluded_row:
                    continue

                row_cols = []
                row_vals = []
                for c in range(n_cols):
                    for idx in range(A_col_pointers[c], A_col_pointers[c + 1]):
                        if A_row_indices[idx] == row:
                            row_cols.append(c)
                            row_vals.append(A_values[idx])

                if not row_cols:
                    continue

                lhs = row_vals[0] * new_vars[row_cols[0]]
                for idx in range(1, len(row_cols)):
                    lhs = lhs + row_vals[idx] * new_vars[row_cols[idx]]

                lb = row_lower[row]
                ub = row_upper[row]

                if lb > -1e20 and ub < 1e20 and abs(lb - ub) > 1e-10:
                    m.con(lhs >= lb, name=f"{row_names[row]}_lb")
                    m.con(lhs <= ub, name=f"{row_names[row]}_ub")
                elif lb > -1e20 and ub >= 1e20 - 1:
                    m.con(lhs >= lb, name=row_names[row])
                elif ub < 1e20 and lb <= -1e20 + 1:
                    m.con(lhs <= ub, name=row_names[row])
                elif abs(lb - ub) <= 1e-10:
                    m.con(lhs == lb, name=row_names[row])

            obj_expr = None
            for j in range(n_cols):
                if abs(col_cost[j]) > 1e-10:
                    if obj_expr is None:
                        obj_expr = col_cost[j] * new_vars[j]
                    else:
                        obj_expr = obj_expr + col_cost[j] * new_vars[j]
            if obj_expr is None:
                obj_expr = 0

            m.obj(obj_expr, direction=direction)

            try:
                m.sol(directions=[direction], solver="highs")
                status = m.get_status()
            except Exception:
                status = "error"

            is_feasible = False
            if isinstance(status, str):
                s = status.lower()
                is_feasible = ('optimal' in s or 'feasible' in s) and 'infeasible' not in s

            if is_feasible:
                output += f"con: {row_names[excluded_row]}\n"

        if output:
            output = "Infeasibility is caused by:\n" + output
        else:
            output = "Could not identify critical constraints."

        return output.rstrip("\n")

    def find_iis(self, callback=None):
        labels = list(self.features.get('constraint_labels') or [])
        if not labels:
            return []

        interface = self.features.get('interface_name', '')

        try:
            import highspy
            if isinstance(self.model, highspy.Highs):
                status, iis_result = self.model.getIis()
                if iis_result.valid_ and iis_result.row_index_:
                    row_names = []
                    lp = self.model.getLp()
                    for idx in iis_result.row_index_:
                        if idx < lp.num_row_:
                            name = lp.row_names_[idx] if lp.row_names_ and idx < len(lp.row_names_) else str(idx)
                            row_names.append(name)
                    if row_names:
                        return row_names
        except Exception:
            pass

        if 'lp_data' in self.features:
            return self._find_iis_additive(callback)

        if interface == 'gurobi':
            return self._find_iis_gurobi()

        if interface == 'cplex':
            return self._find_iis_cplex()

        if interface == 'xpress':
            return self._find_iis_xpress()

        if interface in ['uno', 'scip', 'bonmin', 'couenne']:
            return self._find_iis_nlp()

        if interface in ['ortools_cp', 'cplex_cp', 'picat']:
            return self._find_iis_cp()

        return []

    def _find_iis_additive(self, callback=None):
        import highspy as highs_interface

        lp = self.features.get('lp_data')
        if not lp:
            raise RuntimeError("No LP data available")

        n_cols = lp['n_cols']
        n_rows = lp['n_rows']
        if n_rows == 0 or n_cols == 0:
            return []

        direction = self.features['directions'][self.features['objective_being_optimized']]

        col_lower = lp['col_lower']
        col_upper = lp['col_upper']
        col_cost = lp['col_cost']
        row_lower = lp['row_lower']
        row_upper = lp['row_upper']
        integrality = lp['integrality']
        col_names = lp['col_names'] if lp['col_names'] else [f"x{i}" for i in range(n_cols)]
        row_names = lp['row_names'] if lp['row_names'] else [f"row_{i}" for i in range(n_rows)]
        A_col_pointers = lp['A_col_pointers']
        A_row_indices = lp['A_row_indices']
        A_values = lp['A_values']

        labels = [l for l in self.features['constraint_labels'] if l in set(row_names)]
        total = len(labels)
        tested = 0

        def _make_model():
            m_new = self._make_iis_test_model()
            vs = []
            for j in range(n_cols):
                lb = col_lower[j] if col_lower[j] > -1e20 else None
                ub = col_upper[j] if col_upper[j] < 1e20 else None
                if integrality[j] == highs_interface.HighsVarType.kInteger:
                    v = m_new.ivar(name=col_names[j], bound=[lb, ub])
                elif ub is not None and ub <= 1 and lb == 0:
                    v = m_new.bvar(name=col_names[j], bound=[lb, ub])
                else:
                    v = m_new.pvar(name=col_names[j], bound=[lb, ub])
                vs.append(v)
            obj = None
            for j in range(n_cols):
                if abs(col_cost[j]) > 1e-10:
                    if obj is None:
                        obj = col_cost[j] * vs[j]
                    else:
                        obj = obj + col_cost[j] * vs[j]
            if obj is None:
                obj = 0
            m_new.obj(obj, direction=direction)
            return m_new, vs

        def _add_constraint(m_obj, vs, row, rname):
            row_cols = []
            row_vals = []
            for c in range(n_cols):
                for idx in range(A_col_pointers[c], A_col_pointers[c + 1]):
                    if A_row_indices[idx] == row:
                        row_cols.append(c)
                        row_vals.append(A_values[idx])
            if not row_cols:
                return
            lhs = row_vals[0] * vs[row_cols[0]]
            for idx in range(1, len(row_cols)):
                lhs = lhs + row_vals[idx] * vs[row_cols[idx]]
            rlb = row_lower[row]
            rub = row_upper[row]
            if rlb > -1e20 and rub < 1e20 and abs(rlb - rub) > 1e-10:
                m_obj.con(lhs >= rlb, name=f"{rname}_lb")
                m_obj.con(lhs <= rub, name=f"{rname}_ub")
            elif rlb > -1e20 and rub >= 1e20 - 1:
                m_obj.con(lhs >= rlb, name=rname)
            elif rub < 1e20 and rlb <= -1e20 + 1:
                m_obj.con(lhs <= rub, name=rname)
            elif abs(rlb - rub) <= 1e-10:
                m_obj.con(lhs == rlb, name=rname)

        def _solve(m_obj):
            try:
                m_obj.sol(directions=[direction], solver="highs", show_log=False)
                status = m_obj.get_status()
            except Exception:
                status = "error"
            if isinstance(status, str):
                s = status.lower()
                return ('optimal' in s or 'feasible' in s) and 'infeasible' not in s
            return False

        m, vs = _make_model()
        infeasible_constraints = []
        added_rows = []

        for row in range(n_rows):
            if row >= len(row_names):
                continue
            rname = row_names[row]
            if rname not in set(labels):
                continue

            tested += 1
            if callback:
                callback(tested, total, len(labels) - len(infeasible_constraints), rname)

            _add_constraint(m, vs, row, rname)
            added_rows.append((row, rname))

            if not _solve(m):
                infeasible_constraints.append(rname)
                m, vs = _make_model()
                for arow, aname in added_rows:
                    if aname != rname:
                        _add_constraint(m, vs, arow, aname)
                added_rows = [(arow, aname) for arow, aname in added_rows if aname != rname]

        return infeasible_constraints

    def _find_iis_gurobi(self):
        """Find IIS using Gurobi's native computeIIS()."""
        model_object = self.model
        try:
            model_object.computeIIS()
        except Exception:
            return []

        iis_labels = []
        for c in model_object.getConstrs():
            if c.IISConstr:
                iis_labels.append(c.constrName)

        return iis_labels

    def _find_iis_cplex(self):
        """Find IIS using CPLEX's native compute_iis()."""
        model_object = self.model
        try:
            model_object.cplex.solution.advanced.compute_iis()
        except Exception:
            return []

        iis_labels = []
        for c in model_object.iter_linear_constraints():
            try:
                if model_object.cplex.solution.advanced.get_iis(c.index):
                    iis_labels.append(c.name)
            except Exception:
                pass

        return iis_labels

    def _find_iis_xpress(self):
        """Find IIS using Xpress's native iis()."""
        model_object = self.model
        try:
            model_object.iis()
        except Exception:
            return []

        iis_labels = []
        try:
            iis_rows = model_object.getIISRow()
            for idx in iis_rows:
                c = model_object.getConstraint(idx)
                iis_labels.append(c.name)
        except Exception:
            pass

        return iis_labels

    def _find_iis_nlp(self):
        """Find violated constraints for NLP/MINLP solvers (Uno, Bonmin, Couenne, SCIP) by evaluating constraints at the solution point."""
        labels = list(self.features['constraint_labels'])
        model_constraints = self.features['constraints']
        interface = self.features.get('interface_name', '')

        if not model_constraints:
            return []

        if interface == 'uno':
            return self._find_iis_uno(labels, model_constraints)
        elif interface in ['bonmin', 'couenne']:
            return self._find_iis_pyomo(labels, model_constraints)
        elif interface == 'scip':
            return self._find_iis_scip(labels, model_constraints)
        return []

    def _find_iis_uno(self, labels, model_constraints):
        """Evaluate Uno constraints at the solution point to find violations."""
        import numpy as np
        from ..generators.variable.uno_variable_generator import evaluate_expr

        if self.solution is None:
            return labels

        variables = self.features.get('variables', [])
        if not variables:
            return labels

        try:
            result = self.solution[0]
            primal = result.primal_solution
            x = np.array([primal[v.index] for v in variables], dtype=float)
        except Exception:
            return labels

        violated = []

        for i, constraint in enumerate(model_constraints):
            if i >= len(labels):
                break

            label = labels[i] if labels[i] is not None else f"con_{i}"

            if isinstance(constraint, tuple) and len(constraint) == 3:
                sense = constraint[0]
                lhs = constraint[1]
                rhs = constraint[2]

                try:
                    if isinstance(lhs, (int, float)):
                        lhs_val = float(lhs)
                    else:
                        lhs_val = evaluate_expr(lhs, x)

                    if isinstance(rhs, (int, float)):
                        rhs_val = float(rhs)
                    else:
                        rhs_val = evaluate_expr(rhs, x)

                    violation = 0.0
                    if sense == '<=':
                        violation = max(0.0, lhs_val - rhs_val - 1e-6)
                    elif sense == '>=':
                        violation = max(0.0, rhs_val - lhs_val - 1e-6)
                    elif sense == '==':
                        violation = abs(lhs_val - rhs_val) - 1e-6

                    if violation > 1e-6:
                        violated.append(label)
                except Exception:
                    pass

            elif isinstance(constraint, (int, float)):
                if abs(constraint) > 1e-6:
                    violated.append(label)

        return violated

    def _find_iis_pyomo(self, labels, model_constraints):
        """Find violated constraints for Pyomo-based MINLP by checking bounds of relational expressions against variable bounds."""
        if not model_constraints:
            return []

        violated = []

        variables = self.features.get('variables', [])
        var_by_name = {}
        pyomo_model = self.model
        if pyomo_model is not None:
            for v in pyomo_model.component_objects():
                if hasattr(v, 'values'):
                    for idx, var in v.items():
                        var_by_name[var.name] = var

        for i, con in enumerate(model_constraints):
            label = labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}"

            try:
                con_str = str(con)
                if '<=' in con_str or '>=' in con_str or '==' in con_str:
                    args = con.args
                    if len(args) == 2:
                        arg0, arg1 = args[0], args[1]

                        arg0_is_var = hasattr(arg0, 'lb') and hasattr(arg0, 'ub') and hasattr(arg0, 'name')
                        arg1_is_var = hasattr(arg1, 'lb') and hasattr(arg1, 'ub') and hasattr(arg1, 'name')

                        if '<=' in con_str:
                            if arg0_is_var:
                                vlb = arg0.lb if arg0.lb is not None else float('-inf')
                                vub = arg0.ub if arg0.ub is not None else float('inf')
                                if not isinstance(arg1, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        ub = pyomo_value(arg1)
                                    except Exception:
                                        continue
                                else:
                                    ub = arg1
                                if vlb > ub + 1e-6:
                                    violated.append(label)
                            elif arg1_is_var:
                                vlb = arg1.lb if arg1.lb is not None else float('-inf')
                                vub = arg1.ub if arg1.ub is not None else float('inf')
                                if not isinstance(arg0, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        lb = pyomo_value(arg0)
                                    except Exception:
                                        continue
                                else:
                                    lb = arg0
                                if vub < lb - 1e-6:
                                    violated.append(label)
                        elif '>=' in con_str:
                            if arg1_is_var:
                                vlb = arg1.lb if arg1.lb is not None else float('-inf')
                                vub = arg1.ub if arg1.ub is not None else float('inf')
                                if not isinstance(arg0, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        lb = pyomo_value(arg0)
                                    except Exception:
                                        continue
                                else:
                                    lb = arg0
                                if vub < lb - 1e-6:
                                    violated.append(label)
                            elif arg0_is_var:
                                vlb = arg0.lb if arg0.lb is not None else float('-inf')
                                vub = arg0.ub if arg0.ub is not None else float('inf')
                                if not isinstance(arg1, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        ub = pyomo_value(arg1)
                                    except Exception:
                                        continue
                                else:
                                    ub = arg1
                                if vlb > ub + 1e-6:
                                    violated.append(label)
                        elif '==' in con_str:
                            if arg1_is_var:
                                vlb = arg1.lb if arg1.lb is not None else float('-inf')
                                vub = arg1.ub if arg1.ub is not None else float('inf')
                                if not isinstance(arg0, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        eq_val = pyomo_value(arg0)
                                    except Exception:
                                        continue
                                else:
                                    eq_val = arg0
                                if eq_val < vlb - 1e-6 or eq_val > vub + 1e-6:
                                    violated.append(label)
                            elif arg0_is_var:
                                vlb = arg0.lb if arg0.lb is not None else float('-inf')
                                vub = arg0.ub if arg0.ub is not None else float('inf')
                                if not isinstance(arg1, (int, float)):
                                    try:
                                        from pyomo.environ import value as pyomo_value
                                        eq_val = pyomo_value(arg1)
                                    except Exception:
                                        continue
                                else:
                                    eq_val = arg1
                                if eq_val < vlb - 1e-6 or eq_val > vub + 1e-6:
                                    violated.append(label)
            except Exception:
                pass

        return violated

    def _find_iis_scip(self, labels, model_constraints):
        """Find violated constraints for SCIP by checking constraint satisfaction at solution."""
        if self.solution is None:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        scip_model = self.solution[0]
        if scip_model is None:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        violated = []
        try:
            status = scip_model.getStatus()
            if status.lower() in ['optimal', 'feasible']:
                return []
        except Exception:
            pass

        return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

    def _find_iis_cp(self):
        """Find violated constraints for CP solvers by checking constraint satisfaction."""
        labels = list(self.features['constraint_labels'])
        model_constraints = self.features['constraints']
        interface = self.features.get('interface_name', '')

        if not model_constraints:
            return []

        if self.solution is None:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        if interface == 'ortools_cp':
            return self._find_iis_ortools_cp(labels, model_constraints)
        elif interface == 'cplex_cp':
            return self._find_iis_cplex_cp(labels, model_constraints)
        elif interface == 'picat':
            return self._find_iis_picat(labels, model_constraints)
        return []

    def _find_iis_ortools_cp(self, labels, model_constraints):
        """Check OR-Tools CP constraint satisfaction at solution."""
        try:
            solver = self.solution[0][1]
        except Exception:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        violated = []
        for i, constraint in enumerate(model_constraints):
            label = labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}"

            try:
                if hasattr(constraint, 'LinearExprs'):
                    for expr in constraint.LinearExprs():
                        val = solver.Value(expr)
                        if val != 0:
                            violated.append(label)
                            break
            except Exception:
                pass

        return violated

    def _find_iis_cplex_cp(self, labels, model_constraints):
        """Check CPLEX CP constraint satisfaction at solution."""
        try:
            solution = self.solution[0]
        except Exception:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        violated = []
        for i, constraint in enumerate(model_constraints):
            label = labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}"

            try:
                if not solution.get_value(constraint):
                    violated.append(label)
            except Exception:
                pass

        return violated

    def _find_iis_picat(self, labels, model_constraints):
        """Check Picat constraint satisfaction at solution."""
        if self.solution is None:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]

        status = self.solution[0][0]
        if status != 0:
            return [labels[i] if i < len(labels) and labels[i] is not None else f"con_{i}" for i in range(len(model_constraints))]
        return []
