# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import timeit
import logging

logging.getLogger('pyomo').setLevel(logging.CRITICAL)


def generate_solution(features):

    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    time_limit = features['time_limit']
    objective_id = features['objective_being_optimized']
    log = features['log']
    solver_options = features['solver_options']

    try:
        import pyscipopt as scip
        from pyscipopt.scip import ExprCons, Expr
    except ImportError:
        raise ImportError(
            "The 'pyscipopt' package is required for SCIP interface. "
            "Install with: pip install PySCIPOpt"
        )

    model_object = features.get('model_object_before_solve')
    if model_object is not None and isinstance(model_object, scip.Model):
        scip_model = model_object
        needs_remap = False
    else:
        scip_model = scip.Model()
        needs_remap = True

    if not log:
        scip_model.setParam('display/verblevel', 0)
    variables = features.get('variables', {})
    variable_bound = features.get('variable_bound', {})

    if needs_remap:
        scip_var_map = {}
        for key, v in variables.items():
            base_name = key[1] if isinstance(key, tuple) else key

            if isinstance(key, tuple):
                vtype_str = key[0].lower()
            elif hasattr(v, 'isBinary') and v.isBinary():
                vtype_str = 'binary'
            elif hasattr(v, 'isIntegral') and v.isIntegral():
                vtype_str = 'integer'
            else:
                vtype_str = 'continuous'

            def _add_scip_var(vname, orig_v):
                if vname in variable_bound:
                    vb = variable_bound[vname]
                    lb = vb[0] if vb[0] is not None else -1e20
                    ub = vb[1] if vb[1] is not None else 1e20
                elif hasattr(orig_v, 'getLbOriginal'):
                    try:
                        lb = float(orig_v.getLbOriginal())
                        ub = float(orig_v.getUbOriginal())
                    except Exception:
                        lb, ub = -1e20, 1e20
                else:
                    lb, ub = -1e20, 1e20

                if isinstance(lb, (int, float)):
                    lb = float(lb)
                else:
                    lb = -1e20
                if isinstance(ub, (int, float)):
                    ub = float(ub)
                else:
                    ub = 1e20

                if 'binary' in vtype_str or 'bvar' in vtype_str or vtype_str == 'b':
                    return scip_model.addVar(vname, vtype='BINARY', lb=0, ub=1)
                elif 'integer' in vtype_str or 'ivar' in vtype_str or vtype_str == 'i' or vtype_str == 'int':
                    return scip_model.addVar(vname, vtype='INTEGER', lb=lb, ub=ub)
                else:
                    return scip_model.addVar(vname, vtype='CONTINUOUS', lb=lb, ub=ub)

            if isinstance(v, dict):
                for idx, orig_v in v.items():
                    if isinstance(idx, tuple):
                        indexed_name = f"{base_name}[{','.join(str(k) for k in idx)}]"
                    else:
                        indexed_name = f"{base_name}[{idx}]"
                    scip_var_map[indexed_name] = _add_scip_var(indexed_name, orig_v)
                    if hasattr(orig_v, 'name'):
                        scip_var_map[orig_v.name] = scip_var_map[indexed_name]
            else:
                var_name = base_name
                if hasattr(v, 'name'):
                    var_name = v.name
                scip_var_map[var_name] = _add_scip_var(var_name, v)
                if hasattr(v, 'name') and v.name != var_name:
                    scip_var_map[v.name] = scip_var_map[var_name]
    else:
        scip_var_map = {}

    objective_expr = model_objectives[objective_id]
    direction = directions[objective_id]

    obj_remap = _remap_scip_expr(objective_expr, scip_var_map)
    try:
        if direction == 'max':
            scip_model.setObjective(obj_remap, sense='maximize')
        else:
            scip_model.setObjective(obj_remap, sense='minimize')
    except (ValueError, TypeError):
        obj_aux = scip_model.addVar("obj_aux", vtype='CONTINUOUS', lb=-1e20, ub=1e20)
        scip_model.addCons(obj_aux == obj_remap, name="obj_linearization")
        if direction == 'max':
            scip_model.setObjective(obj_aux, sense='maximize')
        else:
            scip_model.setObjective(obj_aux, sense='minimize')

    for i, constraint in enumerate(model_constraints):
        label = constraint_labels[i] if i < len(constraint_labels) and constraint_labels[i] is not None else f"con_{i}"

        try:
            remapped = _remap_constraint(constraint, scip_var_map)
            if isinstance(remapped, tuple):
                for r in remapped:
                    scip_model.addCons(r, name=label)
            else:
                scip_model.addCons(remapped, name=label)
        except Exception as e:
            if log:
                print(f"Warning: could not add constraint '{label}': {e}")

    if time_limit is not None:
        scip_model.setParam('limits/time', time_limit)

    for key, value_opt in solver_options.items():
        if key.startswith("---"):
            continue
        if value_opt is None:
            continue
        try:
            scip_model.setParam(key, value_opt)
        except Exception:
            pass

    if features.get('_auto_lin_active', False):
        scip_model.setParam('presolving/maxrounds', 0)

    from ..init_generator import flush_init
    flush_init(features, force=True)

    time_solve_begin = timeit.default_timer()
    scip_model.optimize()
    time_solve_end = timeit.default_timer()

    import io, contextlib
    dual_cache = {}
    slack_cache = {}
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        for c in scip_model.getConss():
            cname = c.name
            try:
                if c.isLinear():
                    dual_cache[cname] = scip_model.getDualSolVal(c)
                    slack_cache[cname] = scip_model.getSlack(c)
                else:
                    dual_cache[cname] = 0.0
                    slack_cache[cname] = 0.0
            except Exception:
                dual_cache[cname] = 0.0
                slack_cache[cname] = 0.0

    features['_scip_dual_cache'] = dual_cache
    features['_scip_slack_cache'] = slack_cache

    return scip_model, [time_solve_begin, time_solve_end]


def _remap_constraint(constraint, var_map):
    from pyscipopt.scip import ExprCons

    if isinstance(constraint, ExprCons):
        new_expr = _remap_scip_expr(constraint.expr, var_map)
        lhs = constraint._lhs
        rhs = constraint._rhs
        if lhs is not None and rhs is not None:
            if abs(lhs - rhs) < 1e-10:
                return new_expr == lhs
            else:
                return (new_expr >= lhs, new_expr <= rhs)
        elif lhs is not None:
            return new_expr >= lhs
        elif rhs is not None:
            return new_expr <= rhs
        return new_expr

    return _remap_scip_expr(constraint, var_map)


def _remap_scip_expr(expr, var_map):
    from pyscipopt.scip import Variable, Expr, UnaryExpr, SumExpr, ProdExpr

    if isinstance(expr, (int, float)):
        return float(expr)

    if isinstance(expr, Variable):
        if expr.name in var_map:
            return var_map[expr.name]
        return expr

    if isinstance(expr, UnaryExpr):
        op = expr.getOp()
        child = _remap_scip_expr(expr.children[0], var_map)
        if op == 'log':
            from pyscipopt.scip import log as scip_fn
        elif op == 'exp':
            from pyscipopt.scip import exp as scip_fn
        elif op == 'sin':
            from pyscipopt.scip import sin as scip_fn
        elif op == 'cos':
            from pyscipopt.scip import cos as scip_fn
        elif op == 'sqrt':
            from pyscipopt.scip import sqrt as scip_fn
        elif op == 'tan':
            from pyscipopt.scip import tan as scip_fn
        else:
            return expr
        return scip_fn(child)

    if isinstance(expr, SumExpr):
        result = Expr()
        result = result + expr.constant
        for coef, child in zip(expr.coefs, expr.children):
            result = result + coef * _remap_scip_expr(child, var_map)
        return result

    if isinstance(expr, ProdExpr):
        children = expr.children
        left = _remap_scip_expr(children[0], var_map)
        if len(children) > 1:
            right = _remap_scip_expr(children[1], var_map)
            return left * right
        return left

    if isinstance(expr, Expr):
        result = 0
        for term, coeff in expr.terms.items():
            vt = term.vartuple
            if len(vt) == 1:
                vname = vt[0].name
                if vname in var_map:
                    result = result + coeff * var_map[vname]
                else:
                    result = result + coeff * vt[0]
            elif len(vt) == 2:
                v0, v1 = vt
                if v0.name in var_map and v1.name in var_map:
                    result = result + coeff * var_map[v0.name] * var_map[v1.name]
                elif v0.name == v1.name and v0.name in var_map:
                    result = result + coeff * var_map[v0.name] * var_map[v0.name]
                else:
                    result = result + coeff * v0 * v1
            else:
                tmp = coeff
                for v in vt:
                    if v.name in var_map:
                        tmp = tmp * var_map[v.name]
                    else:
                        tmp = tmp * v
                result = result + tmp
        return result

    tname = type(expr).__name__
    if tname == 'VarExpr' and hasattr(expr, 'children'):
        child = expr.children[0] if expr.children else expr
        return _remap_scip_expr(child, var_map)

    return expr
