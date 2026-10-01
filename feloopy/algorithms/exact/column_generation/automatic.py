"""Automatic callback factories for column generation.

Contains all ``from_automatic*`` classmethods that build CG callbacks
from an ``_AutomaticCaptureModel``, plus internal helpers for native
model construction and Dantzig-Wolfe structure detection.
"""

import itertools
from collections import deque

from ....helpers.containers import as_var_group

from .core import ColumnGeneration


# Interfaces whose solve paths never touch global sys.stdout/stderr
# (no contextlib.redirect_* around the solve, no solver-side tee on the
# process streams).  Only closures on these interfaces are tagged
# ``feloopy_parallel_safe``: concurrent solves on gurobi/pyomo/scip
# interleave those global stream swaps and either deadlock (pyomo's
# tee) or leave output pointing at a dead buffer.
_PARALLEL_SAFE_INTERFACES = {'highs', 'xpress', 'copt'}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _import_model():
    from feloopy.feloopy import model as _model_cls
    return _model_cls


def _build_native(m, captured):
    native = {}
    for (kind, name), spec in captured.variables.items():
        creator = getattr(m, kind)
        value = creator(name=name, dim=spec['dim'], bound=spec['bound'])
        # dict / dict-like container (pyomo IndexedVar, COPT tupledict)
        # / scalar — as_var_group normalizes all three.
        values = as_var_group(value)
        for index, symbolic in spec['values'].items():
            native[symbolic.key] = values[index]
    return native


def _detect_dw_structure(captured, n_constraints, linking_constraints=None):
    """Detect blocks and linking constraints for Dantzig-Wolfe.

    Parameters
    ----------
    captured : _AutomaticCaptureModel
    n_constraints : int
    linking_constraints : list[int] | None
        User-specified linking constraint indices.  When *None* the
        method attempts automatic detection via connected components.

    Returns
    -------
    blocks : list[list[tuple]]
        Variable keys grouped by block.
    linking_indices : list[int]
        Indices of the linking (complicating) constraints.
    block_constraints : dict[int, list[int]]
        Block-local constraint indices keyed by block index.
    linking_rhs : dict[int, float]
        Right-hand-side values of the linking constraints.
    """
    var_to_constraints = {}
    for cidx in range(n_constraints):
        terms = captured.constraints[cidx][0].expression.terms
        for key in terms.keys():
            if key not in var_to_constraints:
                var_to_constraints[key] = set()
            var_to_constraints[key].add(cidx)

    all_var_keys = list(var_to_constraints.keys())
    if not all_var_keys:
        return [], [], {}, {}, {}

    if linking_constraints is not None:
        linking_indices = list(linking_constraints)
    else:
        def _count_components(exclude):
            a = {v: set() for v in all_var_keys}
            for cidx in range(n_constraints):
                if cidx in exclude:
                    continue
                terms = captured.constraints[cidx][0].expression.terms
                vl = list(terms.keys())
                for ai in range(len(vl)):
                    for bi in range(ai + 1, len(vl)):
                        a[vl[ai]].add(vl[bi])
                        a[vl[bi]].add(vl[ai])
            vis = set()
            n_comp = 0
            for v in all_var_keys:
                if v in vis:
                    continue
                n_comp += 1
                q = deque([v])
                while q:
                    u = q.popleft()
                    if u in vis:
                        continue
                    vis.add(u)
                    for w in a[u]:
                        if w not in vis:
                            q.append(w)
            return n_comp

        n_comp_full = _count_components(set())
        if n_comp_full > 1:
            linking_indices = []
        else:
            linking_indices = []
            for cidx in range(n_constraints):
                trial = set(linking_indices) | {cidx}
                if _count_components(trial) > n_comp_full:
                    linking_indices.append(cidx)
                    n_comp_full = _count_components(set(linking_indices))
                    if n_comp_full > 1:
                        break
            if not linking_indices:
                raise ValueError(
                    "Cannot auto-detect blocks for Dantzig-Wolfe. "
                    "Specify linking_constraints.")

    linking_set = set(linking_indices)
    adj = {v: set() for v in all_var_keys}
    for cidx in range(n_constraints):
        if cidx in linking_set:
            continue
        terms = captured.constraints[cidx][0].expression.terms
        vlist = list(terms.keys())
        for a in range(len(vlist)):
            for b in range(a + 1, len(vlist)):
                adj[vlist[a]].add(vlist[b])
                adj[vlist[b]].add(vlist[a])

    visited = set()
    blocks = []
    for v in all_var_keys:
        if v in visited:
            continue
        comp = []
        queue = deque([v])
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    queue.append(w)
        blocks.append(comp)

    var_to_block = {}
    for bi, bvars in enumerate(blocks):
        for vk in bvars:
            var_to_block[vk] = bi

    block_constraints = {k: [] for k in range(len(blocks))}
    for cidx in range(n_constraints):
        if cidx in linking_set:
            continue
        terms = captured.constraints[cidx][0].expression.terms
        vlist = list(terms.keys())
        if not vlist:
            continue
        bi = var_to_block.get(vlist[0], 0)
        block_constraints[bi].append(cidx)

    linking_rhs = {}
    linking_senses = {}
    for cidx in linking_indices:
        linking_rhs[cidx] = -captured.constraints[cidx][0].expression.constant
        linking_senses[cidx] = captured.constraints[cidx][0].sense

    return blocks, linking_indices, block_constraints, linking_rhs, linking_senses


def _sign_combos(n_blocks, budget=64):
    """Sign-combination seeds for Dantzig-Wolfe initial solutions.

    The full ``2**n_blocks`` enumeration is only safe for small block
    counts: at ``n_blocks`` = 23 it means 8.4M combinations, each
    materialised as a full variable->sign dict (plus one block-solve
    per block), which exhausts memory.  Keep the full product while it
    fits *budget*; otherwise return a deterministic corners +
    single-flip set of at most *budget* combinations.
    """
    if 2 ** n_blocks <= budget:
        return list(itertools.product([1.0, -1.0], repeat=n_blocks))
    combos = [[1.0] * n_blocks, [-1.0] * n_blocks]
    for i in range(n_blocks):
        if len(combos) >= budget:
            break
        combo = [1.0] * n_blocks
        combo[i] = -1.0
        combos.append(combo)
    return combos


def _extract_block_values(m, block_keys):
    """Read the solved values of *block_keys* from model *m*.

    One element lookup per key via ``get_var_at``: ``get_numpy_var``
    rebuilds the whole array by querying every index of the variable, so
    calling it once per name still re-extracted full arrays for indices the
    block never touches.  Keys that cannot be read resolve to 0.0, matching
    the original per-name ``try/except`` semantics.
    """
    x_k = {}
    for key in block_keys:
        kind, name, index = key
        try:
            val = m.get_var_at(name, index)
        except Exception:
            val = None
        try:
            if val is None:
                x_k[key] = 0.0
            elif index is None:
                x_k[key] = (float(val)
                            if not hasattr(val, '__len__') else 0.0)
            else:
                x_k[key] = float(val)
        except Exception:
            x_k[key] = 0.0
    return x_k


# ------------------------------------------------------------------
# Automatic callback factories
# ------------------------------------------------------------------

@classmethod
def from_automatic(cls, captured, em, interface, solver, directions,
                   method='exact',
                   time_limit=None, cpu_threads=None,
                   absolute_gap=None, relative_gap=None):
    """Build standard CG callbacks from an _AutomaticCaptureModel.

    The master solver builds the full LP, solves it, and extracts
    constraint duals. The pricing oracle uses duals to compute
    reduced costs for candidate columns.
    Returns (master_solver, pricing_oracle) callable pair.
    """
    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _model = _import_model()

    n_constraints = len(captured.constraints)
    constraint_senses = [c.sense for c, _ in captured.constraints]
    constraint_labels = []
    for idx, (_, label) in enumerate(captured.constraints):
        constraint_labels.append(label if label is not None else f'_auto_cg_con_{idx}')

    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant
    # obj() may carry no explicit direction: fall back to the search
    # directions rather than assuming one.
    _eff_dir = obj_direction or (directions[0] if directions else None)

    def build_master(columns):
        _m = _model(method=method, name='_cg_master', interface=_if)
        # Do NOT pre-seed features['directions']: obj() below appends its
        # own direction slot, and a pre-seeded list leaves a stray None
        # slot, so sol() sees more direction slots than objective_labels
        # entries and raises IndexError.
        native = _build_native(_m, captured)

        for cidx in range(n_constraints):
            label = constraint_labels[cidx]
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            expr = constant
            for key, coefficient in terms.items():
                expr = expr + coefficient * native[key]
            sense = constraint_senses[cidx]
            if sense == '<=':
                _m.con(expr <= 0, name=label)
            elif sense == '>=':
                _m.con(expr >= 0, name=label)
            else:
                _m.con(expr == 0, name=label)

        translated = obj_constant
        for key, coefficient in obj_terms.items():
            translated = translated + coefficient * native[key]
        if isinstance(translated, (int, float)):
            dummy = _m.fvar('_cg_dummy', bound=[0, 0])
            _m.obj(translated + 0 * dummy, direction=_eff_dir, label=obj_label)
        else:
            _m.obj(translated, direction=_eff_dir, label=obj_label)

        _m.sol(directions=directions, solver=_sv,
               time_limit=time_limit, cpu_threads=cpu_threads,
               absolute_gap=absolute_gap, relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = float('inf') if _eff_dir == 'min' else float('-inf')

        constraint_duals = {}
        for cidx in range(n_constraints):
            label = constraint_labels[cidx]
            try:
                d = _m.get_dual(label)
                if d is not None:
                    constraint_duals[cidx] = float(d)
            except Exception:
                pass

        solution = {}
        for (kind, name) in captured.variables:
            try:
                solution[name] = _m.get_numpy_var(name)
            except Exception:
                pass

        return {
            'objective': float(obj_val),
            'duals': constraint_duals,
            'solution': solution,
        }

    def pricing_oracle(duals):
        if not duals:
            return None

        _m = _model(method=method, name='_cg_pricing', interface=_if)
        native = _build_native(_m, captured)

        reduced_cost_expr = 0.0
        for key, coefficient in obj_terms.items():
            if key in native:
                reduced_cost_expr = reduced_cost_expr + coefficient * native[key]

        for cidx, pi_j in duals.items():
            if abs(pi_j) < 1e-12:
                continue
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            sense = constraint_senses[cidx]
            sign = 1.0 if sense == '<=' else -1.0
            for key, a_ij in terms.items():
                if key in native:
                    reduced_cost_expr = reduced_cost_expr + sign * pi_j * a_ij * native[key]

        if isinstance(reduced_cost_expr, (int, float)):
            return None

        dummy = _m.fvar('_pricing_dummy', bound=[0, 0])
        _m.obj(reduced_cost_expr + 0 * dummy, direction='min', label='pricing')

        _m.sol(directions=['min'], solver=_sv, time_limit=time_limit,
               cpu_threads=cpu_threads)

        obj_val = _m.get_obj()
        if obj_val is None:
            return None
        reduced = float(obj_val)
        if reduced >= -1e-6:
            return None
        return {'name': f'col_{len(duals)}', 'reduced_cost': reduced}

    return build_master, pricing_oracle


@classmethod
def from_automatic_incremental(cls, captured, em, interface, solver, directions,
                               method='exact',
                               time_limit=None, cpu_threads=None,
                               absolute_gap=None, relative_gap=None):
    """Build incremental CG using IncrementalModel.

    Returns (master_inc, add_column_fn, pricing_oracle) triple.
    The IncrementalModel is built once; columns are added incrementally
    via add_column_fn without rebuilding the model.
    """
    from feloopy.classes.incremental import IncrementalModel

    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _model = _import_model()

    n_constraints = len(captured.constraints)
    constraint_senses = [c.sense for c, _ in captured.constraints]
    constraint_labels = []
    for idx, (_, label) in enumerate(captured.constraints):
        constraint_labels.append(label if label is not None else f'_auto_cg_con_{idx}')

    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant

    _m = _model(method=method, name='_cg_master', interface=_if)
    _m.features['directions'] = list(directions)
    native = _build_native(_m, captured)

    for cidx in range(n_constraints):
        label = constraint_labels[cidx]
        constant = captured.constraints[cidx][0].expression.constant
        terms = captured.constraints[cidx][0].expression.terms
        expr = constant
        for key, coefficient in terms.items():
            expr = expr + coefficient * native[key]
        sense = constraint_senses[cidx]
        if sense == '<=':
            _m.con(expr <= 0, name=label)
        elif sense == '>=':
            _m.con(expr >= 0, name=label)
        else:
            _m.con(expr == 0, name=label)

    # The objective is identical on every constraint iteration: translate
    # and register it once.  Rebuilding it inside the loop created
    # O(constraints x obj_terms) highspy expression objects (hundreds of
    # thousands of Python operator calls), which dominated cg setup time.
    translated = obj_constant
    for key, coefficient in obj_terms.items():
        translated = translated + coefficient * native[key]
    if isinstance(translated, (int, float)):
        dummy = _m.fvar('_cg_dummy', bound=[0, 0])
        _m.obj(translated + 0 * dummy, direction=obj_direction, label=obj_label)
    else:
        _m.obj(translated, direction=obj_direction, label=obj_label)

    master_inc = IncrementalModel(
        _m,
        directions=directions,
        solver_name=_sv,
        time_limit=time_limit,
        thread_count=cpu_threads,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        log=False,
    )
    master_inc.build()

    def add_column_fn(inc, col_spec):
        cost = col_spec.get('cost', 0.0)
        lb = col_spec.get('lb', 0.0)
        ub = col_spec.get('ub', 1e20)
        is_integer = col_spec.get('is_integer', False)
        row_indices = col_spec.get('row_indices', [])
        row_values = col_spec.get('row_values', [])
        name = col_spec.get('name', None)
        inc.add_column(cost, row_indices, row_values, lb=lb, ub=ub,
                       name=name, is_integer=is_integer)

    def pricing_oracle(duals):
        if not duals:
            return None

        _mp = _model(method=method, name='_cg_pricing', interface=_if)
        pricing_native = _build_native(_mp, captured)

        reduced_cost_expr = 0.0
        for key, coefficient in obj_terms.items():
            if key in pricing_native:
                reduced_cost_expr = reduced_cost_expr + coefficient * pricing_native[key]

        for cidx, pi_j in duals.items():
            if abs(pi_j) < 1e-12:
                continue
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            sense = constraint_senses[cidx]
            sign = 1.0 if sense == '<=' else -1.0
            for key, a_ij in terms.items():
                if key in pricing_native:
                    reduced_cost_expr = reduced_cost_expr + sign * pi_j * a_ij * pricing_native[key]

        if isinstance(reduced_cost_expr, (int, float)):
            return None

        dummy = _mp.fvar('_pricing_dummy', bound=[0, 0])
        _mp.obj(reduced_cost_expr + 0 * dummy, direction='min', label='pricing')
        _mp.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                cpu_threads=cpu_threads)

        obj_val = _mp.get_obj()
        if obj_val is None:
            return None
        reduced = float(obj_val)
        if reduced >= -1e-6:
            return None

        # one extraction per variable name: the accessor re-reads the
        # whole array, and names repeat across constraint terms
        _name_vals = {}

        row_indices = list(range(n_constraints))
        row_values = [0.0] * n_constraints
        for cidx in range(n_constraints):
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            for key, a_ij in terms.items():
                if key in pricing_native:
                    try:
                        if key[1] not in _name_vals:
                            _name_vals[key[1]] = _mp.get_numpy_var(key[1])
                        val = _name_vals[key[1]]
                        if val is not None:
                            if isinstance(val, dict):
                                idx_val = val.get(key[2], 0.0)
                            elif key[2] is None:
                                idx_val = float(val) if not hasattr(val, '__len__') else 0.0
                            else:
                                idx_val = 0.0
                            row_values[cidx] += a_ij * idx_val
                    except Exception:
                        pass

        column_cost = 0.0
        for key, coefficient in obj_terms.items():
            if key in pricing_native:
                try:
                    if key[1] not in _name_vals:
                        _name_vals[key[1]] = _mp.get_numpy_var(key[1])
                    val = _name_vals[key[1]]
                    if val is not None:
                        if isinstance(val, dict):
                            idx_val = val.get(key[2], 0.0)
                        elif key[2] is None:
                            idx_val = float(val) if not hasattr(val, '__len__') else 0.0
                        else:
                            idx_val = 0.0
                        column_cost += coefficient * idx_val
                except Exception:
                    pass

        return {
            'name': f'col_{len(duals)}',
            'reduced_cost': reduced,
            'cost': column_cost,
            'lb': 0.0,
            'ub': 1e20,
            'is_integer': False,
            'row_indices': row_indices,
            'row_values': row_values,
        }

    return master_inc, add_column_fn, pricing_oracle


@classmethod
def from_automatic_ccg(cls, captured, em, interface, solver, directions,
                       method='exact',
                       time_limit=None, cpu_threads=None,
                       absolute_gap=None, relative_gap=None):
    """Build C&CG callbacks. Returns (master_solver, pricing_oracle) pair.

    Column-and-Constraint Generation:
    1. Master solves the full problem (all variables free).
    2. Pricing oracle fixes complicating variables to the master's
       solution and solves the subproblem to find worst-case
       constraint violations.
    3. If a violation is found, a robustifying cut is added to the
       master that eliminates the current fixing.
    """
    from feloopy.classes.incremental import IncrementalModel

    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _model = _import_model()
    cut_count = [0]
    _last_solution = [None]

    n_constraints = len(captured.constraints)
    constraint_senses = [c.sense for c, _ in captured.constraints]
    constraint_labels = []
    for idx, (_, label) in enumerate(captured.constraints):
        constraint_labels.append(label if label is not None else f'_auto_ccg_con_{idx}')

    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant

    _m = _model(method=method, name='_ccg_master', interface=_if)
    _m.features['directions'] = list(directions)
    native = _build_native(_m, captured)

    for cidx in range(n_constraints):
        label = constraint_labels[cidx]
        constant = captured.constraints[cidx][0].expression.constant
        terms = captured.constraints[cidx][0].expression.terms
        expr = constant
        for key, coefficient in terms.items():
            expr = expr + coefficient * native[key]
        sense = constraint_senses[cidx]
        if sense == '<=':
            _m.con(expr <= 0, name=label)
        elif sense == '>=':
            _m.con(expr >= 0, name=label)
        else:
            _m.con(expr == 0, name=label)

    translated = obj_constant
    for key, coefficient in obj_terms.items():
        translated = translated + coefficient * native[key]
    if isinstance(translated, (int, float)):
        dummy = _m.fvar('_ccg_dummy', bound=[0, 0])
        _m.obj(translated + 0 * dummy, direction=obj_direction, label=obj_label)
    else:
        _m.obj(translated, direction=obj_direction, label=obj_label)

    master_inc = IncrementalModel(
        _m,
        directions=directions,
        solver_name=_sv,
        time_limit=time_limit,
        thread_count=cpu_threads,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        log=False,
    )
    master_inc.build()

    var_col_map = master_inc.var_col_map

    def build_master(columns):
        status, _ = master_inc.solve()
        obj_val = master_inc.get_objective_value()
        if obj_val is None:
            obj_val = float('inf') if obj_direction == 'min' else float('-inf')

        constraint_duals = {}
        try:
            duals_list = master_inc.get_row_duals()
            row_count = master_inc.get_num_rows()
        except Exception:
            duals_list, row_count = None, 0
        if duals_list:
            for cidx in range(min(row_count, len(duals_list))):
                if abs(duals_list[cidx]) > 1e-12:
                    constraint_duals[cidx] = float(duals_list[cidx])

        solution = master_inc.get_all_variable_values()
        _last_solution[0] = solution
        return {
            'objective': float(obj_val),
            'duals': constraint_duals,
            'solution': solution,
        }

    def pricing_oracle(duals):
        master_solution = _last_solution[0]
        if master_solution is None:
            return None

        # Fast path: when the master fixed every variable appearing in a
        # <= / >= constraint, the "worst-case violation" is one
        # deterministic evaluation of the same signed sums the
        # fixed-point LP would return (equality terms excluded exactly as
        # in the model path below).  Evaluating directly skips building,
        # fixing and solving a throwaway pricing model every iteration —
        # that build+fix+solve was ccg's entire overhead above a direct
        # solve.  Any unfixed term falls back to the original solve.
        def _sol_value(key):
            v = master_solution.get(key[1])
            if isinstance(v, dict):
                idx = key[2] if len(key) >= 3 else None
                if idx is None or idx not in v:
                    return None
                v = v[idx]
            elif len(key) >= 3 and key[2] is not None:
                return None
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        deterministic = True
        violation_val = 0.0
        for cidx in range(n_constraints):
            sense = constraint_senses[cidx]
            if sense == '=':
                continue
            expression = captured.constraints[cidx][0].expression
            val = float(expression.constant)
            for key, coefficient in expression.terms.items():
                sv = _sol_value(key)
                if sv is None:
                    deterministic = False
                    break
                val += coefficient * sv
            if not deterministic:
                break
            violation_val += val if sense == '<=' else -val

        if deterministic:
            if violation_val <= 1e-6:
                return None
        else:
            _mp = _model(method=method, name='_ccg_pricing', interface=_if)
            pricing_native = _build_native(_mp, captured)

            for (kind, name) in captured.variables:
                if name in pricing_native and name in master_solution:
                    val = master_solution[name]
                    if isinstance(val, dict):
                        for idx, v in val.items():
                            sym = pricing_native.get(name)
                            if isinstance(sym, dict) and idx in sym:
                                try:
                                    _mp.fix(sym[idx], v)
                                except Exception:
                                    # model has no fix() API — pin the
                                    # variable with an equality row instead
                                    try:
                                        _mp.con(sym[idx] == v)
                                    except Exception:
                                        pass
                    else:
                        sym = pricing_native.get(name)
                        if sym is not None and not isinstance(sym, dict):
                            try:
                                _mp.fix(sym, float(val))
                            except Exception:
                                try:
                                    _mp.con(sym == float(val))
                                except Exception:
                                    pass

            violation_expr = 0.0
            for cidx in range(n_constraints):
                sense = constraint_senses[cidx]
                constant = captured.constraints[cidx][0].expression.constant
                terms = captured.constraints[cidx][0].expression.terms
                expr = constant
                for key, coefficient in terms.items():
                    if key in pricing_native:
                        expr = expr + coefficient * pricing_native[key]
                if sense == '<=':
                    violation_expr = violation_expr + expr
                elif sense == '>=':
                    violation_expr = violation_expr - expr

            if isinstance(violation_expr, (int, float)):
                return None

            _mp.obj(violation_expr, direction='max', label='ccg_violation')
            _mp.sol(directions=['max'], solver=_sv, time_limit=time_limit,
                    cpu_threads=cpu_threads)

            violation_val = _mp.get_obj()
            if violation_val is None or violation_val <= 1e-6:
                return None

        row_indices = []
        row_values = []
        for (kind, name) in captured.variables:
            if name in master_solution and name in var_col_map:
                val = master_solution[name]
                col_info = var_col_map[name]
                if isinstance(val, dict) and isinstance(col_info, dict):
                    for idx, v in val.items():
                        if idx in col_info and abs(v) > 1e-12:
                            row_indices.append(col_info[idx])
                            row_values.append(v)
                elif not isinstance(val, dict) and not isinstance(col_info, dict):
                    fv = float(val) if not hasattr(val, '__len__') else 0.0
                    if abs(fv) > 1e-12:
                        row_indices.append(col_info)
                        row_values.append(fv)

        if not row_indices:
            return None

        cut_rhs = sum(
            v * v for _, v in zip(row_indices, row_values)) - 1e-6
        master_inc.add_row(
            lower=-1e20, upper=cut_rhs,
            indices=row_indices, values=row_values,
            label=f'_ccg_cut_{cut_count[0]}')
        cut_count[0] += 1

        return {
            'name': f'ccg_col_{cut_count[0]}',
            'reduced_cost': float(violation_val),
        }

    return build_master, pricing_oracle


@classmethod
def from_automatic_dcg(cls, captured, em, interface, solver, sub_interface,
                       sub_solver, directions, method='exact',
                       time_limit=None, cpu_threads=None,
                       absolute_gap=None, relative_gap=None):
    """Build DCG callbacks. Returns (build_master, master_pricing, sub_pricing) triplet."""
    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _s_if = sub_interface if isinstance(sub_interface, str) else (
        sub_interface[0] if isinstance(sub_interface, (list, tuple)) else _if)
    _s_sv = sub_solver if isinstance(sub_solver, str) else (
        sub_solver[0] if isinstance(sub_solver, (list, tuple)) else _sv)
    _model = _import_model()
    master_columns = []
    sub_columns = []

    n_constraints = len(captured.constraints)
    constraint_senses = [c.sense for c, _ in captured.constraints]
    constraint_labels = []
    for idx, (_, label) in enumerate(captured.constraints):
        constraint_labels.append(label if label is not None else f'_auto_dcg_con_{idx}')

    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant

    int_var_names = set()
    for (kind, name) in captured.variables:
        if kind in ('ivar', 'bvar'):
            int_var_names.add(name)

    master_constraint_indices = []
    sub_constraint_indices = []
    for cidx in range(n_constraints):
        terms = captured.constraints[cidx][0].expression.terms
        has_int = any(k[1] in int_var_names for k in terms.keys())
        if has_int:
            master_constraint_indices.append(cidx)
        else:
            sub_constraint_indices.append(cidx)

    if not sub_constraint_indices and n_constraints > 1:
        mid = n_constraints // 2
        master_constraint_indices = list(range(mid))
        sub_constraint_indices = list(range(mid, n_constraints))
    elif not master_constraint_indices and n_constraints > 1:
        mid = n_constraints // 2
        master_constraint_indices = list(range(mid))
        sub_constraint_indices = list(range(mid, n_constraints))
    else:
        master_constraint_indices = list(range(n_constraints))

    def _build_master_model():
        _m = _model(method=method, name='_dcg_master', interface=_if)
        # Do NOT pre-seed features['directions']: obj() below appends its
        # own direction slot, and a pre-seeded list leaves a stray None
        # slot, so sol() sees more direction slots than objective_labels
        # entries and raises IndexError.
        native = _build_native(_m, captured)

        for cidx in master_constraint_indices:
            label = constraint_labels[cidx]
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            expr = constant
            for key, coefficient in terms.items():
                expr = expr + coefficient * native[key]
            sense = constraint_senses[cidx]
            if sense == '<=':
                _m.con(expr <= 0, name=label)
            elif sense == '>=':
                _m.con(expr >= 0, name=label)
            else:
                _m.con(expr == 0, name=label)

        translated = obj_constant
        for key, coefficient in obj_terms.items():
            translated = translated + coefficient * native[key]
        _eff_dir = obj_direction or (directions[0] if directions else None)
        if isinstance(translated, (int, float)):
            dummy = _m.fvar('_dcg_master_dummy', bound=[0, 0])
            _m.obj(translated + 0 * dummy, direction=_eff_dir, label=obj_label)
        else:
            _m.obj(translated, direction=_eff_dir, label=obj_label)

        return _m, native

    def build_master(columns):
        master_columns.extend(columns)
        _m, native = _build_master_model()

        _m.sol(directions=directions, solver=_sv,
               time_limit=time_limit, cpu_threads=cpu_threads,
               absolute_gap=absolute_gap, relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = float('inf') if obj_direction == 'min' else float('-inf')

        constraint_duals = {}
        for cidx in range(n_constraints):
            label = constraint_labels[cidx]
            try:
                d = _m.get_dual(label)
                if d is not None:
                    constraint_duals[cidx] = float(d)
            except Exception:
                pass

        solution = {}
        for (kind, name) in captured.variables:
            try:
                solution[name] = _m.get_numpy_var(name)
            except Exception:
                pass

        return {
            'objective': float(obj_val),
            'duals': constraint_duals,
            'solution': solution,
        }

    def master_pricing(duals):
        _mp = _model(method=method, name='_dcg_master_pricing', interface=_if)
        pricing_native = _build_native(_mp, captured)

        reduced_cost_expr = 0.0
        for key, coefficient in obj_terms.items():
            if key in pricing_native:
                reduced_cost_expr = reduced_cost_expr + coefficient * pricing_native[key]

        for cidx, pi_j in duals.items():
            if abs(pi_j) < 1e-12:
                continue
            constant = captured.constraints[cidx][0].expression.constant
            terms = captured.constraints[cidx][0].expression.terms
            sense = constraint_senses[cidx]
            sign = 1.0 if sense == '<=' else -1.0
            for key, a_ij in terms.items():
                if key in pricing_native:
                    reduced_cost_expr = reduced_cost_expr + sign * pi_j * a_ij * pricing_native[key]

        if isinstance(reduced_cost_expr, (int, float)):
            if reduced_cost_expr >= -1e-6:
                return None
            return {'name': f'dcg_master_col_{len(master_columns)}', 'reduced_cost': float(reduced_cost_expr)}

        dummy = _mp.fvar('_dcg_mp_dummy', bound=[0, 0])
        _mp.obj(reduced_cost_expr + 0 * dummy, direction='min', label='pricing')
        _mp.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                cpu_threads=cpu_threads)

        obj_val = _mp.get_obj()
        if obj_val is None:
            return None
        reduced = float(obj_val)
        if reduced >= -1e-6:
            return None
        return {'name': f'dcg_master_col_{len(master_columns)}', 'reduced_cost': reduced}

    def sub_pricing(duals):
        _ms = _model(method=method, name='_dcg_sub_pricing', interface=_s_if)
        sub_native = _build_native(_ms, captured)

        translated = obj_constant
        for key, coefficient in obj_terms.items():
            if key in sub_native:
                translated = translated + coefficient * sub_native[key]
        if isinstance(translated, (int, float)):
            dummy = _ms.fvar('_dcg_sp_dummy', bound=[0, 0])
            _ms.obj(translated + 0 * dummy, direction='min', label='pricing')
        else:
            _ms.obj(translated, direction='min', label='pricing')

        for cidx in sub_constraint_indices:
            try:
                label = constraint_labels[cidx]
                constant = captured.constraints[cidx][0].expression.constant
                terms = captured.constraints[cidx][0].expression.terms
                expr_tr = constant
                for key, coefficient in terms.items():
                    expr_tr = expr_tr + coefficient * sub_native[key]
                sense = constraint_senses[cidx]
                if sense == '<=':
                    _ms.con(expr_tr <= 0, name=label)
                elif sense == '>=':
                    _ms.con(expr_tr >= 0, name=label)
                else:
                    _ms.con(expr_tr == 0, name=label)
            except Exception:
                pass

        _ms.sol(directions=['min'], solver=_s_sv, time_limit=time_limit,
                cpu_threads=cpu_threads)

        obj_val = _ms.get_obj()
        if obj_val is None:
            return None
        reduced = float(obj_val)
        if reduced >= -1e-6:
            return None
        return {'name': f'dcg_sub_col_{len(sub_columns)}', 'reduced_cost': reduced}

    if _if in _PARALLEL_SAFE_INTERFACES and _s_if in _PARALLEL_SAFE_INTERFACES:
        # Both pricing solves are stream-clean — the CG loop may run
        # them concurrently (see ColumnGeneration._run_dual_pricing).
        master_pricing.feloopy_parallel_safe = True
        sub_pricing.feloopy_parallel_safe = True

    return build_master, master_pricing, sub_pricing


# ------------------------------------------------------------------
# Dantzig-Wolfe decomposition
# ------------------------------------------------------------------

@classmethod
def from_automatic_dw(cls, captured, em, interface, solver, directions,
                      method='exact', time_limit=None, cpu_threads=None,
                      absolute_gap=None, relative_gap=None,
                      linking_constraints=None):
    """Dantzig-Wolfe decomposition – single convexity constraint (eq. 2.29-2.32).

    Master problem (weighting LP):
        min  Σ_s z_s u_s
        s.t. Σ_s r_{s,i} u_s = b_i   (linking constraints)
             Σ_s u_s = 1              (single convexity)
             u_s ≥ 0

    Subproblems (one per block k):
        min  Σ_{j∈block_k} (c_j − Σ_i λ_i a_{ij}) x_j
        s.t. block-k local constraints

    Convergence: v ≥ σ  where v = Σ_k v_k and σ is the convexity dual.

    Returns (build_master, pricing_oracle, initial_solutions).
    """
    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _model = _import_model()

    n_constraints = len(captured.constraints)
    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant

    blocks, linking_indices, block_constraints, linking_rhs, linking_senses = (
        _detect_dw_structure(captured, n_constraints,
                             linking_constraints))
    n_blocks = len(blocks)
    n_linking = len(linking_indices)

    _block_cache = {}

    def _get_block(block_idx, label_tag='sub'):
        """Build a block model once (vars + local constraints) and reuse it.

        Every pricing iteration and every seeding combo changes only the
        objective, so rebuilding per call re-declared all variable groups
        and re-added every constraint — the dominant cost of the seeding
        loop on large models.  Re-solves reset the objective bookkeeping
        first (``_reset_objective``) so ``obj()`` reproduces the state of a
        fresh single-objective model, and the ``_reusable_rows`` flag makes
        the solver reset the native row set instead of stacking a second
        copy of the constraints on top.
        """
        ck = (block_idx, label_tag)
        if ck in _block_cache:
            return _block_cache[ck]
        _mb = _model(method=method,
                     name=f'_dw_{label_tag}_{block_idx}',
                     interface=_if)
        native = {}
        group_values = {}
        for key in blocks[block_idx]:
            kind, name, index = key
            if (kind, name) not in captured.variables:
                continue
            spec = captured.variables[(kind, name)]
            # Materialise each variable group once: the creator returns
            # the whole group, so calling it again for a second key of
            # the same group re-declares the name (unique-name guard).
            if (kind, name) not in group_values:
                creator = getattr(_mb, kind)
                val = creator(name=name, dim=spec['dim'],
                              bound=spec['bound'])
                if isinstance(val, dict):
                    group_values[(kind, name)] = val
                else:
                    # dict / dict-like container (pyomo IndexedVar, COPT
                    # tupledict) / scalar — as_var_group normalizes all
                    # three with guarded attribute access.
                    group_values[(kind, name)] = as_var_group(val)
            native[key] = group_values[(kind, name)][index]

        for cidx in block_constraints[block_idx]:
            constraint = captured.constraints[cidx][0]
            expr = constraint.expression.constant
            for key, coeff in constraint.expression.terms.items():
                if key in native:
                    expr = expr + coeff * native[key]
            sense = constraint.sense
            if sense == '<=':
                _mb.con(expr <= 0)
            elif sense == '>=':
                _mb.con(expr >= 0)
            else:
                _mb.con(expr == 0)
        _mb.features['_reusable_rows'] = True
        entry = {'mb': _mb, 'native': native, 'dummy': None}
        _block_cache[ck] = entry
        return entry

    def _reset_objective(_mb):
        # obj() appends; a reused model must end in the exact state of a
        # fresh model with a single objective (objective_counter starts
        # at [0, 0] and obj() increments it to [1, 1]).
        _mb.features['directions'] = []
        _mb.features['objectives'] = []
        _mb.features['objective_labels'] = []
        _mb.features['objective_counter'] = [0, 0]

    def _block_dummy(entry):
        if entry['dummy'] is None:
            entry['dummy'] = entry['mb'].fvar('_dw_dummy', bound=[0, 0])
        return entry['dummy']

    def _pricing_objective(native, lambda_vals):
        obj_expr = 0.0
        for key, coeff in obj_terms.items():
            if key in native:
                modifier = 0.0
                for i, cidx in enumerate(linking_indices):
                    a_ij = (captured.constraints[cidx][0]
                            .expression.terms.get(key, 0.0))
                    modifier = modifier + lambda_vals[i] * a_ij
                obj_expr = obj_expr + (coeff - modifier) * native[key]
        return obj_expr

    def _solve_block(block_idx, lambda_vals, label_tag='sub'):
        entry = _get_block(block_idx, label_tag)
        _mb, native = entry['mb'], entry['native']
        obj_expr = _pricing_objective(native, lambda_vals)
        if isinstance(obj_expr, (int, float)) and obj_expr == 0.0:
            obj_expr = 0 * _block_dummy(entry)
        _reset_objective(_mb)
        _mb.obj(obj_expr, direction='min',
                label=f'_dw_{label_tag}_obj_{block_idx}')
        _mb.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                cpu_threads=cpu_threads, absolute_gap=absolute_gap,
                relative_gap=relative_gap)
        v_k = _mb.get_obj()
        if v_k is None:
            v_k = 0.0
        x_k = _extract_block_values(_mb, blocks[block_idx])
        return float(v_k), x_k

    def _eval_obj(block_solutions):
        val = obj_constant
        for key, coeff in obj_terms.items():
            for bk_x in block_solutions.values():
                if key in bk_x:
                    val += coeff * bk_x[key]
        return val

    def _eval_r(block_solutions):
        result = []
        for cidx in linking_indices:
            r_i = 0.0
            for key, coeff in (captured.constraints[cidx][0]
                               .expression.terms.items()):
                for bk_x in block_solutions.values():
                    if key in bk_x:
                        r_i += coeff * bk_x[key]
            result.append(r_i)
        return result

    def _solve_all_blocks(lambda_vals, label_tag='init'):
        block_solutions = {}
        total_v = 0.0
        for bk in range(n_blocks):
            v_k, x_k = _solve_block(bk, lambda_vals, label_tag)
            block_solutions[bk] = x_k
            total_v += v_k
        return block_solutions, total_v

    def _solve_all_blocks_with_obj(obj_coeffs, lambda_vals,
                                   label_tag='init'):
        block_solutions = {}
        total_v = 0.0
        for bk in range(n_blocks):
            entry = _get_block(bk, label_tag)
            _mb, native = entry['mb'], entry['native']
            obj_e = 0.0
            for key, coeff in obj_coeffs.items():
                if key in native:
                    modifier = 0.0
                    for i, cidx in enumerate(linking_indices):
                        a_ij = (captured.constraints[cidx][0]
                                .expression.terms.get(key, 0.0))
                        modifier += lambda_vals[i] * a_ij
                    obj_e += (coeff - modifier) * native[key]
            if isinstance(obj_e, (int, float)) and obj_e == 0.0:
                obj_e = 0 * _block_dummy(entry)
            _reset_objective(_mb)
            _mb.obj(obj_e, direction='min')
            _mb.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                    cpu_threads=cpu_threads, absolute_gap=absolute_gap,
                    relative_gap=relative_gap)
            v_k = _mb.get_obj()
            if v_k is None:
                v_k = 0.0
            total_v += float(v_k)
            x_k = _extract_block_values(_mb, blocks[bk])
            block_solutions[bk] = x_k
        return block_solutions, total_v

    sign_combos = _sign_combos(n_blocks)
    random_objectives = []
    for signs in sign_combos:
        obj = {}
        for bk_idx, bk_vars in enumerate(blocks):
            for key in bk_vars:
                obj[key] = signs[bk_idx]
        random_objectives.append(obj)

    initial_solutions = []
    for obj_coeffs in random_objectives:
        block_solutions, _ = _solve_all_blocks_with_obj(
            obj_coeffs, [0.0] * n_linking, label_tag='init')
        z_val = _eval_obj(block_solutions)
        r_val = _eval_r(block_solutions)
        initial_solutions.append({'z': z_val, 'r': r_val,
                                  'x': block_solutions})

    def build_master(solution_records):
        _m = _model(method=method, name='_dw_master', interface=_if)
        _m.features['directions'] = list(directions)
        u_syms = []
        z_vals = []
        r_vals = []
        for s, rec in enumerate(solution_records):
            u = _m.fvar(f'u_{s}', bound=[0, None])
            u_syms.append(u)
            z_vals.append(rec['z'])
            r_vals.append(rec['r'])

        obj = 0.0
        for s in range(len(solution_records)):
            obj = obj + z_vals[s] * u_syms[s]
        if isinstance(obj, (int, float)):
            dummy = _m.fvar('_dw_m_dummy', bound=[0, 0])
            _m.obj(obj + 0 * dummy, direction='min', label='_dw_obj')
        else:
            _m.obj(obj, direction='min', label='_dw_obj')

        used_labels = set()
        for i in range(n_linking):
            base = f'_dw_comp_{i}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            con_expr = 0.0
            for s in range(len(solution_records)):
                con_expr = con_expr + r_vals[s][i] * u_syms[s]
            cidx = linking_indices[i]
            csense = linking_senses[cidx]
            if csense == '<=':
                _m.con(con_expr <= linking_rhs[cidx], name=label)
            elif csense == '>=':
                _m.con(con_expr >= linking_rhs[cidx], name=label)
            else:
                _m.con(con_expr == linking_rhs[cidx], name=label)

        base_conv = '_dw_convexity'
        label_conv = base_conv
        cnt = 0
        while label_conv in used_labels:
            cnt += 1
            label_conv = f'{base_conv}_{cnt}'
        used_labels.add(label_conv)
        conv_expr = 0.0
        for s in range(len(solution_records)):
            conv_expr = conv_expr + u_syms[s]
        _m.con(conv_expr == 1, name=label_conv)

        _m.sol(directions=directions, solver=_sv, time_limit=time_limit,
               cpu_threads=cpu_threads, absolute_gap=absolute_gap,
               relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = (float('inf') if obj_direction == 'min'
                       else float('-inf'))

        duals = {}
        for i in range(n_linking):
            base = f'_dw_comp_{i}'
            label = base
            cnt = 0
            while True:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[i] = float(d)
                    break
                except Exception:
                    cnt += 1
                    label = f'{base}_{cnt}'
                    if cnt > 10:
                        break
        try:
            d_conv = _m.get_dual(label_conv)
            if d_conv is not None:
                duals[n_linking] = float(d_conv)
        except Exception:
            pass

        u_vals = {}
        for s in range(len(solution_records)):
            try:
                v = _m.get_numpy_var(f'u_{s}')
                u_vals[s] = (float(v) if not hasattr(v, '__len__')
                             else 0.0)
            except Exception:
                u_vals[s] = 0.0

        x_star = {k: {} for k in range(n_blocks)}
        for s, rec in enumerate(solution_records):
            w = u_vals.get(s, 0.0)
            if abs(w) < 1e-12:
                continue
            for bk, block_x in rec['x'].items():
                for key, val in block_x.items():
                    x_star[bk][key] = (x_star[bk].get(key, 0.0)
                                       + w * val)

        return {
            'objective': float(obj_val),
            'duals': duals,
            'solution': x_star,
        }

    def build_modified_master(solution_records):
        """Modified master with artificial variables for violated linking constraints.

        For <= constraints: linking_expr - art[i] <= b_i
        For >= constraints: linking_expr + art[i] >= b_i
        This allows art[i] >= 0 to absorb infeasibility.
        The objective includes a large penalty M * sum(art[i]) to drive
        artificial variables to zero as columns are added.
        """
        _m = _model(method=method, name='_dw_mod_master', interface=_if)
        _m.features['directions'] = list(directions)
        u_syms = []
        z_vals = []
        r_vals = []
        for s, rec in enumerate(solution_records):
            u = _m.fvar(f'u_{s}', bound=[0, None])
            u_syms.append(u)
            z_vals.append(rec['z'])
            r_vals.append(rec['r'])

        M = 1e6
        art = []
        for i in range(n_linking):
            a_i = _m.fvar(f'_art_{i}', bound=[0, None])
            art.append(a_i)

        obj = 0.0
        for s in range(len(solution_records)):
            obj = obj + z_vals[s] * u_syms[s]
        obj = obj + M * sum(art[i] for i in range(n_linking))
        if isinstance(obj, (int, float)):
            dummy = _m.fvar('_dw_mm_dummy', bound=[0, 0])
            _m.obj(obj + 0 * dummy, direction='min', label='_dw_obj')
        else:
            _m.obj(obj, direction='min', label='_dw_obj')

        used_labels = set()
        for i in range(n_linking):
            base = f'_dw_comp_{i}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            con_expr = 0.0
            for s in range(len(solution_records)):
                con_expr = con_expr + r_vals[s][i] * u_syms[s]
            cidx = linking_indices[i]
            csense = linking_senses[cidx]
            if csense == '<=':
                _m.con(con_expr - art[i] <= linking_rhs[cidx], name=label)
            elif csense == '>=':
                _m.con(con_expr + art[i] >= linking_rhs[cidx], name=label)
            else:
                _m.con(con_expr - art[i] <= linking_rhs[cidx], name=label + '_le')
                _m.con(con_expr + art[i] >= linking_rhs[cidx], name=label + '_ge')

        base_conv = '_dw_convexity'
        label_conv = base_conv
        cnt = 0
        while label_conv in used_labels:
            cnt += 1
            label_conv = f'{base_conv}_{cnt}'
        used_labels.add(label_conv)
        conv_expr = 0.0
        for s in range(len(solution_records)):
            conv_expr = conv_expr + u_syms[s]
        _m.con(conv_expr == 1, name=label_conv)

        _m.sol(directions=directions, solver=_sv, time_limit=time_limit,
               cpu_threads=cpu_threads, absolute_gap=absolute_gap,
               relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = (float('inf') if obj_direction == 'min'
                       else float('-inf'))

        duals = {}
        for i in range(n_linking):
            base = f'_dw_comp_{i}'
            label = base
            cnt = 0
            while True:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[i] = float(d)
                    break
                except Exception:
                    cnt += 1
                    label = f'{base}_{cnt}'
                    if cnt > 10:
                        break
        try:
            d_conv = _m.get_dual(label_conv)
            if d_conv is not None:
                duals[n_linking] = float(d_conv)
        except Exception:
            pass

        u_vals = {}
        for s in range(len(solution_records)):
            try:
                v = _m.get_numpy_var(f'u_{s}')
                u_vals[s] = (float(v) if not hasattr(v, '__len__')
                             else 0.0)
            except Exception:
                u_vals[s] = 0.0

        x_star = {k: {} for k in range(n_blocks)}
        for s, rec in enumerate(solution_records):
            w = u_vals.get(s, 0.0)
            if abs(w) < 1e-12:
                continue
            for bk, block_x in rec['x'].items():
                for key, val in block_x.items():
                    x_star[bk][key] = (x_star[bk].get(key, 0.0)
                                       + w * val)

        return {
            'objective': float(obj_val),
            'duals': duals,
            'solution': x_star,
        }

    def pricing_oracle(duals):
        lambda_vals = [duals.get(i, 0.0) for i in range(n_linking)]
        sigma = duals.get(n_linking, 0.0)

        total_v = 0.0
        new_x = {}
        for bk in range(n_blocks):
            v_k, x_k = _solve_block(bk, lambda_vals)
            total_v += v_k
            new_x[bk] = x_k

        if total_v >= sigma - 1e-6:
            return None

        z_new = _eval_obj(new_x)
        r_new = _eval_r(new_x)

        lambda_b = sum(lambda_vals[i] * linking_rhs[linking_indices[i]]
                       for i in range(n_linking))
        lower_bound = total_v + lambda_b

        return [{'z': z_new, 'r': r_new, 'x': new_x,
                 '_lower_bound': lower_bound}]

    return build_master, build_modified_master, pricing_oracle, initial_solutions


@classmethod
def from_automatic_dw2(cls, captured, em, interface, solver, directions,
                       method='exact', time_limit=None, cpu_threads=None,
                       absolute_gap=None, relative_gap=None,
                       linking_constraints=None):
    """Dantzig-Wolfe decomposition – per-block convexity (eq. 2.83-2.86).

    Master problem (per-block weighting):
        min  Σ_k Σ_s z_{s,k} u_{s,k}
        s.t. Σ_k Σ_s r_{s,k,i} u_{s,k} = b_i  (linking constraints)
             Σ_s u_{s,k} = 1                     (convexity per block k)
             u_{s,k} ≥ 0

    Subproblems are identical to ``from_automatic_dw``.

    Convergence: v_k ≥ σ_k for every block k.

    Returns (build_master, pricing_oracle, initial_solutions).
    """
    _if = interface if isinstance(interface, str) else (
        interface[0] if isinstance(interface, (list, tuple)) else 'highs')
    _sv = solver if isinstance(solver, str) else (
        solver[0] if isinstance(solver, (list, tuple)) else 'highs')
    _model = _import_model()

    n_constraints = len(captured.constraints)
    obj_expression, obj_direction, obj_label = captured.objectives[0]
    obj_terms = dict(obj_expression.terms)
    obj_constant = obj_expression.constant

    blocks, linking_indices, block_constraints, linking_rhs, linking_senses = (
        _detect_dw_structure(captured, n_constraints,
                             linking_constraints))
    n_blocks = len(blocks)
    n_linking = len(linking_indices)

    _block_cache = {}

    def _get_block(block_idx, label_tag='sub'):
        """Build a block model once (vars + local constraints) and reuse it.

        See from_automatic_dw._get_block: every pricing iteration and every
        seeding combo changes only the objective, so rebuilding per call
        re-declared all variable groups and re-added every constraint.
        Re-solves reset the objective bookkeeping first (``_reset_objective``)
        and the ``_reusable_rows`` flag makes the solver reset the native row
        set instead of stacking a second copy of the constraints.
        """
        ck = (block_idx, label_tag)
        if ck in _block_cache:
            return _block_cache[ck]
        _mb = _model(method=method,
                     name=f'_dw2_{label_tag}_{block_idx}',
                     interface=_if)
        native = {}
        group_values = {}
        for key in blocks[block_idx]:
            kind, name, index = key
            if (kind, name) not in captured.variables:
                continue
            spec = captured.variables[(kind, name)]
            # one creation per group (see _get_block)
            if (kind, name) not in group_values:
                creator = getattr(_mb, kind)
                val = creator(name=name, dim=spec['dim'],
                              bound=spec['bound'])
                if isinstance(val, dict):
                    group_values[(kind, name)] = val
                else:
                    # dict / dict-like container (pyomo IndexedVar, COPT
                    # tupledict) / scalar — as_var_group normalizes all
                    # three with guarded attribute access.
                    group_values[(kind, name)] = as_var_group(val)
            native[key] = group_values[(kind, name)][index]

        for cidx in block_constraints[block_idx]:
            constraint = captured.constraints[cidx][0]
            expr = constraint.expression.constant
            for key, coeff in constraint.expression.terms.items():
                if key in native:
                    expr = expr + coeff * native[key]
            sense = constraint.sense
            if sense == '<=':
                _mb.con(expr <= 0)
            elif sense == '>=':
                _mb.con(expr >= 0)
            else:
                _mb.con(expr == 0)
        _mb.features['_reusable_rows'] = True
        entry = {'mb': _mb, 'native': native, 'dummy': None}
        _block_cache[ck] = entry
        return entry

    def _reset_objective(_mb):
        # obj() appends; a reused model must end in the exact state of a
        # fresh model with a single objective (objective_counter starts
        # at [0, 0] and obj() increments it to [1, 1]).
        _mb.features['directions'] = []
        _mb.features['objectives'] = []
        _mb.features['objective_labels'] = []
        _mb.features['objective_counter'] = [0, 0]

    def _block_dummy(entry):
        if entry['dummy'] is None:
            entry['dummy'] = entry['mb'].fvar('_dw2_dummy', bound=[0, 0])
        return entry['dummy']

    def _pricing_objective(native, lambda_vals):
        obj_expr = 0.0
        for key, coeff in obj_terms.items():
            if key in native:
                modifier = 0.0
                for i, cidx in enumerate(linking_indices):
                    a_ij = (captured.constraints[cidx][0]
                            .expression.terms.get(key, 0.0))
                    modifier = modifier + lambda_vals[i] * a_ij
                obj_expr = obj_expr + (coeff - modifier) * native[key]
        return obj_expr

    def _solve_block(block_idx, lambda_vals, label_tag='sub'):
        entry = _get_block(block_idx, label_tag)
        _mb, native = entry['mb'], entry['native']
        obj_expr = _pricing_objective(native, lambda_vals)
        if isinstance(obj_expr, (int, float)) and obj_expr == 0.0:
            obj_expr = 0 * _block_dummy(entry)
        _reset_objective(_mb)
        _mb.obj(obj_expr, direction='min',
                label=f'_dw2_{label_tag}_obj_{block_idx}')
        _mb.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                cpu_threads=cpu_threads, absolute_gap=absolute_gap,
                relative_gap=relative_gap)
        v_k = _mb.get_obj()
        if v_k is None:
            v_k = 0.0
        x_k = _extract_block_values(_mb, blocks[block_idx])
        return float(v_k), x_k

    def _block_obj_val(x_k):
        z_val = obj_constant
        for key, coeff in obj_terms.items():
            if key in x_k:
                z_val = z_val + coeff * x_k[key]
        return z_val

    def _block_r_val(x_k):
        r_val = []
        for cidx in linking_indices:
            r_i = 0.0
            for key, coeff in (captured.constraints[cidx][0]
                               .expression.terms.items()):
                if key in x_k:
                    r_i = r_i + coeff * x_k[key]
            r_val.append(r_i)
        return r_val

    def _solve_block_with_obj(block_idx, obj_coeffs, lambda_vals,
                              label_tag='init'):
        entry = _get_block(block_idx, label_tag)
        _mb, native = entry['mb'], entry['native']
        obj_e = 0.0
        for key, coeff in obj_coeffs.items():
            if key in native:
                modifier = 0.0
                for i, cidx in enumerate(linking_indices):
                    a_ij = (captured.constraints[cidx][0]
                            .expression.terms.get(key, 0.0))
                    modifier = modifier + lambda_vals[i] * a_ij
                obj_e += (coeff - modifier) * native[key]
        if isinstance(obj_e, (int, float)) and obj_e == 0.0:
            obj_e = 0 * _block_dummy(entry)
        _reset_objective(_mb)
        _mb.obj(obj_e, direction='min')
        _mb.sol(directions=['min'], solver=_sv, time_limit=time_limit,
                cpu_threads=cpu_threads, absolute_gap=absolute_gap,
                relative_gap=relative_gap)
        v_k = _mb.get_obj()
        if v_k is None:
            v_k = 0.0
        x_k = _extract_block_values(_mb, blocks[block_idx])
        return float(v_k), x_k

    sign_combos = _sign_combos(n_blocks)
    random_objectives = []
    for signs in sign_combos:
        obj = {}
        for bk_idx, bk_vars in enumerate(blocks):
            for key in bk_vars:
                obj[key] = signs[bk_idx]
        random_objectives.append(obj)

    initial_solutions = []
    for obj_coeffs in random_objectives:
        for bk in range(n_blocks):
            v_k, x_k = _solve_block_with_obj(
                bk, obj_coeffs, [0.0] * n_linking,
                label_tag='init')
            initial_solutions.append({
                'block_idx': bk,
                'z': _block_obj_val(x_k),
                'r': _block_r_val(x_k),
                'x': x_k,
            })

    def build_master(solution_records):
        _m = _model(method=method, name='_dw2_master', interface=_if)
        _m.features['directions'] = list(directions)

        block_groups = {}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            if bk not in block_groups:
                block_groups[bk] = []
            block_groups[bk].append((s, rec))

        u_syms = {}
        z_vals = {}
        r_vals = {}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            u = _m.fvar(f'u_{bk}_{s}', bound=[0, None])
            u_syms[(bk, s)] = u
            z_vals[(bk, s)] = rec['z']
            r_vals[(bk, s)] = rec['r']

        obj = 0.0
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            obj = obj + z_vals[(bk, s)] * u_syms[(bk, s)]
        if isinstance(obj, (int, float)):
            dummy = _m.fvar('_dw2_m_dummy', bound=[0, 0])
            _m.obj(obj + 0 * dummy, direction='min', label='_dw2_obj')
        else:
            _m.obj(obj, direction='min', label='_dw2_obj')

        used_labels = set()
        for i in range(n_linking):
            base = f'_dw2_comp_{i}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            con_expr = 0.0
            for s, rec in enumerate(solution_records):
                bk = rec['block_idx']
                con_expr = (con_expr
                            + r_vals[(bk, s)][i] * u_syms[(bk, s)])
            cidx = linking_indices[i]
            csense = linking_senses[cidx]
            if csense == '<=':
                _m.con(con_expr <= linking_rhs[cidx], name=label)
            elif csense == '>=':
                _m.con(con_expr >= linking_rhs[cidx], name=label)
            else:
                _m.con(con_expr == linking_rhs[cidx], name=label)

        convexity_labels = {}
        for bk in range(n_blocks):
            base = f'_dw2_conv_{bk}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            convexity_labels[bk] = label
            recs = block_groups.get(bk, [])
            if not recs:
                continue
            conv_expr = 0.0
            for s, rec in recs:
                conv_expr = conv_expr + u_syms[(bk, s)]
            _m.con(conv_expr == 1, name=label)

        _m.sol(directions=directions, solver=_sv, time_limit=time_limit,
               cpu_threads=cpu_threads, absolute_gap=absolute_gap,
               relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = (float('inf') if obj_direction == 'min'
                       else float('-inf'))

        duals = {}
        for i in range(n_linking):
            base = f'_dw2_comp_{i}'
            label = base
            cnt = 0
            while True:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[i] = float(d)
                    break
                except Exception:
                    cnt += 1
                    label = f'{base}_{cnt}'
                    if cnt > 10:
                        break
        for bk in range(n_blocks):
            label = convexity_labels.get(bk)
            if label:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[n_linking + bk] = float(d)
                except Exception:
                    pass

        x_star = {k: {} for k in range(n_blocks)}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            try:
                v = _m.get_numpy_var(f'u_{bk}_{s}')
                w = float(v) if not hasattr(v, '__len__') else 0.0
            except Exception:
                w = 0.0
            if abs(w) < 1e-12:
                continue
            for key, val in rec['x'].items():
                x_star[bk][key] = x_star[bk].get(key, 0.0) + w * val

        return {
            'objective': float(obj_val),
            'duals': duals,
            'solution': x_star,
        }

    def pricing_oracle(duals):
        lambda_vals = [duals.get(i, 0.0) for i in range(n_linking)]
        sigma = {bk: duals.get(n_linking + bk, 0.0)
                 for bk in range(n_blocks)}

        all_converged = True
        new_columns = []
        total_v = 0.0
        for bk in range(n_blocks):
            v_k, x_k = _solve_block(bk, lambda_vals)
            total_v += v_k
            if v_k < sigma.get(bk, 0.0) - 1e-6:
                all_converged = False
            new_columns.append({
                'block_idx': bk,
                'z': _block_obj_val(x_k),
                'r': _block_r_val(x_k),
                'x': x_k,
            })

        if all_converged:
            return None

        lambda_b = sum(lambda_vals[i] * linking_rhs[linking_indices[i]]
                       for i in range(n_linking))
        lower_bound = total_v + lambda_b
        if new_columns:
            new_columns[0]['_lower_bound'] = lower_bound
        return new_columns

    def build_modified_master(solution_records):
        """Modified master with artificial variables for violated linking constraints."""
        _m = _model(method=method, name='_dw2_mod_master', interface=_if)
        _m.features['directions'] = list(directions)

        block_groups = {}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            if bk not in block_groups:
                block_groups[bk] = []
            block_groups[bk].append((s, rec))

        u_syms = {}
        z_vals = {}
        r_vals = {}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            u = _m.fvar(f'u_{bk}_{s}', bound=[0, None])
            u_syms[(bk, s)] = u
            z_vals[(bk, s)] = rec['z']
            r_vals[(bk, s)] = rec['r']

        M = 1e6
        art = []
        for i in range(n_linking):
            a_i = _m.fvar(f'_art_{i}', bound=[0, None])
            art.append(a_i)

        obj = 0.0
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            obj = obj + z_vals[(bk, s)] * u_syms[(bk, s)]
        obj = obj + M * sum(art[i] for i in range(n_linking))
        if isinstance(obj, (int, float)):
            dummy = _m.fvar('_dw2_mm_dummy', bound=[0, 0])
            _m.obj(obj + 0 * dummy, direction='min', label='_dw2_obj')
        else:
            _m.obj(obj, direction='min', label='_dw2_obj')

        used_labels = set()
        for i in range(n_linking):
            base = f'_dw2_comp_{i}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            con_expr = 0.0
            for s, rec in enumerate(solution_records):
                bk = rec['block_idx']
                con_expr = (con_expr
                            + r_vals[(bk, s)][i] * u_syms[(bk, s)])
            cidx = linking_indices[i]
            csense = linking_senses[cidx]
            if csense == '<=':
                _m.con(con_expr - art[i] <= linking_rhs[cidx], name=label)
            elif csense == '>=':
                _m.con(con_expr + art[i] >= linking_rhs[cidx], name=label)
            else:
                _m.con(con_expr - art[i] <= linking_rhs[cidx], name=label + '_le')
                _m.con(con_expr + art[i] >= linking_rhs[cidx], name=label + '_ge')

        convexity_labels = {}
        for bk in range(n_blocks):
            base = f'_dw2_conv_{bk}'
            label = base
            cnt = 0
            while label in used_labels:
                cnt += 1
                label = f'{base}_{cnt}'
            used_labels.add(label)
            convexity_labels[bk] = label
            recs = block_groups.get(bk, [])
            if not recs:
                continue
            conv_expr = 0.0
            for s, rec in recs:
                conv_expr = conv_expr + u_syms[(bk, s)]
            _m.con(conv_expr == 1, name=label)

        _m.sol(directions=directions, solver=_sv, time_limit=time_limit,
               cpu_threads=cpu_threads, absolute_gap=absolute_gap,
               relative_gap=relative_gap)

        obj_val = _m.get_obj()
        if obj_val is None:
            obj_val = (float('inf') if obj_direction == 'min'
                       else float('-inf'))

        duals = {}
        for i in range(n_linking):
            base = f'_dw2_comp_{i}'
            label = base
            cnt = 0
            while True:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[i] = float(d)
                    break
                except Exception:
                    cnt += 1
                    label = f'{base}_{cnt}'
                    if cnt > 10:
                        break
        for bk in range(n_blocks):
            label = convexity_labels.get(bk)
            if label:
                try:
                    d = _m.get_dual(label)
                    if d is not None:
                        duals[n_linking + bk] = float(d)
                except Exception:
                    pass

        x_star = {k: {} for k in range(n_blocks)}
        for s, rec in enumerate(solution_records):
            bk = rec['block_idx']
            try:
                v = _m.get_numpy_var(f'u_{bk}_{s}')
                w = float(v) if not hasattr(v, '__len__') else 0.0
            except Exception:
                w = 0.0
            if abs(w) < 1e-12:
                continue
            for key, val in rec['x'].items():
                x_star[bk][key] = x_star[bk].get(key, 0.0) + w * val

        return {
            'objective': float(obj_val),
            'duals': duals,
            'solution': x_star,
        }

    return build_master, build_modified_master, pricing_oracle, initial_solutions


# ------------------------------------------------------------------
# Bind classmethods to ColumnGeneration
# ------------------------------------------------------------------

ColumnGeneration.from_automatic = from_automatic
ColumnGeneration.from_automatic_incremental = from_automatic_incremental
ColumnGeneration.from_automatic_ccg = from_automatic_ccg
ColumnGeneration.from_automatic_dcg = from_automatic_dcg
ColumnGeneration.from_automatic_dw = from_automatic_dw
ColumnGeneration.from_automatic_dw2 = from_automatic_dw2
ColumnGeneration._detect_dw_structure = staticmethod(_detect_dw_structure)
