import numpy as np

from .proxy import _SelectProxy, _VarProxy, _Proxy


def _normalize_bounds(info, default=(None, None)):
    """Return a flat ``[lb, ub]`` pair from a stored bounds entry.

    Stage variables keep their bounds in whichever shape the declaration
    used: ``[lb, ub]``, feloopy's per-dimension ``[[lb, ub]]``, ``None``
    for a free variable, or no ``bounds`` key at all.  Normalising here
    keeps every declaration path usable by the generated stage model.
    """
    b = info.get('bounds', None)
    if b is None:
        b = list(default)
    else:
        b = list(b)
        if len(b) == 1 and isinstance(b[0], (list, tuple)):
            b = list(b[0])
        if len(b) > 2:
            b = [b[0], b[1]]
        while len(b) < 2:
            b.append(default[len(b)])
    return b


def _make_build_model_fn(sdp_vars, sdm, include_value=False,
                          interface=None):
    """Create a build_model(state, theta) function for CFA/VFA/DLA.

    Uses m.obj() as-is for objective, m.con() for constraints.
    No auto-generation or theta weighting.
    """
    try:
        from ....helpers.solver_params import _HEURISTIC_IFACES
    except Exception:  # pragma: no cover
        _HEURISTIC_IFACES = frozenset()
    # a heuristic-only interface (mealpy, niapy, ...) has no stage model to
    # build: fall back to the default exact interface instead of failing
    # inside flp.model()
    stage_interface = interface or 'highs'
    if stage_interface in _HEURISTIC_IFACES:
        stage_interface = 'highs'

    _obj_tree = sdm.get('obj_tree', None)
    _value_tree = sdm.get('value_tree', None) if include_value else None
    _direction = sdm.get('direction', 'min')

    _state_vars = {k: v for k, v in sdp_vars.items() if v['type'] == 'state'}
    _dp_vars = {k: v for k, v in sdp_vars.items() if v['type'] == 'dp'}
    _tpar_vars = {k: v for k, v in sdp_vars.items() if v['type'] == 'tpar'}
    _exo_vars = {k: v for k, v in sdp_vars.items() if v['type'] == 'exo'}
    _sdp_tpar_vals = {k: v.get('initial', 0.0) for k, v in _tpar_vars.items()}

    def _compile_select_lp(val, ctx, m, counter):
        if isinstance(val, _SelectProxy):
            cond_val = _compile_select_lp(val._cond, ctx, m, counter)
            if np.all(cond_val):
                return _compile_select_lp(val._true, ctx, m, counter)
            else:
                return _compile_select_lp(val._false, ctx, m, counter)
        if isinstance(val, _VarProxy):
            v = ctx[val._name]
            return float(v) if np.ndim(v) == 0 else v
        if isinstance(val, _Proxy):
            return _replay_lp(val._tree, ctx, m, counter)
        if hasattr(val, '_tree'):
            return _replay_lp(val._tree, ctx, m, counter)
        v = np.asarray(val, dtype=float)
        return float(v) if v.ndim == 0 else v

    def _replay_lp(tree, ctx, m, counter):
        tag = tree[0]
        if tag == 'const':
            val = tree[1]
            if isinstance(val, _SelectProxy):
                return _compile_select_lp(val, ctx, m, counter)
            return val
        if tag == 'var':
            return ctx[tree[1]]
        if tag == 'idx':
            return _replay_lp(tree[1], ctx, m, counter)[tree[2]]
        if tag == 'attr':
            return getattr(_replay_lp(tree[1], ctx, m, counter), tree[2])
        if tag == 'add':
            return _replay_lp(tree[1], ctx, m, counter) + _replay_lp(tree[2], ctx, m, counter)
        if tag == 'sub':
            return _replay_lp(tree[1], ctx, m, counter) - _replay_lp(tree[2], ctx, m, counter)
        if tag == 'mul':
            return _replay_lp(tree[1], ctx, m, counter) * _replay_lp(tree[2], ctx, m, counter)
        if tag == 'div':
            return _replay_lp(tree[1], ctx, m, counter) / _replay_lp(tree[2], ctx, m, counter)
        if tag == 'neg':
            return -_replay_lp(tree[1], ctx, m, counter)
        if tag == 'abs':
            return abs(_replay_lp(tree[1], ctx, m, counter))
        if tag == 'pow':
            return _replay_lp(tree[1], ctx, m, counter) ** _replay_lp(tree[2], ctx, m, counter)
        if tag == 'max':
            a = _replay_lp(tree[1], ctx, m, counter)
            b = _replay_lp(tree[2], ctx, m, counter)
            z = m.fvar('_zmax' + str(counter[0]), bound=[None, None])
            counter[0] += 1
            m.con(z >= a)
            m.con(z >= b)
            return z
        if tag == 'min':
            a = _replay_lp(tree[1], ctx, m, counter)
            b = _replay_lp(tree[2], ctx, m, counter)
            z = m.fvar('_zmin' + str(counter[0]), bound=[None, None])
            counter[0] += 1
            m.con(z <= a)
            m.con(z <= b)
            return z
        if tag == 'gt':
            return _replay_lp(tree[1], ctx, m, counter) > _replay_lp(tree[2], ctx, m, counter)
        if tag == 'lt':
            return _replay_lp(tree[1], ctx, m, counter) < _replay_lp(tree[2], ctx, m, counter)
        if tag == 'ge':
            return _replay_lp(tree[1], ctx, m, counter) >= _replay_lp(tree[2], ctx, m, counter)
        if tag == 'le':
            return _replay_lp(tree[1], ctx, m, counter) <= _replay_lp(tree[2], ctx, m, counter)
        if tag == 'eq':
            return _replay_lp(tree[1], ctx, m, counter) == _replay_lp(tree[2], ctx, m, counter)
        if tag == 'ne':
            return _replay_lp(tree[1], ctx, m, counter) != _replay_lp(tree[2], ctx, m, counter)
        if tag == 'ufunc':
            args = tuple(_replay_lp(a, ctx, m, counter) for a in tree[2])
            return tree[1](*args)
        if tag == 'math_func':
            fn, args = tree[1], tree[2]
            args = tuple(_replay_lp(a, ctx, m, counter) for a in args)
            return fn(*args)
        raise ValueError(f"Unknown tree tag: {tag!r}")

    def build_model(state, theta_or_data, *args, **kwargs):
        import feloopy as flp
        m = flp.model(interface=stage_interface)
        if isinstance(theta_or_data, dict):
            theta_dict = {k: _sdp_tpar_vals[k] for k in _tpar_vars}
        else:
            theta_dict = (theta_or_data if isinstance(theta_or_data, dict)
                          else {k: theta_or_data[i] for i, k in enumerate(_tpar_vars)})
        ctx = {}
        if isinstance(state, dict):
            for k, v in state.items():
                arr = np.atleast_1d(np.asarray(v, dtype=float)).flatten()
                ctx[k] = float(arr[0]) if len(arr) == 1 else arr
        for k, v in theta_dict.items():
            arr = np.atleast_1d(np.asarray(v, dtype=float)).flatten()
            ctx[k] = float(arr[0]) if len(arr) == 1 else arr

        for name, info in _dp_vars.items():
            fdim = info.get('dim', 1)
            feloopy_dim = 0 if fdim == 1 else [range(fdim)]
            b = _normalize_bounds(info)
            if b[0] is None:
                b[0] = -1e9
            if b[1] is None:
                b[1] = 1e9
            vt = info.get('var_type', 'cont')
            if vt == 'int':
                mv = m.ivar(name, feloopy_dim, bound=b)
            elif vt == 'bin':
                mv = m.bvar(name, feloopy_dim)
            else:
                mv = m.fvar(name, feloopy_dim, bound=b)
            ctx[name] = mv

        for name, info in _state_vars.items():
            if name not in ctx:
                ctx[name] = 0.0

        for name, info in _exo_vars.items():
            if name not in ctx:
                ctx[name] = 0.0

        counter = [0]

        result = None
        if _obj_tree is not None:
            result = _replay_lp(_obj_tree, ctx, m, counter)
            if isinstance(result, np.ndarray) and result.ndim == 0:
                result = float(result)

        if _value_tree is not None:
            val = _replay_lp(_value_tree, ctx, m, counter)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = float(val)
            if result is None:
                result = -val if _direction == 'max' else val
            elif _direction == 'max':
                result = result - val
            else:
                result = result + val

        if result is not None:
            # one combined objective: registering the stage cost and the
            # value term separately would build a two-objective model
            m.obj(result, direction=_direction)

        _constraint_trees = sdm.get('constraint_trees', [])
        for tree in _constraint_trees:
            expr = _replay_lp(tree, ctx, m, counter)
            if isinstance(expr, tuple) and len(expr) == 2:
                op, (lhs, rhs) = expr[0], expr[1]
                if op == 'le':
                    m.con(lhs <= rhs)
                elif op == 'ge':
                    m.con(lhs >= rhs)
                elif op == 'eq':
                    m.con(lhs == rhs)
            elif hasattr(expr, '__class__') and 'inequality' in str(type(expr)).lower():
                m.con(expr)

        return m

    return build_model
