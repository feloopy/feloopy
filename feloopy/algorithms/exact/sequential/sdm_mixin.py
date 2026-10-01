import numpy as np

from .enums import SequentialError, _resolve_dim
from .proxy import (
    _SDMProxy, _Proxy, _RngProxy, _VarProxy, _SelectProxy,
    _replay, pmax, pmin, _FieldContext,
)
from ....operators.update_operators import _assert_unique_variable_name

# resolved once on the first MathAPI call (see SDMMixin.sum); None means
# "not imported yet", which deliberately mirrors the old per-call import
_MathAPI = None


class _SDMFieldDescriptor:
    def __init__(self, field):
        self.field = field

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _SDMProxy(self.field, obj)


class SDMMixin:
    """Shared SDM methods for ``model`` and ``sdm_model``.

    Storage is unified into the instance ``_sdm_components`` dict.
    """

    state = _SDMFieldDescriptor('state')
    decision = _SDMFieldDescriptor('decision')
    exogenous = _SDMFieldDescriptor('exogenous')
    exo = _SDMFieldDescriptor('exo')
    tpar = _SDMFieldDescriptor('tpar')
    info = _SDMFieldDescriptor('info')

    @property
    def rng(self):
        return _RngProxy('rng')

    def _ensure_sdm(self):
        comps = getattr(self, '_sdm_components', None)
        if comps is None:
            comps = {}
            self._sdm_components = comps
        return comps

    def _setup_sdm_proxies(self):
        """No-op: state/decision/exogenous/rng come from class descriptors/property."""
        pass

    def _get_sdm_components(self):
        """Return registered SDM components with nested keys flattened for the search class."""
        comps = dict(getattr(self, '_sdm_components', {}) or {})
        if '_sdp_vars' in comps:
            comps['sdp_vars'] = comps['_sdp_vars']
        if '_S0' in comps:
            if not isinstance(comps.get('S0'), dict):
                comps['S0'] = comps['_S0']
        if '_theta0' in comps:
            t0 = comps['_theta0']
            if isinstance(t0, dict):
                comps['theta0'] = np.concatenate([v for v in t0.values()]) if t0 else None
            else:
                comps['theta0'] = t0
        if '_theta_bounds' in comps:
            tb = comps['_theta_bounds']
            if isinstance(tb, dict):
                comps['theta_bounds'] = [b for b in tb.values()]
            else:
                comps['theta_bounds'] = tb
        if '_exogenous_fn' in comps:
            comps['exogenous'] = comps['_exogenous_fn']
        if '_inform_target' in comps:
            comps['inform_target'] = comps['_inform_target']
        comps.setdefault('direction', 'min')
        comps.setdefault('transition', None)
        comps.setdefault('cost', None)
        comps.setdefault('reward', None)
        comps.setdefault('exogenous', None)
        comps.setdefault('policy', None)
        comps.setdefault('theta0', None)
        comps.setdefault('theta_bounds', None)
        comps.setdefault('S0', None)
        comps.setdefault('T', None)
        comps.setdefault('N', None)
        comps.setdefault('discount', None)
        comps.setdefault('sdp_vars', comps.get('_sdp_vars', {}))
        comps.setdefault('transition_trees', {})
        comps.setdefault('obj_tree', None)
        comps.setdefault('obj_expression', None)
        comps.setdefault('value_tree', None)
        comps.setdefault('constraint_trees', [])
        comps.setdefault('transition_target', None)
        comps.setdefault('inform_target', None)
        return comps

    def transit(self, target, expr=None):
        """Register a transition function for sequential decision problems.

        Supports:
        1. Lambda/callable: m.transit(lambda s,d,w,t: [s[0]+d[0]-w[0]])
        2. Variable-based: m.transit(I, I + x - D)
        3. Legacy expression list: m.transit([max(0.0, m.state[0] + m.decision[0] - m.exogenous[0])])
        """
        comps = self._ensure_sdm()
        if callable(target) and expr is None:
            comps['transition'] = target
            comps.pop('transition_target', None)
        elif isinstance(target, _VarProxy) and expr is not None:
            target_name = target._name
            tree = expr._tree if hasattr(expr, '_tree') else ('const', expr)
            comps['transition_trees'] = {target_name: tree}
            has_new_vars = bool(comps.get('_sdp_vars'))

            def _compiled(state, decision, exogenous, t):
                s_vars = {k: _VarProxy(k, backing=v) for k, v in state.items()}
                d_vars = {k: _VarProxy(k, backing=v) for k, v in decision.items()}
                e_vars = {k: _VarProxy(k, backing=v) for k, v in exogenous.items()}
                ctx = {**s_vars, **d_vars, **e_vars, 't': t}
                result = _replay(tree, ctx)
                val = result._backing if isinstance(result, _VarProxy) else np.asarray(result)
                if has_new_vars:
                    new_state = {k: v.copy() for k, v in state.items()}
                    new_state[target_name] = val
                    return new_state
                return val

            comps['transition'] = _compiled
            comps['transition_target'] = target_name
        else:
            items = target if isinstance(target, (list, tuple)) else [target]
            trees = [x._tree if hasattr(x, '_tree') else ('const', x) for x in items]

            def _compiled(state, decision, exogenous, t):
                ctx = _FieldContext(
                    {'state': state, 'decision': decision,
                     'exogenous': exogenous, 't': t},
                    field_names=getattr(self, '_field_names', {}))
                return np.array([_replay(tr, ctx) for tr in trees])

            comps['transition'] = _compiled
            comps.pop('transition_target', None)
        return self

    def _register_sdm_cost(self, expression, direction=None):
        """Register stage cost/reward expression into _sdm_components."""
        comps = self._ensure_sdm()
        if direction is not None:
            comps['direction'] = direction
        if callable(expression):
            comps['cost'] = expression
            comps['obj_expression'] = expression
            comps['obj_tree'] = None
        elif hasattr(expression, '_tree'):
            tree = expression._tree
            comps['obj_tree'] = tree
            comps['obj_expression'] = expression
            if direction is None:
                comps.setdefault('direction', 'min')
            has_new_vars = bool(comps.get('_sdp_vars'))
            if has_new_vars:

                def _cost_fn(state, decision, exogenous, t):
                    s_vars = {k: _VarProxy(k, backing=v) for k, v in state.items()}
                    d_vars = {k: _VarProxy(k, backing=v) for k, v in decision.items()}
                    e_vars = {k: _VarProxy(k, backing=v) for k, v in exogenous.items()}
                    ctx = {**s_vars, **d_vars, **e_vars, 't': t}
                    r = _replay(tree, ctx)
                    val = r._backing if isinstance(r, _VarProxy) else r
                    return float(val)
            else:

                def _cost_fn(state, decision, exogenous, t):
                    ctx = _FieldContext(
                        {'state': state, 'decision': decision,
                         'exogenous': exogenous, 't': t},
                        field_names=getattr(self, '_field_names', {}))
                    r = _replay(tree, ctx)
                    if isinstance(r, _VarProxy):
                        return float(r._backing)
                    return float(r[0]) if hasattr(r, '__len__') else float(r)

            comps['cost'] = _cost_fn
        else:
            comps['obj_expression'] = expression
            comps['obj_tree'] = None
            if expression is not None:
                comps['cost'] = lambda s, d, w, t: float(expression)
        return self

    def inform(self, target, expr=None):
        """Register exogenous information (legacy API)."""
        comps = self._ensure_sdm()
        if callable(target) and expr is None:
            comps['exogenous'] = target
            comps.pop('inform_target', None)
        else:
            items = target if isinstance(target, (list, tuple)) else [target]
            trees = [x._tree if hasattr(x, '_tree') else ('const', x) for x in items]

            def _compiled(t, rng):
                ctx = _FieldContext({'t': t, 'rng': rng},
                                    field_names=getattr(self, '_field_names', {}))
                return np.array([_replay(tr, ctx) for tr in trees])

            comps['exogenous'] = _compiled
            comps.pop('inform_target', None)
        return self

    def update(self, fn):
        """Register a decision rule for sequential decision problems.

        Parameters
        ----------
        fn : callable or dict
            - callable: fn(state, theta) -> decision dict
            - dict: {var: expression} or {name_str: expression}
        """
        comps = self._ensure_sdm()
        if callable(fn):
            comps['policy'] = fn
            return self
        if isinstance(fn, dict):
            _sdp_vars = comps.get('_sdp_vars', {})
            _tpar_names = {k for k, v in _sdp_vars.items() if v.get('type') == 'tpar'}
            _dict = fn

            def _compile_select(val, ctx):
                if isinstance(val, _SelectProxy):
                    cond_val = _compile_select(val._cond, ctx)
                    if np.all(cond_val):
                        return _compile_select(val._true, ctx)
                    else:
                        return _compile_select(val._false, ctx)
                if isinstance(val, _VarProxy):
                    return ctx[val._name]
                if isinstance(val, _Proxy):
                    return _replay(val._tree, ctx)
                if hasattr(val, '_tree'):
                    return _replay(val._tree, ctx)
                return np.asarray(val, dtype=float)

            def _compiled_policy(state, theta, **kwargs):
                theta_dict = (theta if isinstance(theta, dict)
                              else {_k: theta for _k in _tpar_names})
                ctx = {}
                for k, v in state.items():
                    ctx[k] = v
                for k, v in theta_dict.items():
                    ctx[k] = v
                result = {}
                for key, expr in _dict.items():
                    name = key._name if isinstance(key, _VarProxy) else str(key)
                    result[name] = np.atleast_1d(
                        _compile_select(expr, ctx)).flatten()
                return result

            comps['policy'] = _compiled_policy
            return self
        raise SequentialError(
            "update() requires a callable or a dict of expressions")

    def value(self, expression):
        """Register a value function for VFA sequential decision problems."""
        comps = self._ensure_sdm()
        if hasattr(expression, '_tree'):
            comps['value_tree'] = expression._tree
        else:
            comps['value_tree'] = ('const', float(expression))
        return self

    val = value

    def theta(self, initial=None, bounds=None):
        comps = self._ensure_sdm()
        if initial is not None:
            comps['theta0'] = (np.asarray(initial, dtype=float)
                               if not isinstance(initial, dict) else initial)
        if bounds is not None:
            comps['theta_bounds'] = bounds
        return self

    def state0(self, S0):
        comps = self._ensure_sdm()
        comps['S0'] = S0
        return self

    def horizon(self, T):
        comps = self._ensure_sdm()
        comps['T'] = int(T)
        return self

    def nsim(self, N):
        comps = self._ensure_sdm()
        comps['N'] = int(N)
        return self

    def discount(self, gamma):
        comps = self._ensure_sdm()
        comps['discount'] = float(gamma)
        return self

    def state_fields(self, names):
        if not hasattr(self, '_field_names'):
            self._field_names = {}
        self._field_names['state'] = {name: i for i, name in enumerate(names)}
        return self

    def decision_fields(self, names):
        if not hasattr(self, '_field_names'):
            self._field_names = {}
        self._field_names['decision'] = {name: i for i, name in enumerate(names)}
        return self

    def exogenous_fields(self, names):
        if not hasattr(self, '_field_names'):
            self._field_names = {}
        self._field_names['exogenous'] = {name: i for i, name in enumerate(names)}
        return self

    def _register_state(self, name, dim=1, initial=0.0):
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        initial = np.atleast_1d(np.asarray(initial, dtype=float))
        if initial.size == 1:
            initial = np.full(dim, initial[0])
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'state', 'dim': dim, 'initial': initial, 'proxy': proxy
        }
        if '_S0' not in comps or not isinstance(comps['_S0'], dict):
            comps['_S0'] = {}
        comps['_S0'][name] = initial.copy()
        return proxy

    def _register_exo(self, name, dim=1):
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {'type': 'exo', 'dim': dim, 'proxy': proxy}
        return proxy

    def _register_decision(self, name, dim=1, init=None, bounds=None):
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        proxy = _VarProxy(name)
        entry = {'type': 'dp', 'dim': dim, 'proxy': proxy}
        if init is not None:
            init = np.atleast_1d(np.asarray(init, dtype=float))
            if init.size == 1:
                init = np.full(dim, init[0])
            entry['initial'] = init
        if bounds is not None:
            entry['bounds'] = bounds
        comps['_sdp_vars'][name] = entry
        return proxy

    def _register_tpar(self, name, dim=1, init=None, bounds=None):
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        init = np.atleast_1d(np.asarray(
            init if init is not None else np.zeros(dim), dtype=float))
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'tpar', 'dim': dim, 'initial': init,
            'bounds': bounds, 'proxy': proxy
        }
        if '_theta0' not in comps or not isinstance(comps['_theta0'], dict):
            comps['_theta0'] = {}
        comps['_theta0'][name] = init.copy()
        if bounds is not None:
            if '_theta_bounds' not in comps or not isinstance(comps['_theta_bounds'], dict):
                comps['_theta_bounds'] = {}
            comps['_theta_bounds'][name] = bounds
        return proxy

    def select(self, cond, true_val, false_val):
        """Proxy-aware ternary: m.select(I < thr, tgt - I, 0.0)."""
        return _SelectProxy(cond, true_val, false_val)

    def max(self, *args):
        """Proxy-aware max with MathAPI fallback for solver expressions."""
        global _MathAPI
        if any(hasattr(a, '_tree') for a in args):
            if len(args) == 2:
                return pmax(args[0], args[1])
            result = args[0]
            for a in args[1:]:
                result = pmax(result, a)
            return result
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            try:
                _mapi = _MathAPI
                if _mapi is None:
                    from ....classes.math_api import MathAPI
                    _mapi = _MathAPI = MathAPI
                return _mapi.max(self, *args)
            except Exception:
                pass
        if len(args) == 1:
            return np.max(args[0])
        if len(args) == 2 and np.ndim(args[0]) == 0 and np.ndim(args[1]) == 0:
            return np.maximum(args[0], args[1])
        return np.maximum.reduce(args)

    def min(self, *args):
        """Proxy-aware min with MathAPI fallback for solver expressions."""
        global _MathAPI
        if any(hasattr(a, '_tree') for a in args):
            if len(args) == 2:
                return pmin(args[0], args[1])
            result = args[0]
            for a in args[1:]:
                result = pmin(result, a)
            return result
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            try:
                _mapi = _MathAPI
                if _mapi is None:
                    from ....classes.math_api import MathAPI
                    _mapi = _MathAPI = MathAPI
                return _mapi.min(self, *args)
            except Exception:
                pass
        if len(args) == 1:
            return np.min(args[0])
        if len(args) == 2 and np.ndim(args[0]) == 0 and np.ndim(args[1]) == 0:
            return np.minimum(args[0], args[1])
        return np.minimum.reduce(args)

    def sum(self, x):
        """Proxy-aware sum with MathAPI fallback for solver expressions."""
        global _MathAPI
        if hasattr(x, '_tree') or hasattr(x, '_native'):
            tx = x._tree if hasattr(x, '_tree') else ('const', x)
            return _Proxy(('math_func', lambda v: float(np.sum(v)), (tx,)))
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            try:
                # the import stays inside the same try/except as the call, but
                # the resolved class is cached: this path runs six figures per
                # solve and the import statement alone costs ~1us every time
                _mapi = _MathAPI
                if _mapi is None:
                    from ....classes.math_api import MathAPI
                    _mapi = _MathAPI = MathAPI
                return _mapi.sum(self, x)
            except Exception:
                pass
        return sum(x)

    def log(self, x):
        if hasattr(x, '_tree'):
            return _Proxy(('math_func', lambda v: float(np.log(v)), (x._tree,)))
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            from ....classes.math_api import MathAPI
            return MathAPI.log(self, x)
        return np.log(x)

    def sqrt(self, x):
        if hasattr(x, '_tree'):
            return _Proxy(('math_func', lambda v: float(np.sqrt(v)), (x._tree,)))
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            from ....classes.math_api import MathAPI
            return MathAPI.sqrt(self, x)
        return np.sqrt(x)

    def exp(self, x):
        if hasattr(x, '_tree'):
            return _Proxy(('math_func', lambda v: float(np.exp(v)), (x._tree,)))
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            from ....classes.math_api import MathAPI
            return MathAPI.exp(self, x)
        return np.exp(x)

    def abs(self, x):
        if hasattr(x, '_tree'):
            return _Proxy(('math_func', lambda v: float(np.abs(v)), (x._tree,)))
        features = getattr(self, 'features', None)
        if features is not None and 'interface_name' in features:
            from ....classes.math_api import MathAPI
            return MathAPI.abs(self, x)
        return np.abs(x)

    def clip(self, x, lo, hi):
        if hasattr(x, '_tree') or hasattr(lo, '_tree') or hasattr(hi, '_tree'):
            tx = x._tree if hasattr(x, '_tree') else ('const', x)
            tlo = lo._tree if hasattr(lo, '_tree') else ('const', lo)
            thi = hi._tree if hasattr(hi, '_tree') else ('const', hi)
            return _Proxy(('math_func', lambda v, a, b: float(np.clip(v, a, b)),
                           (tx, tlo, thi)))
        return np.clip(x, lo, hi)
