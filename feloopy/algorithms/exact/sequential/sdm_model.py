import numpy as np

from .enums import SequentialError, _resolve_dim
from .proxy import (
    _SDMProxy, _Proxy, _RngProxy, _VarProxy, _SelectProxy,
    _replay, pmax, pmin, _FieldContext,
)
from .sdm_mixin import SDMMixin
from .build_model import _normalize_bounds
from ....operators.update_operators import _assert_unique_variable_name


class sdm_model(SDMMixin):
    """A feloopy-like model for sequential decision problems.

    Shared SDM methods (transit, obj, inform, update, value, theta, etc.)
    come from ``SDMMixin`` with unified ``_sdm_components`` dict storage.

    SDM-specific methods defined here:
    - xvar/ivar/pvar/bvar/fvar : variable registration
    - ppar/svar                : parameter/state registration
    - sol                      : dispatch to flp.search()
    """

    def __init__(self, name='sdm'):
        self.name = name
        self._sdm_components = {
            'direction': 'min',
            '_sdp_vars': {},
        }
        self.period = 0
        self.features = {
            'directions': ['min'],
            'objectives': [],
            'variables': {},
        }
        self.solutions = {}
        self.data = {}

    def xvar(self, name, dim=0, inform=None):
        """Define an exogenous information variable with its generation function."""
        fdim = _resolve_dim(dim)
        proxy = self._register_exo(name, fdim)
        if inform is not None:
            comps = self._ensure_sdm()
            if callable(inform) and not hasattr(inform, '_tree'):
                comps['_exogenous_fn'] = inform
            elif hasattr(inform, '_tree'):
                tree = inform._tree

                def _compiled(t, rng):
                    ctx = {'rng': rng, 't': t}
                    result = _replay(tree, ctx)
                    val = result._backing if isinstance(result, _VarProxy) else np.asarray(result)
                    return {name: np.atleast_1d(val).flatten()}

                comps['_exogenous_fn'] = _compiled
                comps['_inform_target'] = name
        return proxy

    def con(self, *args, **kwargs):
        """No-op for SDM models (constraints handled by policy)."""
        pass

    def ivar(self, name, dim=0, init=0, bound=None, bounds=None):
        """Define an integer decision variable (returns proxy for expressions)."""
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        bounds = bounds or bound or [0, None]
        fdim = _resolve_dim(dim)
        init = np.atleast_1d(np.asarray(init, dtype=float))
        if init.size == 1:
            init = np.full(fdim, init[0])
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'dp', 'dim': fdim, 'initial': init,
            'bounds': bounds, 'var_type': 'int', 'proxy': proxy
        }
        return proxy

    def pvar(self, name, dim=0, init=0.0, bound=None, bounds=None):
        """Define a positive continuous decision variable (returns proxy for expressions)."""
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        bounds = bounds or bound
        fdim = _resolve_dim(dim)
        init = np.atleast_1d(np.asarray(init, dtype=float))
        if init.size == 1:
            init = np.full(fdim, init[0])
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'dp', 'dim': fdim, 'initial': init,
            'bounds': bounds or [0, None], 'var_type': 'cont', 'proxy': proxy
        }
        return proxy

    def bvar(self, name, dim=0, init=0):
        """Define a binary decision variable (returns proxy for expressions)."""
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        fdim = _resolve_dim(dim)
        init = np.atleast_1d(np.asarray(init, dtype=float))
        if init.size == 1:
            init = np.full(fdim, init[0])
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'dp', 'dim': fdim, 'initial': init,
            'bounds': [[0, 1]], 'var_type': 'bin', 'proxy': proxy
        }
        return proxy

    def fvar(self, name, dim=0, bound=None):
        """Define a free decision variable (returns proxy for expressions)."""
        comps = self._ensure_sdm()
        if '_sdp_vars' not in comps:
            comps['_sdp_vars'] = {}
        _assert_unique_variable_name(
            name, self.features, sdp_types=comps['_sdp_vars'])
        fdim = _resolve_dim(dim)
        proxy = _VarProxy(name)
        comps['_sdp_vars'][name] = {
            'type': 'dp', 'dim': fdim, 'bounds': bound or [None, None],
            'var_type': 'cont', 'proxy': proxy
        }
        return proxy

    def get(self, name, default=None):
        """Get attribute by name."""
        return getattr(self, name, default)

    def obj(self, expression=None, direction='min', label=None):
        """Register stage cost/reward."""
        return self._register_sdm_cost(expression, direction)

    def ppar(self, name, dim=0, initial=0.0, bound=None):
        """Define a tunable parameter (individual, named)."""
        fdim = _resolve_dim(dim)
        init_val = np.atleast_1d(np.asarray(initial, dtype=float)).flatten()
        if init_val.size == 1 and fdim > 1:
            init_val = np.full(fdim, init_val[0])
        return self._register_tpar(name, fdim, init_val, bound)

    def svar(self, name, initial):
        """Define a state variable whose shape and value come from a parameter."""
        val = initial
        if hasattr(val, '_arr'):
            val = val._arr
        elif hasattr(val, 'value'):
            val = val.value
        val = np.atleast_1d(np.asarray(val, dtype=float))
        dim = val.size
        return self._register_state(name, dim, val)

    def sol(self, directions=None, solver=None, solver_options=None,
            obj_operators=None, uncertainty_set_constraints=None, **kwargs):
        """Solve this model by dispatching to flp.search()."""
        import feloopy as flp

        comps = self._get_sdm_components()
        _directions = directions or self.features.get('directions', [comps.get('direction', 'min')])
        _sdp_vars = comps.get('sdp_vars', {})
        _transition_trees = comps.get('transition_trees', {})
        _obj_tree = comps.get('obj_tree')
        _direction = comps.get('direction', 'min')
        _policy_fn = comps.get('policy')

        def _env(m):
            for name, info in _sdp_vars.items():
                vt = info['type']
                if vt == 'state':
                    m.svar(name=name, initial=info.get('initial', 0.0))
                elif vt == 'dp':
                    b = _normalize_bounds(info)
                    if b[0] is None:
                        b[0] = -1e9
                    if b[1] is None:
                        b[1] = 1e9
                    vtype = info.get('var_type', 'cont')
                    if vtype == 'int':
                        m.ivar(name=name, bound=b)
                    elif vtype == 'bin':
                        m.bvar(name=name, bound=[0, 1])
                    else:
                        m.fvar(name=name, bound=b)
                elif vt == 'tpar':
                    b = info.get('bounds', [None, None])
                    m.ppar(name=name, initial=info.get('initial', 0.0), bound=b)
                elif vt == 'exo':
                    m.xvar(name=name, inform=info.get('inform', None))

            if _transition_trees:
                proxy_ctx = {}
                for n, info in _sdp_vars.items():
                    proxy_ctx[n] = info['proxy']
                for target_name, tree in _transition_trees.items():
                    result = _replay(tree, proxy_ctx)
                    if target_name in _sdp_vars:
                        m.transit(_sdp_vars[target_name]['proxy'], result)

            if _obj_tree is not None:
                proxy_ctx = {}
                for n, info in _sdp_vars.items():
                    proxy_ctx[n] = info['proxy']
                result = _replay(_obj_tree, proxy_ctx)
                m.obj(result, direction=_direction)

            if _policy_fn is not None:
                m.update(_policy_fn)

            return m

        r = flp.search(
            environment=_env,
            method='heuristic',
            interface='feloopy',
            solver=solver or 'orig-de',
            directions=_directions,
            options=solver_options or {},
            verbose=False,
            repeat=1,
            should_run=True,
        )

        self.solutions = {}
        self.features['variables'] = {}
        if hasattr(r, 'solutions') and r.solutions:
            for (typ, vn), val in r.solutions.items():
                self.solutions[vn] = val
                self.features['variables'][(typ, vn)] = val
        if hasattr(r, 'data'):
            self.data = r.data

        return self
