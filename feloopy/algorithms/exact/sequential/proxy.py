import numpy as np


class _SDMProxy:
    """Dual-purpose proxy that supports both APIs:
    - New: m.state(name="I", dim=1, initial=0)  -> registers var, returns _VarProxy
    - Legacy: m.state[0] / m.state.inventory     -> records expression tree
    """
    __slots__ = ('_name', '_tree', '_model')

    def __init__(self, name, model):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_tree', ('var', name))
        object.__setattr__(self, '_model', model)

    def __call__(self, name=None, dim=1, initial=0.0, init=None, bounds=None):
        if name is None:
            return self
        model = self._model
        if self._name == 'state':
            return model._register_state(name, dim, initial)
        elif self._name == 'decision':
            return model._register_decision(name, dim, init=init, bounds=bounds)
        elif self._name == 'tpar':
            return model._register_tpar(name, dim, init=init, bounds=bounds)
        elif self._name in ('exo', 'exogenous'):
            return model._register_exo(name, dim)
        return None

    def __getitem__(self, idx):
        return _Proxy(('idx', self._tree, idx))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _Proxy(('attr', self._tree, name))

    def __add__(self, o):
        return _Proxy(('add', self._tree, getattr(o, '_tree', ('const', o))))
    def __radd__(self, o):
        return _Proxy(('add', getattr(o, '_tree', ('const', o)), self._tree))
    def __sub__(self, o):
        return _Proxy(('sub', self._tree, getattr(o, '_tree', ('const', o))))
    def __rsub__(self, o):
        return _Proxy(('sub', getattr(o, '_tree', ('const', o)), self._tree))
    def __mul__(self, o):
        return _Proxy(('mul', self._tree, getattr(o, '_tree', ('const', o))))
    def __rmul__(self, o):
        return _Proxy(('mul', getattr(o, '_tree', ('const', o)), self._tree))
    def __truediv__(self, o):
        return _Proxy(('div', self._tree, getattr(o, '_tree', ('const', o))))
    def __rtruediv__(self, o):
        return _Proxy(('div', getattr(o, '_tree', ('const', o)), self._tree))
    def __neg__(self):
        return _Proxy(('neg', self._tree))
    def __abs__(self):
        return _Proxy(('abs', self._tree))
    def __pow__(self, o):
        return _Proxy(('pow', self._tree, getattr(o, '_tree', ('const', o))))
    def __gt__(self, o):
        return _Proxy(('gt', self._tree, getattr(o, '_tree', ('const', o))))
    def __lt__(self, o):
        return _Proxy(('lt', self._tree, getattr(o, '_tree', ('const', o))))
    def __ge__(self, o):
        return _Proxy(('ge', self._tree, getattr(o, '_tree', ('const', o))))
    def __le__(self, o):
        return _Proxy(('le', self._tree, getattr(o, '_tree', ('const', o))))
    def __float__(self):
        return 0.0
    def __int__(self):
        return 0
    def __bool__(self):
        return True
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return _Proxy(('ufunc', ufunc,
                       tuple(a._tree if hasattr(a, '_tree') else ('const', a)
                             for a in args)))
    def __repr__(self):
        return f'SDMProxy({self._name})'


class _Proxy:
    """Records arithmetic/indexing/math operations for deferred evaluation."""
    __slots__ = ('_tree',)

    def __init__(self, tree):
        object.__setattr__(self, '_tree', tree)

    def __getitem__(self, idx):
        return _Proxy(('idx', self._tree, idx))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _Proxy(('attr', self._tree, name))

    def __add__(self, o):
        return _Proxy(('add', self._tree, getattr(o, '_tree', ('const', o))))
    def __radd__(self, o):
        return _Proxy(('add', getattr(o, '_tree', ('const', o)), self._tree))
    def __sub__(self, o):
        return _Proxy(('sub', self._tree, getattr(o, '_tree', ('const', o))))
    def __rsub__(self, o):
        return _Proxy(('sub', getattr(o, '_tree', ('const', o)), self._tree))
    def __mul__(self, o):
        return _Proxy(('mul', self._tree, getattr(o, '_tree', ('const', o))))
    def __rmul__(self, o):
        return _Proxy(('mul', getattr(o, '_tree', ('const', o)), self._tree))
    def __truediv__(self, o):
        return _Proxy(('div', self._tree, getattr(o, '_tree', ('const', o))))
    def __rtruediv__(self, o):
        return _Proxy(('div', getattr(o, '_tree', ('const', o)), self._tree))
    def __neg__(self):
        return _Proxy(('neg', self._tree))
    def __abs__(self):
        return _Proxy(('abs', self._tree))
    def __pow__(self, o):
        return _Proxy(('pow', self._tree, getattr(o, '_tree', ('const', o))))
    def __gt__(self, o):
        return _Proxy(('gt', self._tree, getattr(o, '_tree', ('const', o))))
    def __lt__(self, o):
        return _Proxy(('lt', self._tree, getattr(o, '_tree', ('const', o))))
    def __ge__(self, o):
        return _Proxy(('ge', self._tree, getattr(o, '_tree', ('const', o))))
    def __le__(self, o):
        return _Proxy(('le', self._tree, getattr(o, '_tree', ('const', o))))
    def __eq__(self, o):
        return _Proxy(('eq', self._tree, getattr(o, '_tree', ('const', o))))
    def __ne__(self, o):
        return _Proxy(('ne', self._tree, getattr(o, '_tree', ('const', o))))

    def __float__(self):
        return 0.0
    def __int__(self):
        return 0
    def __bool__(self):
        return True
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return _Proxy(('ufunc', ufunc,
                       tuple(a._tree if hasattr(a, '_tree') else ('const', a)
                             for a in args)))

    def __repr__(self):
        return f'Proxy({self._tree})'


class _RngProxy:
    """Proxy for np.random.Generator that captures method calls for deferred evaluation."""
    __slots__ = ('_name',)
    def __init__(self, name='rng'):
        object.__setattr__(self, '_name', name)
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return _Proxy(('rng_method', self._name, name, args, tuple(kwargs.items())))
        return _method


class _VarProxy:
    """Named variable proxy that records indexing operations for deferred evaluation.

    Two modes:
    - Recording mode: backing=None, __getitem__ returns _Proxy (for building expression trees)
    - Simulation mode: backing=array, __getitem__ returns array[idx] (for computing values)
    """
    __slots__ = ('_name', '_backing', '_tree')

    def __init__(self, name, backing=None):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_backing', backing)
        object.__setattr__(self, '_tree', ('var', name))

    def __getitem__(self, idx):
        if self._backing is not None:
            return self._backing[idx]
        return _Proxy(('idx', self._tree, idx))

    def __add__(self, o):
        if self._backing is not None:
            return self._backing + (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('add', self._tree, getattr(o, '_tree', ('const', o))))
    def __radd__(self, o):
        if self._backing is not None:
            return (o._backing if isinstance(o, _VarProxy) else o) + self._backing
        return _Proxy(('add', getattr(o, '_tree', ('const', o)), self._tree))
    def __sub__(self, o):
        if self._backing is not None:
            return self._backing - (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('sub', self._tree, getattr(o, '_tree', ('const', o))))
    def __rsub__(self, o):
        if self._backing is not None:
            return (o._backing if isinstance(o, _VarProxy) else o) - self._backing
        return _Proxy(('sub', getattr(o, '_tree', ('const', o)), self._tree))
    def __mul__(self, o):
        if self._backing is not None:
            return self._backing * (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('mul', self._tree, getattr(o, '_tree', ('const', o))))
    def __rmul__(self, o):
        if self._backing is not None:
            return (o._backing if isinstance(o, _VarProxy) else o) * self._backing
        return _Proxy(('mul', getattr(o, '_tree', ('const', o)), self._tree))
    def __truediv__(self, o):
        if self._backing is not None:
            return self._backing / (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('div', self._tree, getattr(o, '_tree', ('const', o))))
    def __neg__(self):
        if self._backing is not None:
            return -self._backing
        return _Proxy(('neg', self._tree))
    def __abs__(self):
        if self._backing is not None:
            return abs(self._backing)
        return _Proxy(('abs', self._tree))
    def __pow__(self, o):
        if self._backing is not None:
            return self._backing ** (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('pow', self._tree, getattr(o, '_tree', ('const', o))))
    def __gt__(self, o):
        if self._backing is not None:
            return self._backing > (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('gt', self._tree, getattr(o, '_tree', ('const', o))))
    def __lt__(self, o):
        if self._backing is not None:
            return self._backing < (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('lt', self._tree, getattr(o, '_tree', ('const', o))))
    def __ge__(self, o):
        if self._backing is not None:
            return self._backing >= (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('ge', self._tree, getattr(o, '_tree', ('const', o))))
    def __le__(self, o):
        if self._backing is not None:
            return self._backing <= (o._backing if isinstance(o, _VarProxy) else o)
        return _Proxy(('le', self._tree, getattr(o, '_tree', ('const', o))))
    def __float__(self):
        return float(self._backing) if self._backing is not None else 0.0
    def __int__(self):
        return int(self._backing) if self._backing is not None else 0
    def __bool__(self):
        return True
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if self._backing is not None:
            resolved = tuple(a._backing if isinstance(a, _VarProxy) else a for a in args)
            return ufunc(*resolved)
        return _Proxy(('ufunc', ufunc,
                       tuple(a._tree if hasattr(a, '_tree') else ('const', a) for a in args)))
    def __repr__(self):
        return f'VarProxy({self._name})'


class _SelectProxy:
    """Records m.select(cond, true_val, false_val) for deferred evaluation."""
    __slots__ = ('_cond', '_true', '_false')

    def __init__(self, cond, true_val, false_val):
        object.__setattr__(self, '_cond', cond)
        object.__setattr__(self, '_true', true_val)
        object.__setattr__(self, '_false', false_val)


class _FieldContext(dict):
    """Dict with attached field name mappings for named attribute access."""
    def __init__(self, *args, field_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._field_names = field_names or {}


def _replay(tree, ctx):
    """Evaluate an expression tree with concrete values from ctx."""
    tag = tree[0]
    if tag == 'const':
        return tree[1]
    if tag == 'var':
        return ctx[tree[1]]
    if tag == 'idx':
        return _replay(tree[1], ctx)[tree[2]]
    if tag == 'attr':
        obj = _replay(tree[1], ctx)
        attr_name = tree[2]
        if hasattr(ctx, '_field_names') and tree[1][0] == 'var':
            var_name = tree[1][1]
            fields = ctx._field_names.get(var_name, None)
            if fields and attr_name in fields:
                idx = fields[attr_name]
                return obj[idx]
        return getattr(obj, attr_name)
    if tag == 'add':
        return _replay(tree[1], ctx) + _replay(tree[2], ctx)
    if tag == 'sub':
        return _replay(tree[1], ctx) - _replay(tree[2], ctx)
    if tag == 'mul':
        return _replay(tree[1], ctx) * _replay(tree[2], ctx)
    if tag == 'div':
        return _replay(tree[1], ctx) / _replay(tree[2], ctx)
    if tag == 'neg':
        return -_replay(tree[1], ctx)
    if tag == 'abs':
        return abs(_replay(tree[1], ctx))
    if tag == 'pow':
        return _replay(tree[1], ctx) ** _replay(tree[2], ctx)
    if tag == 'gt':
        return _replay(tree[1], ctx) > _replay(tree[2], ctx)
    if tag == 'lt':
        return _replay(tree[1], ctx) < _replay(tree[2], ctx)
    if tag == 'ge':
        return _replay(tree[1], ctx) >= _replay(tree[2], ctx)
    if tag == 'le':
        return _replay(tree[1], ctx) <= _replay(tree[2], ctx)
    if tag == 'eq':
        return _replay(tree[1], ctx) == _replay(tree[2], ctx)
    if tag == 'ne':
        return _replay(tree[1], ctx) != _replay(tree[2], ctx)
    if tag == 'ufunc':
        args = tuple(_replay(a, ctx) for a in tree[2])
        return tree[1](*args)
    if tag == 'math_func':
        fn, args = tree[1], tree[2]
        args = tuple(_replay(a, ctx) for a in args)
        return fn(*args)
    if tag == 'max':
        a = _replay(tree[1], ctx)
        b = _replay(tree[2], ctx)
        return np.maximum(a, b)
    if tag == 'min':
        a = _replay(tree[1], ctx)
        b = _replay(tree[2], ctx)
        return np.minimum(a, b)
    if tag == 'rng_method':
        rng_var_name, method_name, args, kwargs = tree[1], tree[2], tree[3], tree[4]
        rng_obj = ctx[rng_var_name]
        def _replay_arg(a):
            if isinstance(a, tuple) and len(a) > 0 and isinstance(a[0], str):
                return _replay(a, ctx)
            return a
        args = tuple(_replay_arg(a) for a in args)
        kwargs = {k: _replay_arg(v) for k, v in kwargs}
        return getattr(rng_obj, method_name)(*args, **kwargs)
    raise ValueError(f"Unknown tree tag: {tag}")


def pmax(a, b):
    """Proxy-aware max: works in SDM expressions like pmax(0.0, m.state[0])."""
    ta = a._tree if hasattr(a, '_tree') else ('const', a)
    tb = b._tree if hasattr(b, '_tree') else ('const', b)
    return _Proxy(('max', ta, tb))


def pmin(a, b):
    """Proxy-aware min: works in SDM expressions like pmin(m.state[0], m.exogenous[0])."""
    ta = a._tree if hasattr(a, '_tree') else ('const', a)
    tb = b._tree if hasattr(b, '_tree') else ('const', b)
    return _Proxy(('min', ta, tb))


class StateDict(dict):
    """Array wrapper that supports both index and named attribute access."""
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No field '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
