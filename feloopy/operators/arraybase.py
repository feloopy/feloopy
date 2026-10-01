# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Shared ndarray base for every array feloopy hands to user code.
"""

import numpy as np

# hoisted: these run inside the penalty consumer millions of times per
# solve, so the slots and tuples are bound once at import
_INT_TYPES = (int, np.integer)
_nd_getitem = np.ndarray.__getitem__
_nd_sub = np.ndarray.__sub__
_SL = slice(None)
_ALL = (_SL,)
_NEWAXIS = (None,)


def _hash_values(arr):
    """Value-based hash for array-like values.

    A size-1 array hashes exactly like the equivalent Python number, so an
    element pulled out of a variable can key a dict built from plain ints
    (``routes = {1: []}; routes[x[i]]``).  Larger arrays hash over their
    contents, so equal arrays always collide and plain numpy-style
    "unhashable type" errors disappear.  The hash is value based only --
    never dtype or shape -- because ``==`` compares values as well.
    """
    if arr.size == 1:
        return hash(arr.item())
    try:
        return hash(tuple(arr.reshape(-1).tolist()))
    except TypeError:
        # object arrays holding unhashable leaves: bytes still hash
        return hash(arr.tobytes())


def _is_variable(value):
    """True when *value* is a decision variable.

    Data compared against data keeps numpy's plain boolean answer; the
    moment a decision variable is involved, the comparison becomes the
    constraint residual the penalty consumer reads.
    """
    return isinstance(value, _ArrayBase) and not value._is_data


def _as_residual(value):
    """Present a comparison result as a ``_Residual``."""
    if type(value) is _Residual:
        return value
    if isinstance(value, np.ndarray):
        return value.view(_Residual)
    # an operand with ``__array_ufunc__`` (DataRef) can turn a 0-d
    # operation into a numpy scalar
    return _Residual(np.asarray(value))


class _ArrayBase(np.ndarray):

    # every instance without a set-dim tuple->slot map falls back here; keeping
    # it at class level means the instance dict is only written for vars that
    # actually carry a map
    _key_map = None
    # axis 0 only holds the population when the interface vectorizes (feloopy,
    # pymoo).  Every other solver lays a variable out as exactly its declared
    # dims, so the axis-skipping fast path in __getitem__ has to stay off there
    # or a full index lands one axis too deep.
    _vec = False
    # data parameters extract Python scalars from element access so values
    # stay usable as indexes, json numbers and isinstance targets; decision
    # variables keep views so a comparison stays a numeric residual.
    # Deliberately not named ``_native``: LinearizationProxy and the SDM
    # machinery duck-type on ``_native`` for solver-owned objects.
    _is_data = False

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # views keep the set-dim tuple->slot map (None for every other var);
        # the getattr() builtin resolves a missing attribute in C (a Python
        # try/except is ~3x slower on that path), and skipping the write for
        # the None case keeps the instance dict untouched on fresh objects
        km = getattr(obj, '_key_map', None)
        if km is not None:
            self._key_map = km
        vc = getattr(obj, '_vec', None)
        if vc:
            self._vec = vc

    def _declared_key(self, key):
        """True when a lone key addresses the declared dims (population
        implicit) and so needs the axis-skip prepend.

        Slices, lists, ranges and index arrays all name declared elements.
        Boolean masks only qualify when their shape fits the declared axes:
        masks that span the population axis already encode it, so they keep
        numpy's own meaning (``x[x > 0.5]`` selects per-agent elements).
        """
        if isinstance(key, (slice, list, range)):
            return True
        if isinstance(key, np.ndarray):
            return (key.dtype != bool
                    or key.shape == (self.shape[1],)
                    or key.shape == self.shape[1:])
        return False

    def __getitem__(self, key):
        _km = self._key_map
        if _km:
            try:
                # set-dim variable: d[i,v,t,p] -> its slot in the set
                key = _km[key]
            except (TypeError, KeyError):
                pass
        if not self._vec:
            # no population axis: each declared dim is a real axis, so a full
            # index is a scalar and a partial one is the trailing sub-array --
            # plain numpy, which is exactly what the non-vectorized penalty
            # consumer (model.sol, float() per constraint) expects
            return _nd_getitem(self, key)
        nd = self.ndim
        if nd >= 2:
            if isinstance(key, _INT_TYPES):
                # index straight to a (n, 1) column instead of indexing and
                # then reshaping: one numpy call, one object, one finalize.
                # bool keys are excluded: numpy reads them as 0-D masks, so
                # they add an axis instead of consuming one.
                if nd == 2 and type(key) is not bool:
                    return _nd_getitem(self, (_SL, key, None))
                result = _nd_getitem(self, (_SL, key))
                if result.ndim < 2:
                    result = result.reshape(-1, 1)
                return result
            if isinstance(key, tuple):
                _bool_idx = False
                for k in key:
                    if type(k) is bool:
                        _bool_idx = True
                    elif not isinstance(k, _INT_TYPES):
                        break
                else:
                    if len(key) == nd:
                        # a full index over every axis (numpy's own repr and
                        # iteration probe arrays with ``a[i, -1]``): there is
                        # no population axis left to skip, and prepending one
                        # would index one axis too deep
                        return _nd_getitem(self, key)
                    if not _bool_idx and len(key) == nd - 1:
                        # all axes after axis 0 are consumed -> would collapse
                        # to 1-D; None appends the column axis directly
                        return _nd_getitem(self, _ALL + key + _NEWAXIS)
                    result = _nd_getitem(self, _ALL + key)
                    if result.ndim < 2:
                        result = result.reshape(-1, 1)
                    return result
                # mixed tuple (slice / list / index-array elements): fewer
                # keys than array axes still means the population axis stays
                # implicit, so every element addresses the declared dims --
                # the same rule the lone int, slice and fancy keys apply.
                # A tuple that spans all axes falls through to numpy's
                # explicit-population meaning below.
                if len(key) < nd and all(
                        isinstance(k, (_INT_TYPES, slice, list, range,
                                       np.ndarray)) or type(k) is bool
                        for k in key):
                    result = _nd_getitem(self, _ALL + key)
                    if result.ndim < 2:
                        result = result.reshape(-1, 1)
                    return result
                return _nd_getitem(self, key)
            if self._declared_key(key):
                # a lone key addresses the declared dims just like a lone
                # int/slice: the population axis stays implicit, so on
                # dim=[30], x[1:] is variables 1..29 for every agent ->
                # (pop, 29) and x[[0, 2]] is those two variables -> (pop, 2).
                # Plain numpy would apply the key to axis 0 and hand back
                # agent rows instead of variables (the ZDT1 x[1:] blow-up,
                # and its list/array/range cousins).
                result = _nd_getitem(self, _ALL + (key,))
                if result.ndim < 2:
                    result = result.reshape(-1, 1)
                return result
        return _nd_getitem(self, key)

    def __sub__(self, other):
        # ndarray ops already preserve the subclass, so re-wrapping only
        # rebuilt the same view through asarray().view() a second time
        out = _nd_sub(self, other)
        if type(out) is type(self):
            return out
        return type(self)(out)

    # -- Python value protocol ----------------------------------------

    def __hash__(self):
        return _hash_values(self)

    def __contains__(self, item):
        # raw value equality: __eq__ answers a decision variable with
        # constraint residuals (0 means *equal*), which must not leak into
        # ``x in variable`` membership tests
        return bool(np.any(np.equal(self, item)))

    def __index__(self):
        if self.size == 1:
            value = self.item()
            if isinstance(value, (bool, np.bool_, int, np.integer)):
                return int(value)
        raise TypeError(
            "only integer scalar arrays can be converted to a scalar index")

    def __int__(self):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return int(self.item())

    def __float__(self):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return float(self.item())

    def __complex__(self):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return complex(self.item())

    def __round__(self, ndigits=None):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be rounded")
        value = self.item()
        return round(value) if ndigits is None else round(value, ndigits)

    # -- comparisons ----------------------------------------------------
    #
    # A decision variable answers with the residual the penalty consumer
    # reads (``<=`` is ``lhs - rhs``, ``>=`` is ``rhs - lhs``, ``==`` is
    # ``|lhs - rhs|``, ``!=`` is ``1 - |lhs - rhs| / eps``); data compared
    # with data keeps numpy's boolean answer, exactly as before the two
    # array types shared this base.  Either way the result is a
    # ``_Residual``, so its *value* stays numeric while its truth value
    # reports whether the comparison holds.

    def __le__(self, other):
        if self._is_data and not _is_variable(other):
            return _as_residual(np.less_equal(self, other))
        return _as_residual(_nd_sub(self, other))

    def __ge__(self, other):
        if self._is_data and not _is_variable(other):
            return _as_residual(np.greater_equal(self, other))
        return _as_residual(np.subtract(other, self))

    def __eq__(self, other):
        # ``a == b`` has to hand the penalty consumer (model.sol) a residual
        # like __le__/__ge__ do, not a boolean: the penalty clips negatives,
        # so a satisfied equality must evaluate to 0 and a violated one to
        # |a - b|.  ndarray.__eq__ returns True (=1) exactly when the
        # constraint holds, which got penalised as a violation of 1 and made
        # every equality-constrained heuristic model report a positive
        # penalty for feasible solutions.
        if self._is_data and not _is_variable(other):
            return _as_residual(np.equal(self, other))
        return _as_residual(np.abs(_nd_sub(self, other)))

    def __ne__(self, other):
        # the exact path encodes ``a != b`` as |a - b| >= epsilon (1e-6).
        # The distance to that feasible set is at most epsilon itself, which
        # is exactly the feasibility tolerance, so a raw residual would never
        # be penalised (penalty**2 ~ 1e-12) nor ever push the status over
        # ``tol`` -- a violated ``!=`` looked feasible and cost nothing.
        # Normalise the distance by epsilon instead: 1 when the values are
        # equal, falling linearly to 0 at the boundary, so the violation is
        # both visible to the status check and gives search a gradient.
        if self._is_data and not _is_variable(other):
            return _as_residual(np.not_equal(self, other))
        return _as_residual(1.0 - np.abs(_nd_sub(self, other)) / 1e-6)

    def __lt__(self, other):
        # strict comparisons cannot produce a residual (the feasible set has
        # no boundary to measure against), so they keep numpy's boolean
        # answer where True already means satisfied
        return _as_residual(np.less(self, other))

    def __gt__(self, other):
        return _as_residual(np.greater(self, other))


class _Residual(_ArrayBase):
    """A comparison answer: numeric for the penalty, truthful for Python.

    The value is what the penalty consumer reads -- a signed residual for
    ``<=``/``>=``, ``|lhs - rhs|`` for ``==``, ``1 - |lhs - rhs| / eps``
    for ``!=``, numpy's boolean mask for ``<``/``>`` and for data-to-data
    comparisons where no decision variable is involved.

    The truth value answers the question Python actually asked: a
    comparison is True when it *holds* -- every residual <= 0, every mask
    True -- so residuals drop straight into dict lookups (``d[k] == v``
    colliding in a hash bucket), ``in`` tests and ``if`` statements
    instead of raising or reporting the violation as True.
    """

    def __bool__(self):
        # strip the subclass first: ufuncs and reductions preserve it, so
        # ``bool(np.all(self))`` would receive another _Residual and re-enter
        # this method forever; ``self <= 0`` would likewise re-enter __le__
        raw = self.view(np.ndarray)
        if raw.dtype == bool:
            return bool(np.all(raw))
        return bool(np.all(raw <= 0))
