# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mosek

INF = 1e30


class MosekVar:
    """Lightweight variable wrapper supporting arithmetic for MOSEK."""
    __slots__ = ('_task', '_idx', '_name')

    def __init__(self, task, idx, name=''):
        object.__setattr__(self, '_task', task)
        object.__setattr__(self, '_idx', idx)
        object.__setattr__(self, '_name', name)

    @property
    def idx(self):
        return self._idx

    def _make_lin(self, coeff, const=0.0):
        return MosekLin(self._task, {self._idx: coeff}, const)

    def __add__(self, other):
        if isinstance(other, MosekVar):
            return self._make_lin(1.0) + other._make_lin(1.0)
        if isinstance(other, MosekLin):
            return self._make_lin(1.0) + other
        return self._make_lin(1.0, float(other))

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self._make_lin(1.0, float(other))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MosekVar):
            return self._make_lin(1.0) + other._make_lin(-1.0)
        if isinstance(other, MosekLin):
            return self._make_lin(1.0) + other.negate()
        return self._make_lin(1.0, -float(other))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self._make_lin(-1.0, float(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._make_lin(float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self._make_lin(float(other))
        return NotImplemented

    def __neg__(self):
        return self._make_lin(-1.0)

    def __le__(self, other):
        return MosekConstr(self._task, self._make_lin(1.0), '<=', other)

    def __ge__(self, other):
        return MosekConstr(self._task, self._make_lin(1.0), '>=', other)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return MosekConstr(self._task, self._make_lin(1.0), '==', other)
        return NotImplemented

    def __repr__(self):
        return f"MosekVar({self._name}, idx={self._idx})"


class MosekLin:
    """Linear expression: coeffs dict + constant."""
    __slots__ = ('_task', '_coeffs', '_const')

    def __init__(self, task, coeffs, const=0.0):
        object.__setattr__(self, '_task', task)
        object.__setattr__(self, '_coeffs', dict(coeffs))
        object.__setattr__(self, '_const', const)

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def const(self):
        return self._const

    def negate(self):
        return MosekLin(self._task, {k: -v for k, v in self._coeffs.items()}, -self._const)

    def _combine(self, other, sign):
        new_coeffs = dict(self._coeffs)
        if isinstance(other, MosekVar):
            new_coeffs[other._idx] = new_coeffs.get(other._idx, 0) + sign
            return MosekLin(self._task, new_coeffs, self._const)
        if isinstance(other, MosekLin):
            for k, v in other._coeffs.items():
                new_coeffs[k] = new_coeffs.get(k, 0) + sign * v
            return MosekLin(self._task, new_coeffs, self._const + sign * other._const)
        if isinstance(other, (int, float)):
            return MosekLin(self._task, new_coeffs, self._const + sign * float(other))
        return NotImplemented

    def __add__(self, other):
        return self._combine(other, 1.0)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return MosekLin(self._task, dict(self._coeffs), self._const + float(other))
        return NotImplemented

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            neg = self.negate()
            return MosekLin(self._task, dict(neg._coeffs), neg._const + float(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = float(other)
            return MosekLin(self._task, {k: v * c for k, v in self._coeffs.items()}, self._const * c)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.negate()

    def __le__(self, other):
        return MosekConstr(self._task, self, '<=', other)

    def __ge__(self, other):
        return MosekConstr(self._task, self, '>=', other)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return MosekConstr(self._task, self, '==', other)
        return NotImplemented

    def __repr__(self):
        return f"MosekLin({self._coeffs}, {self._const})"


class MosekConstr:
    """Stores a constraint for later addition to the task."""
    __slots__ = ('_task', '_expr', '_sense', '_rhs')

    def __init__(self, task, expr, sense, rhs):
        object.__setattr__(self, '_task', task)
        object.__setattr__(self, '_expr', expr)
        object.__setattr__(self, '_sense', sense)
        object.__setattr__(self, '_rhs', rhs)

    @property
    def coeffs(self):
        return self._expr.coeffs

    @property
    def const(self):
        return self._expr.const

    @property
    def sense(self):
        return self._sense

    @property
    def rhs(self):
        return self._rhs

    def __repr__(self):
        return f"MosekConstr({self._expr} {self._sense} {self._rhs})"
