# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def jl_safe_name(name):
    s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name))
    if s and s[0].isdigit():
        s = '_' + s
    return s


class JumpExpr:
    """A Julia expression string with operator overloading."""

    __slots__ = ('_expr',)

    def __init__(self, expr=''):
        object.__setattr__(self, '_expr', str(expr))

    @property
    def expr(self):
        return self._expr

    def _binary(self, op, other):
        if isinstance(other, JumpExpr):
            return JumpExpr(f"({self._expr}) {op} ({other._expr})")
        if isinstance(other, JumpVar):
            return JumpExpr(f"({self._expr}) {op} ({other._full_name()})")
        if isinstance(other, (int, float)):
            return JumpExpr(f"({self._expr}) {op} {other}")
        return NotImplemented

    def _rbinary(self, op, other):
        if isinstance(other, (int, float)):
            return JumpExpr(f"{other} {op} ({self._expr})")
        return NotImplemented

    def __add__(self, other): return self._binary('+', other)
    def __radd__(self, other): return self._rbinary('+', other)
    def __sub__(self, other): return self._binary('-', other)
    def __rsub__(self, other): return self._rbinary('-', other)
    def __mul__(self, other): return self._binary('*', other)
    def __rmul__(self, other): return self._rbinary('*', other)
    def __truediv__(self, other): return self._binary('/', other)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return JumpExpr(f"{other} / ({self._expr})")
        return NotImplemented
    def __floordiv__(self, other): return self._binary('\\', other)
    def __mod__(self, other): return self._binary('%', other)
    def __pow__(self, other): return self._binary('^', other)
    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return JumpExpr(f"{other}^({self._expr})")
        return NotImplemented

    def __neg__(self): return JumpExpr(f"-({self._expr})")
    def __pos__(self): return JumpExpr(f"+({self._expr})")
    def __abs__(self): return JumpExpr(f"abs({self._expr})")

    def __le__(self, other):
        if isinstance(other, JumpExpr): return JumpConstr(f"({self._expr}) <= ({other._expr})")
        if isinstance(other, JumpVar): return JumpConstr(f"({self._expr}) <= ({other._name})")
        if isinstance(other, (int, float)): return JumpConstr(f"({self._expr}) <= {other}")
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, JumpExpr): return JumpConstr(f"({self._expr}) >= ({other._expr})")
        if isinstance(other, JumpVar): return JumpConstr(f"({self._expr}) >= ({other._name})")
        if isinstance(other, (int, float)): return JumpConstr(f"({self._expr}) >= {other}")
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, JumpExpr): return JumpConstr(f"({self._expr}) < ({other._expr})")
        if isinstance(other, JumpVar): return JumpConstr(f"({self._expr}) < ({other._name})")
        if isinstance(other, (int, float)): return JumpConstr(f"({self._expr}) < {other}")
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, JumpExpr): return JumpConstr(f"({self._expr}) > ({other._expr})")
        if isinstance(other, JumpVar): return JumpConstr(f"({self._expr}) > ({other._name})")
        if isinstance(other, (int, float)): return JumpConstr(f"({self._expr}) > {other}")
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)): return JumpConstr(f"({self._expr}) == {other}")
        return NotImplemented

    def __repr__(self): return self._expr
    def __str__(self): return self._expr


class JumpVar:
    """A JuMP variable proxy that builds Julia expression strings."""

    __slots__ = ('_name', '_declaration', '_index_str')

    def __init__(self, name, declaration='', index_str=''):
        object.__setattr__(self, '_name', str(name))
        object.__setattr__(self, '_declaration', declaration)
        object.__setattr__(self, '_index_str', index_str)

    @property
    def name(self):
        return self._name

    @property
    def declaration(self):
        return self._declaration

    @property
    def index_str(self):
        return self._index_str

    def _full_name(self):
        if self._index_str:
            return f"{self._name}[{self._index_str}]"
        return self._name

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key_str = ', '.join(str(k) for k in key)
        else:
            key_str = str(key)
        return JumpVar(self._name, self._declaration, key_str)

    def _binary(self, op, other):
        fn = self._full_name()
        if isinstance(other, JumpVar): return JumpExpr(f"({fn}) {op} ({other._full_name()})")
        if isinstance(other, JumpExpr): return JumpExpr(f"({fn}) {op} ({other._expr})")
        if isinstance(other, (int, float)): return JumpExpr(f"({fn}) {op} {other}")
        return NotImplemented

    def _rbinary(self, op, other):
        fn = self._full_name()
        if isinstance(other, (int, float)): return JumpExpr(f"{other} {op} ({fn})")
        return NotImplemented

    def __add__(self, other): return self._binary('+', other)
    def __radd__(self, other): return self._rbinary('+', other)
    def __sub__(self, other): return self._binary('-', other)
    def __rsub__(self, other): return self._rbinary('-', other)
    def __mul__(self, other): return self._binary('*', other)
    def __rmul__(self, other): return self._rbinary('*', other)
    def __truediv__(self, other): return self._binary('/', other)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)): return JumpExpr(f"{other} / ({self._full_name()})")
        return NotImplemented
    def __floordiv__(self, other): return self._binary('\\', other)
    def __mod__(self, other): return self._binary('%', other)
    def __pow__(self, other): return self._binary('^', other)
    def __rpow__(self, other):
        if isinstance(other, (int, float)): return JumpExpr(f"{other}^({self._full_name()})")
        return NotImplemented

    def __neg__(self): return JumpExpr(f"-({self._full_name()})")
    def __pos__(self): return JumpExpr(f"+({self._full_name()})")
    def __abs__(self): return JumpExpr(f"abs({self._full_name()})")

    def __le__(self, other):
        fn = self._full_name()
        if isinstance(other, JumpVar): return JumpConstr(f"({fn}) <= ({other._full_name()})")
        if isinstance(other, JumpExpr): return JumpConstr(f"({fn}) <= ({other._expr})")
        if isinstance(other, (int, float)): return JumpConstr(f"({fn}) <= {other}")
        return NotImplemented

    def __ge__(self, other):
        fn = self._full_name()
        if isinstance(other, JumpVar): return JumpConstr(f"({fn}) >= ({other._full_name()})")
        if isinstance(other, JumpExpr): return JumpConstr(f"({fn}) >= ({other._expr})")
        if isinstance(other, (int, float)): return JumpConstr(f"({fn}) >= {other}")
        return NotImplemented

    def __lt__(self, other):
        fn = self._full_name()
        if isinstance(other, JumpVar): return JumpConstr(f"({fn}) < ({other._full_name()})")
        if isinstance(other, JumpExpr): return JumpConstr(f"({fn}) < ({other._expr})")
        if isinstance(other, (int, float)): return JumpConstr(f"({fn}) < {other}")
        return NotImplemented

    def __gt__(self, other):
        fn = self._full_name()
        if isinstance(other, JumpVar): return JumpConstr(f"({fn}) > ({other._full_name()})")
        if isinstance(other, JumpExpr): return JumpConstr(f"({fn}) > ({other._expr})")
        if isinstance(other, (int, float)): return JumpConstr(f"({fn}) > {other}")
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)): return JumpConstr(f"({self._full_name()}) == {other}")
        return NotImplemented

    def __repr__(self): return self._full_name()
    def __str__(self): return self._full_name()


class JumpConstr:
    """A JuMP constraint string."""

    __slots__ = ('_constr_str',)

    def __init__(self, constr_str=''):
        object.__setattr__(self, '_constr_str', str(constr_str))

    @property
    def constr_str(self):
        return self._constr_str

    def __repr__(self): return self._constr_str
    def __str__(self): return self._constr_str


def _expr_to_str(obj):
    """Convert a JumpVar, JumpExpr, JumpConstr, or raw value to a Julia expression string."""
    if isinstance(obj, JumpConstr): return obj.constr_str
    if isinstance(obj, JumpExpr): return obj.expr
    if isinstance(obj, JumpVar): return obj._full_name()
    return str(obj)


def _is_jump_obj(obj):
    return isinstance(obj, (JumpVar, JumpExpr, JumpConstr))


def jump_sum(iterable):
    """Sum of JuMP expressions — drop-in for Python sum()."""
    it = list(iterable)
    if not it:
        return JumpExpr("0")
    result = it[0]
    for item in it[1:]:
        result = result + item
    if isinstance(result, (int, float)):
        return JumpExpr(str(result))
    return result


def jump_abs(x):
    if _is_jump_obj(x):
        return JumpExpr(f"abs({_expr_to_str(x)})")
    return abs(x)


def jump_min(a, b):
    a_str = _expr_to_str(a) if _is_jump_obj(a) else str(a)
    b_str = _expr_to_str(b) if _is_jump_obj(b) else str(b)
    return JumpExpr(f"min({a_str}, {b_str})")


def jump_max(a, b):
    a_str = _expr_to_str(a) if _is_jump_obj(a) else str(a)
    b_str = _expr_to_str(b) if _is_jump_obj(b) else str(b)
    return JumpExpr(f"max({a_str}, {b_str})")


def jump_sin(x):  return JumpExpr(f"sin({_expr_to_str(x)})")
def jump_cos(x):  return JumpExpr(f"cos({_expr_to_str(x)})")
def jump_tan(x):  return JumpExpr(f"tan({_expr_to_str(x)})")
def jump_log(x):  return JumpExpr(f"log({_expr_to_str(x)})")
def jump_exp(x):  return JumpExpr(f"exp({_expr_to_str(x)})")
def jump_sqrt(x): return JumpExpr(f"sqrt({_expr_to_str(x)})")
