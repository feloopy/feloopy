# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def _fmt_literal(other):
    """Format a numeric literal for Picat code generation.

    Picat integer variables never satisfy ``X #= 2.0`` (float literal), and
    numpy scalars stringify with a trailing ``.0`` — emit integral floats
    and numpy numbers as plain integers and leave everything else unchanged.
    """
    if isinstance(other, str):
        return other
    try:
        f = float(other)
    except (TypeError, ValueError):
        return str(other)
    if f.is_integer() and abs(f) < 9.2e18:
        return str(int(f))
    return str(other)


class PicatExpr:
    """Symbolic expression for Picat code generation."""

    def __init__(self, expr_str):
        self._picat_expr = str(expr_str)

    def _wrap(self, other):
        if isinstance(other, (PicatVar, PicatExpr)):
            return other._picat_expr
        return _fmt_literal(other)

    def __add__(self, other):
        return PicatExpr(f"({self._picat_expr}+{self._wrap(other)})")

    def __radd__(self, other):
        return PicatExpr(f"({self._wrap(other)}+{self._picat_expr})")

    def __sub__(self, other):
        return PicatExpr(f"({self._picat_expr}-{self._wrap(other)})")

    def __rsub__(self, other):
        return PicatExpr(f"({self._wrap(other)}-{self._picat_expr})")

    def __mul__(self, other):
        return PicatExpr(f"({self._picat_expr}*{self._wrap(other)})")

    def __rmul__(self, other):
        return PicatExpr(f"({self._wrap(other)}*{self._picat_expr})")

    def __truediv__(self, other):
        return PicatExpr(f"({self._picat_expr}//{self._wrap(other)})")

    def __rtruediv__(self, other):
        return PicatExpr(f"({self._wrap(other)}//{self._picat_expr})")

    def __mod__(self, other):
        return PicatExpr(f"({self._picat_expr} mod {self._wrap(other)})")

    def __neg__(self):
        return PicatExpr(f"(-{self._picat_expr})")

    def __pos__(self):
        return PicatExpr(f"(+{self._picat_expr})")

    def __abs__(self):
        return PicatExpr(f"abs({self._picat_expr})")

    def __pow__(self, other):
        return PicatExpr(f"({self._picat_expr}**{self._wrap(other)})")

    def __le__(self, other):
        return PicatConstraint(self._picat_expr, '#<=', self._wrap(other))

    def __ge__(self, other):
        return PicatConstraint(self._picat_expr, '#>=', self._wrap(other))

    def __eq__(self, other):
        return PicatConstraint(self._picat_expr, '#=', self._wrap(other))

    def __ne__(self, other):
        return PicatConstraint(self._picat_expr, '#!=', self._wrap(other))

    def __lt__(self, other):
        return PicatConstraint(self._picat_expr, '#<', self._wrap(other))

    def __gt__(self, other):
        return PicatConstraint(self._picat_expr, '#>', self._wrap(other))

    def __repr__(self):
        return self._picat_expr

    def __hash__(self):
        return hash(self._picat_expr)


class PicatVar:
    """Picat variable with arithmetic operator overloading."""

    def __init__(self, name, lb, ub, model, vtype='ivar'):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.model = model
        self.vtype = vtype
        self._picat_expr = name

    def _wrap(self, other):
        if isinstance(other, (PicatVar, PicatExpr)):
            return other._picat_expr
        return _fmt_literal(other)

    def __add__(self, other):
        return PicatExpr(f"({self._picat_expr}+{self._wrap(other)})")

    def __radd__(self, other):
        return PicatExpr(f"({self._wrap(other)}+{self._picat_expr})")

    def __sub__(self, other):
        return PicatExpr(f"({self._picat_expr}-{self._wrap(other)})")

    def __rsub__(self, other):
        return PicatExpr(f"({self._wrap(other)}-{self._picat_expr})")

    def __mul__(self, other):
        return PicatExpr(f"({self._picat_expr}*{self._wrap(other)})")

    def __rmul__(self, other):
        return PicatExpr(f"({self._wrap(other)}*{self._picat_expr})")

    def __truediv__(self, other):
        return PicatExpr(f"({self._picat_expr}//{self._wrap(other)})")

    def __rtruediv__(self, other):
        return PicatExpr(f"({self._wrap(other)}//{self._picat_expr})")

    def __mod__(self, other):
        return PicatExpr(f"({self._picat_expr} mod {self._wrap(other)})")

    def __neg__(self):
        return PicatExpr(f"(-{self._picat_expr})")

    def __pos__(self):
        return PicatExpr(f"(+{self._picat_expr})")

    def __abs__(self):
        return PicatExpr(f"abs({self._picat_expr})")

    def __pow__(self, other):
        return PicatExpr(f"({self._picat_expr}**{self._wrap(other)})")

    def __le__(self, other):
        return PicatConstraint(self._picat_expr, '#<=', self._wrap(other))

    def __ge__(self, other):
        return PicatConstraint(self._picat_expr, '#>=', self._wrap(other))

    def __eq__(self, other):
        if other is None:
            return NotImplemented
        return PicatConstraint(self._picat_expr, '#=', self._wrap(other))

    def __ne__(self, other):
        if other is None:
            return NotImplemented
        return PicatConstraint(self._picat_expr, '#!=', self._wrap(other))

    def __lt__(self, other):
        return PicatConstraint(self._picat_expr, '#<', self._wrap(other))

    def __gt__(self, other):
        return PicatConstraint(self._picat_expr, '#>', self._wrap(other))

    def __repr__(self):
        return self._picat_expr

    def __hash__(self):
        return hash(self._picat_expr)


class PicatConstraint:
    """Stores a constraint for Picat serialization."""

    def __init__(self, left, op, right):
        if isinstance(left, (PicatVar, PicatExpr)):
            self.left = left._picat_expr
        else:
            self.left = str(left)
        self.op = op
        if isinstance(right, (PicatVar, PicatExpr)):
            self.right = right._picat_expr
        else:
            self.right = str(right)

    def to_picat(self):
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self):
        return self.to_picat()

    def __hash__(self):
        return hash((self.left, self.op, self.right))
