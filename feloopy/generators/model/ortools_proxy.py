# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

INFINITY = 1e30


class OrtoolsExprProxy:
    __slots__ = ('terms', 'constant', 'is_integer')

    def __init__(self, terms=None, constant=0.0, is_integer=False):
        self.terms = terms if terms is not None else {}
        self.constant = constant
        self.is_integer = is_integer

    def _coerce(self, other):
        if isinstance(other, OrtoolsVarProxy):
            return OrtoolsExprProxy({other._name: 1.0}, 0.0, other.is_integer)
        if isinstance(other, OrtoolsExprProxy):
            return other
        if isinstance(other, (int, float)):
            return OrtoolsExprProxy({}, float(other), False)
        return NotImplemented

    def _add_expr(self, other):
        new_terms = dict(self.terms)
        new_const = self.constant
        new_int = self.is_integer
        for k, v in other.terms.items():
            new_terms[k] = new_terms.get(k, 0.0) + v
        new_const += other.constant
        if other.is_integer:
            new_int = True
        return OrtoolsExprProxy(new_terms, new_const, new_int)

    def _sub_expr(self, other):
        new_terms = dict(self.terms)
        new_const = self.constant
        for k, v in other.terms.items():
            new_terms[k] = new_terms.get(k, 0.0) - v
        new_const -= other.constant
        return OrtoolsExprProxy(new_terms, new_const, self.is_integer)

    def _mul_scalar(self, scalar):
        s = float(scalar)
        new_terms = {k: v * s for k, v in self.terms.items()}
        return OrtoolsExprProxy(new_terms, self.constant * s, self.is_integer)

    def __add__(self, other):
        c = self._coerce(other)
        if c is NotImplemented:
            return NotImplemented
        return self._add_expr(c)

    def __radd__(self, other):
        c = self._coerce(other)
        if c is NotImplemented:
            return NotImplemented
        return c._add_expr(self)

    def __sub__(self, other):
        c = self._coerce(other)
        if c is NotImplemented:
            return NotImplemented
        return self._sub_expr(c)

    def __rsub__(self, other):
        c = self._coerce(other)
        if c is NotImplemented:
            return NotImplemented
        return c._sub_expr(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._mul_scalar(other)
        if isinstance(other, OrtoolsVarProxy):
            if len(self.terms) == 0:
                return OrtoolsExprProxy({other._name: self.constant}, 0.0, self.is_integer or other.is_integer)
            raise NotImplementedError("Nonlinear terms not supported in linear programming")
        if isinstance(other, OrtoolsExprProxy):
            if len(self.terms) == 0:
                return other._mul_scalar(self.constant)
            if len(other.terms) == 0:
                return self._mul_scalar(other.constant)
            raise NotImplementedError("Nonlinear terms not supported in linear programming")
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self._mul_scalar(other)
        return NotImplemented

    def __pow__(self, other):
        raise NotImplementedError("Nonlinear terms not supported in linear programming")

    def __neg__(self):
        return self._mul_scalar(-1.0)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._mul_scalar(1.0 / other)
        return NotImplemented

    def __rtruediv__(self, other):
        raise NotImplementedError("Division with expression as denominator not supported in linear programming")

    def __pos__(self):
        return OrtoolsExprProxy(dict(self.terms), self.constant, self.is_integer)

    def __le__(self, other):
        return OrtoolsComparisonProxy(self, '<=', other)

    def __ge__(self, other):
        return OrtoolsComparisonProxy(self, '>=', other)

    def __eq__(self, other):
        return OrtoolsComparisonProxy(self, '==', other)

    def __lt__(self, other):
        return OrtoolsComparisonProxy(self, '<', other)

    def __gt__(self, other):
        return OrtoolsComparisonProxy(self, '>', other)

    def __ne__(self, other):
        return OrtoolsComparisonProxy(self, '!=', other)

    def evaluate(self, real_vars):
        from ortools.linear_solver import pywraplp as ortools_interface
        items = [(coeff, real_vars[var_name]) for var_name, coeff in self.terms.items() if var_name in real_vars]
        if not items:
            return self.constant
        coeff, var = items[0]
        result = coeff * var
        for c, v in items[1:]:
            result = result + c * v
        if self.constant != 0:
            result = result + self.constant
        return result


class OrtoolsComparisonProxy:
    __slots__ = ('lhs', 'sense', 'rhs', 'epsilon')

    def __init__(self, lhs, sense, rhs, epsilon=1e-6):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs
        self.epsilon = epsilon

    def _to_expr(self, value, real_vars):
        if isinstance(value, OrtoolsExprProxy):
            return value.evaluate(real_vars)
        if isinstance(value, OrtoolsVarProxy):
            return 1.0 * real_vars[value._name] if value._name in real_vars else value._name
        if isinstance(value, (int, float)):
            return float(value)
        return value

    def evaluate(self, real_vars):
        lhs_expr = self._to_expr(self.lhs, real_vars)
        rhs_expr = self._to_expr(self.rhs, real_vars)
        match self.sense:
            case '<=':
                return lhs_expr <= rhs_expr
            case '>=':
                return lhs_expr >= rhs_expr
            case '==':
                return lhs_expr == rhs_expr
            case '<':
                if isinstance(rhs_expr, (int, float)):
                    return lhs_expr <= rhs_expr - self.epsilon
                return lhs_expr <= rhs_expr - self.epsilon
            case '>':
                if isinstance(rhs_expr, (int, float)):
                    return lhs_expr >= rhs_expr + self.epsilon
                return lhs_expr >= rhs_expr + self.epsilon
            case '!=':
                raise NotImplementedError("!= constraints are decomposed into <= and >= by the framework")


class OrtoolsVarProxy:
    __slots__ = ('_name', '_lb', '_ub', 'is_integer', '_index')

    def __init__(self, name, lb, ub, is_integer=False, index=0):
        self._name = name
        self._lb = lb
        self._ub = ub
        self.is_integer = is_integer
        self._index = index

    def name(self):
        return self._name

    def lb(self):
        return self._lb

    def ub(self):
        return self._ub

    def bounds(self):
        return (self._lb, self._ub)

    def index(self):
        return self._index

    def solution_value(self):
        return None

    def __add__(self, other):
        return OrtoolsExprProxy({self._name: 1.0}, 0.0, self.is_integer).__add__(other)

    def __radd__(self, other):
        return OrtoolsExprProxy({self._name: 1.0}, 0.0, self.is_integer).__radd__(other)

    def __sub__(self, other):
        return OrtoolsExprProxy({self._name: 1.0}, 0.0, self.is_integer).__sub__(other)

    def __rsub__(self, other):
        return OrtoolsExprProxy({self._name: 1.0}, 0.0, self.is_integer).__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return OrtoolsExprProxy({self._name: float(other)}, 0.0, self.is_integer)
        if isinstance(other, (OrtoolsVarProxy, OrtoolsExprProxy)):
            raise NotImplementedError("Nonlinear terms not supported in linear programming")
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return OrtoolsExprProxy({self._name: float(other)}, 0.0, self.is_integer)
        return NotImplemented

    def __neg__(self):
        return OrtoolsExprProxy({self._name: -1.0}, 0.0, self.is_integer)

    def __pos__(self):
        return OrtoolsExprProxy({self._name: 1.0}, 0.0, self.is_integer)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return OrtoolsExprProxy({self._name: 1.0 / other}, 0.0, self.is_integer)
        return NotImplemented

    def __pow__(self, other):
        raise NotImplementedError("Nonlinear terms not supported in linear programming")

    def __le__(self, other):
        return OrtoolsComparisonProxy(self, '<=', other)

    def __ge__(self, other):
        return OrtoolsComparisonProxy(self, '>=', other)

    def __eq__(self, other):
        return OrtoolsComparisonProxy(self, '==', other)

    def __lt__(self, other):
        return OrtoolsComparisonProxy(self, '<', other)

    def __gt__(self, other):
        return OrtoolsComparisonProxy(self, '>', other)

    def __ne__(self, other):
        return OrtoolsComparisonProxy(self, '!=', other)


class OrtoolsObjectiveProxy:
    __slots__ = ('expr', '_value')

    def __init__(self, expr=None):
        self.expr = expr
        self._value = None

    def Value(self):
        return self._value


class OrtoolsModelProxy:
    def __init__(self):
        self._variable_specs = []
        self._var_index_counter = 0
        self._variables = {}

    @property
    def variable_specs(self):
        return self._variable_specs

    def NumVar(self, lb, ub, name):
        if lb is None:
            lb = -INFINITY
        if ub is None:
            ub = INFINITY
        proxy = OrtoolsVarProxy(name, lb, ub, is_integer=False, index=self._var_index_counter)
        self._var_index_counter += 1
        self._variable_specs.append({'name': name, 'lb': lb, 'ub': ub, 'is_integer': False})
        self._variables[name] = proxy
        return proxy

    def IntVar(self, lb, ub, name):
        if lb is None:
            lb = -INFINITY
        if ub is None:
            ub = INFINITY
        proxy = OrtoolsVarProxy(name, lb, ub, is_integer=True, index=self._var_index_counter)
        self._var_index_counter += 1
        self._variable_specs.append({'name': name, 'lb': lb, 'ub': ub, 'is_integer': True})
        self._variables[name] = proxy
        return proxy

    def infinity(self):
        return INFINITY

    def variables(self):
        return list(self._variables.values())

    def Objective(self):
        return OrtoolsObjectiveProxy()
