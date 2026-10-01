# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
import numpy as np

sets = it.product

INFINITY = float('inf')


class UnoVar:
    """Symbolic variable for Uno interface that supports algebraic operations."""

    _counter = 0

    def __init__(self, index, name='', lb=None, ub=None):
        self.index = index
        self.name = name
        self.lb = lb if lb is not None else -INFINITY
        self.ub = ub if ub is not None else INFINITY

    def __repr__(self):
        return f"UnoVar({self.name}, idx={self.index})"

    def __add__(self, other):
        return UnoExpr('+', self, other)

    def __radd__(self, other):
        return UnoExpr('+', other, self)

    def __sub__(self, other):
        return UnoExpr('-', self, other)

    def __rsub__(self, other):
        return UnoExpr('-', other, self)

    def __mul__(self, other):
        return UnoExpr('*', self, other)

    def __rmul__(self, other):
        return UnoExpr('*', other, self)

    def __truediv__(self, other):
        return UnoExpr('/', self, other)

    def __rtruediv__(self, other):
        return UnoExpr('/', other, self)

    def __pow__(self, other):
        return UnoExpr('**', self, other)

    def __neg__(self):
        return UnoExpr('*', -1, self)

    def __pos__(self):
        return self

    def __abs__(self):
        return UnoExpr('abs', self)

    def log(self):
        return UnoExpr('log', self)

    def sin(self):
        return UnoExpr('sin', self)

    def cos(self):
        return UnoExpr('cos', self)

    def exp(self):
        return UnoExpr('exp', self)

    def sqrt(self):
        return UnoExpr('sqrt', self)

    def tan(self):
        return UnoExpr('tan', self)

    def log10(self):
        return UnoExpr('log10', self)

    def sinh(self):
        return UnoExpr('sinh', self)

    def cosh(self):
        return UnoExpr('cosh', self)

    def tanh(self):
        return UnoExpr('tanh', self)

    def asin(self):
        return UnoExpr('asin', self)

    def acos(self):
        return UnoExpr('acos', self)

    def atan(self):
        return UnoExpr('atan', self)

    def __le__(self, other):
        return ('<=', self, other)

    def __ge__(self, other):
        return ('>=', self, other)

    def __eq__(self, other):
        return ('==', self, other)

    def __lt__(self, other):
        return ('<', self, other)

    def __gt__(self, other):
        return ('>', self, other)


class UnoExpr:
    """Symbolic expression tree for Uno interface."""

    def __init__(self, op, left, right=None):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        if self.right is None:
            return f"UnoExpr({self.op}, {self.left})"
        return f"UnoExpr({self.op}, {self.left}, {self.right})"

    def __add__(self, other):
        return UnoExpr('+', self, other)

    def __radd__(self, other):
        return UnoExpr('+', other, self)

    def __sub__(self, other):
        return UnoExpr('-', self, other)

    def __rsub__(self, other):
        return UnoExpr('-', other, self)

    def __mul__(self, other):
        return UnoExpr('*', self, other)

    def __rmul__(self, other):
        return UnoExpr('*', other, self)

    def __truediv__(self, other):
        return UnoExpr('/', self, other)

    def __rtruediv__(self, other):
        return UnoExpr('/', other, self)

    def __pow__(self, other):
        return UnoExpr('**', self, other)

    def __neg__(self):
        return UnoExpr('*', -1, self)

    def __pos__(self):
        return self

    def __abs__(self):
        return UnoExpr('abs', self)

    def log(self):
        return UnoExpr('log', self)

    def sin(self):
        return UnoExpr('sin', self)

    def cos(self):
        return UnoExpr('cos', self)

    def exp(self):
        return UnoExpr('exp', self)

    def sqrt(self):
        return UnoExpr('sqrt', self)

    def tan(self):
        return UnoExpr('tan', self)

    def log10(self):
        return UnoExpr('log10', self)

    def sinh(self):
        return UnoExpr('sinh', self)

    def cosh(self):
        return UnoExpr('cosh', self)

    def tanh(self):
        return UnoExpr('tanh', self)

    def asin(self):
        return UnoExpr('asin', self)

    def acos(self):
        return UnoExpr('acos', self)

    def atan(self):
        return UnoExpr('atan', self)

    def __le__(self, other):
        return ('<=', self, other)

    def __ge__(self, other):
        return ('>=', self, other)

    def __eq__(self, other):
        return ('==', self, other)

    def __lt__(self, other):
        return ('<', self, other)

    def __gt__(self, other):
        return ('>', self, other)


def evaluate_expr(expr, x):
    """Evaluate an expression tree at a point x."""
    if isinstance(expr, UnoVar):
        # Uno can probe marginally outside the variable box (e.g. -7e-9 for a
        # variable bounded below by 0); clamp the read so domain-restricted
        # operations such as (x[0]/g) ** 0.5 never produce NaN.
        val = x[expr.index]
        if val < expr.lb:
            return expr.lb
        if val > expr.ub:
            return expr.ub
        return val
    elif isinstance(expr, (int, float)):
        return float(expr)
    elif isinstance(expr, UnoExpr):
        left_val = evaluate_expr(expr.left, x)
        right_val = evaluate_expr(expr.right, x) if expr.right is not None else None
        match expr.op:
            case '+':
                return left_val + right_val
            case '-':
                return left_val - right_val
            case '*':
                return left_val * right_val
            case '/':
                return left_val / right_val
            case '**':
                return left_val ** right_val
            case 'abs':
                return abs(left_val)
            case 'log':
                import math
                return math.log(left_val)
            case 'sin':
                import math
                return math.sin(left_val)
            case 'cos':
                import math
                return math.cos(left_val)
            case 'exp':
                import math
                return math.exp(left_val)
            case 'sqrt':
                import math
                return math.sqrt(left_val)
            case 'tan':
                import math
                return math.tan(left_val)
            case 'log10':
                import math
                return math.log10(left_val)
            case 'sinh':
                import math
                return math.sinh(left_val)
            case 'cosh':
                import math
                return math.cosh(left_val)
            case 'tanh':
                import math
                return math.tanh(left_val)
            case 'asin':
                import math
                return math.asin(left_val)
            case 'acos':
                import math
                return math.acos(left_val)
            case 'atan':
                import math
                return math.atan(left_val)
    elif isinstance(expr, tuple):
        left_val = evaluate_expr(expr[1], x)
        right_val = evaluate_expr(expr[2], x)
        return left_val - right_val
    return float(expr)


def compute_gradient(expr, x, var_index, epsilon=1e-7, bounds=None):
    """Compute partial derivative of expr with respect to variable at var_index.

    Central differences are used while both probe points (x +/- epsilon) stay
    inside the variable bounds; otherwise a one-sided difference is taken from
    the clamped base point. Probing outside the bounds would evaluate the
    expression outside its domain (e.g. a negative x[0] makes
    (x[0]/g) ** 0.5 return NaN and abort Uno with an algorithmic error).
    """
    lb, ub = bounds if bounds is not None else (-INFINITY, INFINITY)

    x_base = float(x[var_index])
    if x_base < lb:
        x_base = lb
    elif x_base > ub:
        x_base = ub

    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[var_index] = x_base + epsilon
    x_minus[var_index] = x_base - epsilon

    central = (x_base - epsilon >= lb) and (x_base + epsilon <= ub)
    forward = x_base + epsilon <= ub
    backward = x_base - epsilon >= lb

    if central:
        f_plus = evaluate_expr(expr, x_plus)
        f_minus = evaluate_expr(expr, x_minus)
        return (f_plus - f_minus) / (2 * epsilon)
    if forward:
        x_minus[var_index] = x_base
        f_plus = evaluate_expr(expr, x_plus)
        f_minus = evaluate_expr(expr, x_minus)
        return (f_plus - f_minus) / epsilon
    if backward:
        x_plus[var_index] = x_base
        f_plus = evaluate_expr(expr, x_plus)
        f_minus = evaluate_expr(expr, x_minus)
        return (f_plus - f_minus) / epsilon
    # variable is fixed (lb == ub): no variation to measure
    return 0.0


def expr_to_gradient_func(expr, n, bounds=None):
    """Create a gradient function for an expression with respect to n variables.

    bounds is an optional sequence of (lb, ub) pairs, one per variable index,
    used to keep the finite-difference probes inside the feasible domain.
    """
    def gradient_func(x, gradient):
        x_arr = np.array(x, dtype=float)
        for i in range(n):
            bnd = bounds[i] if bounds is not None else None
            gradient[i] = compute_gradient(expr, x_arr, i, bounds=bnd)
    return gradient_func


def collect_vars(expr):
    """Collect all UnoVar instances in an expression tree."""
    if isinstance(expr, UnoVar):
        return {expr.index: expr}
    elif isinstance(expr, UnoExpr):
        vars_left = collect_vars(expr.left) if expr.left is not None else {}
        vars_right = collect_vars(expr.right) if expr.right is not None else {}
        vars_left.update(vars_right)
        return vars_left
    elif isinstance(expr, tuple):
        return collect_vars(expr[1])
    return {}


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    lb = variable_bound[0] if variable_bound[0] is not None else -INFINITY
    ub = variable_bound[1] if variable_bound[1] is not None else +INFINITY

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    match variable_type:

        case 'pvar':

            if variable_dim == 0:
                idx = model_object.number_variables
                model_object.number_variables += 1
                model_object.variables_lower_bounds.append(lb)
                model_object.variables_upper_bounds.append(ub)
                generated_variable = UnoVar(idx, variable_name, lb, ub)

            else:
                if len(variable_dim) == 1:
                    generated_variable = {}
                    for key in variable_dim[0]:
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)
                else:
                    generated_variable = {}
                    for key in sets(*variable_dim):
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)

        case 'bvar':

            if variable_dim == 0:
                idx = model_object.number_variables
                model_object.number_variables += 1
                model_object.variables_lower_bounds.append(0)
                model_object.variables_upper_bounds.append(1)
                generated_variable = UnoVar(idx, variable_name, 0, 1)

            else:
                if len(variable_dim) == 1:
                    generated_variable = {}
                    for key in variable_dim[0]:
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(0)
                        model_object.variables_upper_bounds.append(1)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", 0, 1)
                else:
                    generated_variable = {}
                    for key in sets(*variable_dim):
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(0)
                        model_object.variables_upper_bounds.append(1)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", 0, 1)

        case 'ivar':

            if variable_dim == 0:
                idx = model_object.number_variables
                model_object.number_variables += 1
                model_object.variables_lower_bounds.append(lb)
                model_object.variables_upper_bounds.append(ub)
                generated_variable = UnoVar(idx, variable_name, lb, ub)

            else:
                if len(variable_dim) == 1:
                    generated_variable = {}
                    for key in variable_dim[0]:
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)
                else:
                    generated_variable = {}
                    for key in sets(*variable_dim):
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)

        case 'fvar':

            if variable_dim == 0:
                idx = model_object.number_variables
                model_object.number_variables += 1
                model_object.variables_lower_bounds.append(lb)
                model_object.variables_upper_bounds.append(ub)
                generated_variable = UnoVar(idx, variable_name, lb, ub)

            else:
                if len(variable_dim) == 1:
                    generated_variable = {}
                    for key in variable_dim[0]:
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)
                else:
                    generated_variable = {}
                    for key in sets(*variable_dim):
                        idx = model_object.number_variables
                        model_object.number_variables += 1
                        model_object.variables_lower_bounds.append(lb)
                        model_object.variables_upper_bounds.append(ub)
                        generated_variable[key] = UnoVar(idx, f"{variable_name}{key}", lb, ub)

    return generated_variable
