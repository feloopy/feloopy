# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Symbolic model capture used by the automatic decomposition methods.

The automatic Benders, Lagrangian and column-generation runners execute the
user environment against a lightweight stand-in model.  The captured variables,
constraints and objective are later replayed onto real solver models without
touching a solver backend during capture.

Classes:
    _DecomposeExpression    -- placeholder expression used while building an
                               automatic master.
    _AutomaticCaptureModel  -- stand-in model collecting variables, constraints
                               and objectives.

Functions:
    _decompose_placeholders      -- placeholder grid for a dimension spec.
    _materialize_automatic_model -- replay a captured model onto a real model.
"""

import itertools as it

import numpy as np

from ...helpers.containers import as_var_group

from ...operators.fix_operators import fix_dims
from ...classes.linearization import (
    _AutomaticLinearExpression,
    _AutomaticLinearVariable,
    _AutomaticLinearConstraint,
    _AutomaticVariableArray,
)


class _DecomposeExpression:
    """Small expression wrapper used while building an automatic master."""

    def __init__(self, value=0, contains_subproblem_var=True):
        self.value = value
        self.contains_subproblem_var = contains_subproblem_var

    def _coerce(self, other):
        if isinstance(other, _DecomposeExpression):
            return other.value, other.contains_subproblem_var
        return other, False

    def _binary(self, other, operation, reflected=False):
        other_value, other_has_subproblem = self._coerce(other)
        try:
            value = operation(other_value, self.value) if reflected else operation(self.value, other_value)
        except Exception:
            value = self.value if not reflected else other_value
        return _DecomposeExpression(
            value,
            self.contains_subproblem_var or other_has_subproblem,
        )

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary(other, lambda a, b: a + b, reflected=True)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: a - b, reflected=True)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary(other, lambda a, b: a * b, reflected=True)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary(other, lambda a, b: a / b, reflected=True)

    def __neg__(self):
        return _DecomposeExpression(-self.value, self.contains_subproblem_var)

    def __le__(self, other):
        return self._constraint(other)

    def __ge__(self, other):
        return self._constraint(other)

    def __eq__(self, other):
        return self._constraint(other)

    def _constraint(self, other):
        return _DecomposeExpression(0, True)


def _decompose_placeholders(dim):
    if dim in (None, 0, []):
        return _DecomposeExpression()
    normalized_dim = fix_dims(dim)
    dimensions = [list(d) for d in normalized_dim]
    if len(dimensions) == 1:
        return {index: _DecomposeExpression() for index in dimensions[0]}
    return {
        index: _DecomposeExpression()
        for index in it.product(*dimensions)
    }


class _AutomaticCaptureModel:
    def __init__(self):
        self.variables = {}
        self.objectives = []
        self.constraints = []

    @staticmethod
    def sets(*args):
        """Cartesian-product helper mirroring ``model.sets``."""
        if len(args) == 1:
            return args[0]
        return it.product(*args)

    @staticmethod
    def sum(input, domain_tuple=None):
        """Plain ``sum`` mirroring ``MathAPI.sum``'s default branch."""
        return sum(input)

    def _variable(self, kind, name, dim=0, bound=None):
        indices = [None]
        if dim not in (None, 0, []):
            normalized_dim = fix_dims(dim)
            if isinstance(normalized_dim, set):
                # set-dim variables are one-dimensional, keyed by each
                # element of the set — the variable generators wrap a set
                # as [set] (tuples for a set of index tuples, scalars
                # otherwise); product-of-elements would explode here
                indices = list(normalized_dim)
            else:
                dimensions = [list(item) for item in normalized_dim]
                indices = dimensions[0] if len(dimensions) == 1 else list(it.product(*dimensions))
        values = _AutomaticVariableArray()
        for index in indices:
            key = (kind, name, index)
            values[index] = _AutomaticLinearVariable(key, self)
        self.variables[(kind, name)] = {
            'dim': dim,
            'bound': bound,
            'values': values,
        }
        return values[None] if indices == [None] else values

    def linearize_square(self, left, right):
        if (left.constant != 0 or right.constant != 0 or
                left.terms != right.terms or len(left.terms) != 1):
            raise ValueError("Automatic Benders supports bounded variable squares only")
        key = next(iter(left.terms))
        spec = self.variables[(key[0], key[1])]
        lower, upper = spec['bound']
        lower = 0.0 if lower is None else float(lower)
        upper = 1e3 if upper is None else float(upper)
        aux_name = f"_benders_square_{len(self.variables)}"
        aux = self._variable('fvar', aux_name, 0, [0, max(lower * lower, upper * upper)])
        x = left
        self.con(aux - ((2 * lower) * x - lower * lower) >= 0)
        self.con(aux - ((2 * upper) * x - upper * upper) >= 0)
        self.con(aux - ((lower + upper) * x - lower * upper) <= 0)
        return aux

    def fvar(self, name='x', dim=0, bound=[None, None], **kwargs):
        return self._variable('fvar', name, dim, bound)

    def pvar(self, name='x', dim=0, bound=[0, None], **kwargs):
        return self._variable('pvar', name, dim, bound)

    def ivar(self, name='x', dim=0, bound=[None, None], **kwargs):
        return self._variable('ivar', name, dim, bound)

    def bvar(self, name='x', dim=0, bound=[0, 1], **kwargs):
        return self._variable('bvar', name, dim, bound)

    def obj(self, expression=0, direction=None, label=None):
        self.objectives.append((expression, direction, label))

    def maximize(self, expression=0, label=None):
        self.objectives.append((expression, 'max', label))

    def minimize(self, expression=0, label=None):
        self.objectives.append((expression, 'min', label))

    def con(self, expression, name=None, **kwargs):
        if isinstance(expression, _AutomaticLinearConstraint):
            self.constraints.append((expression, name))
            return
        # Data guards (e.g. ``len(S) >= 2``) evaluate to a constant before
        # they reach us: mirror the solution sanitizer — vacuous True / None
        # and bare numbers add no row, while a constant False makes the
        # model infeasible by construction.
        if expression is None or isinstance(expression, (bool, np.bool_)):
            if expression is not None and not bool(expression):
                from ...helpers.error import ConstantConstraintError
                raise ConstantConstraintError(
                    "A constraint%s contains no variables and evaluates to "
                    "False, so the model can never be satisfied. Check the "
                    "data feeding it."
                    % ((" (%s)" % name) if name else ""))
            return
        if isinstance(expression, (int, float, np.integer, np.floating)):
            return
        # ``mdl.con([...])`` / ``mdl.con(x for x in ...)``: a sequence of
        # constraints, labeled like the real model (name + index when a
        # scalar name is given, element-wise when a list of names is)
        if isinstance(expression, (list, tuple, set)) or (
                hasattr(expression, '__iter__') and
                not isinstance(expression, (str, bytes))):
            elements = list(expression)
            if isinstance(name, (list, tuple)):
                names = list(name)
                names += [None] * max(0, len(elements) - len(names))
            elif name is None:
                names = [None] * len(elements)
            else:
                names = ["%s%s" % (name, i) for i in range(len(elements))]
            for element, element_name in zip(elements, names):
                self.con(element, name=element_name, **kwargs)
            return
        raise ValueError("boost mode requires linear constraints")


def _capture_from_model_fn(model_fn):
    """Run ``model_fn`` against a capture model and return the structure."""
    captured = _AutomaticCaptureModel()
    model_fn(captured)
    return captured


def _materialize_automatic_model(captured, m, objective_scope):
    native = {}
    for (kind, name), spec in captured.variables.items():
        creator = getattr(m, kind)
        value = creator(name=name, dim=spec['dim'], bound=spec['bound'])
        if isinstance(value, dict):
            values = value
        else:
            # dict / dict-like container (pyomo IndexedVar, COPT
            # tupledict) / scalar — as_var_group normalizes all three
            # with guarded attribute access.
            values = as_var_group(value)
        for index, symbolic in spec['values'].items():
            native[symbolic.key] = values[index]

    def translate(expression):
        return expression.evaluate(native)

    for cidx, (constraint, label) in enumerate(captured.constraints):
        has_continuous = any(key[0] not in ('ivar', 'bvar')
                             for key in constraint.expression.terms)
        if objective_scope == 'master' and has_continuous:
            continue
        expression = translate(constraint.expression)
        if label is None:
            label = f'_auto_constraint_{cidx}'
        if constraint.sense == '<=':
            m.con(expression <= 0, name=label)
        elif constraint.sense == '>=':
            m.con(expression >= 0, name=label)
        else:
            m.con(expression == 0, name=label)

    expression, direction, label = captured.objectives[0]
    if objective_scope == 'master':
        expression = _AutomaticLinearExpression(
            expression.constant,
            {key: value for key, value in expression.terms.items()
             if key[0] in ('ivar', 'bvar')})
    elif objective_scope == 'subproblem':
        expression = _AutomaticLinearExpression(
            expression.constant,
            {key: value for key, value in expression.terms.items()
             if key[0] not in ('ivar', 'bvar')})
    translated = translate(expression)
    if isinstance(translated, (int, float)):
        dummy = m.fvar('_benders_dummy_obj', bound=[0, 0])
        m.obj(translated + 0 * dummy, direction=direction, label=label)
    else:
        m.obj(translated, direction=direction, label=label)
    return m
