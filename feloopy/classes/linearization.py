# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Linearization module.

Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

import math as mt
from typing import List, Optional, Tuple, Callable, Union

import numpy as np

from ..helpers.cache import _PersistentLRUCache
from .multidim_variable import MultidimVariable
from .special_constraint import Expression

__all__ = [
    "LinearizationClass",
    "LinearizationProxy",
    "needs_auto_linearization",
    "unwrap_proxies",
    "BIG_M",
    "DEFAULT_PWL_BREAKPOINTS",
    "MINLP_SOLVERS",
    "clear_linearization_cache",
    "linearization_cache_stats",
]

# Pre-allocated list for O(1) batch constraint label appending
_AL_NONE_LIST = [None] * 100000

"""
Auto-Linearization Module

Provides transparent linearization of all nonlinear expressions for solver
interfaces that only support LP/MILP (no MINLP/NLP).

Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

MINLP_SOLVERS = {
    'scip', 'bonmin', 'couenne', 'ipopt', 'cyipopt', 'uno',
    'baron', 'knitro', 'minotaur', 'octeract', 'alphaecp',
    'antigone', 'lindoglobal', 'minlp', 'mindtpy', 'gdpopt',
    'juniper', 'pajarito', 'pavito', 'eago', 'alpine',
    'msnlp', 'minlp', 'decis', 'dicopt', 'snopt', 'minos',
    'conopt', 'path', 'pathnlp', 'nlpec', 'multistart',
    'trustregion', 'mpec-minlp', 'mpec-nlp', 'alpine',
    'raposa', 'madnlp', 'percival', 'nlopt', 'maingo',
    'manopt', 'cosmo', 'artelys_knitro', 'tulip',
}
def needs_auto_linearization(interface_name: str, solver_name: str = '') -> bool:
    if solver_name in MINLP_SOLVERS:
        return False
    return True
BIG_M = 1e6
DEFAULT_PWL_BREAKPOINTS = 20

# ── Search-result cache for linearized runs ─────────────────────────────

_LINEARIZATION_CACHE = _PersistentLRUCache(maxsize=512)


def _stable_repr(obj, depth=0):
    """``repr`` that is stable across processes.

    Code objects and arbitrary objects are rendered without their ``0x``
    addresses (which change every run and made cache keys unusable across
    sessions), nested code objects recurse structurally, and numpy arrays
    are content-hashed instead of relying on ``str``'s element truncation.
    """
    import re
    import hashlib

    if depth > 8:
        return '...'
    if obj is None or isinstance(obj, (bool, int, float, complex, str,
                                       bytes, bytearray)):
        return repr(obj)
    if isinstance(obj, type):
        return f'<type {obj.__module__}.{obj.__qualname__}>'
    if hasattr(obj, 'co_code') and hasattr(obj, 'co_consts'):
        return (f'<code {obj.co_name} {obj.co_code.hex()} '
                f'{_stable_repr(obj.co_consts, depth + 1)}>')
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            digest = _stable_repr(obj.tolist(), depth + 1)
        else:
            digest = hashlib.sha1(
                np.ascontiguousarray(obj).tobytes()).hexdigest()
        return f'<ndarray {obj.shape} {obj.dtype} {digest}>'
    if isinstance(obj, np.generic):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        inner = ','.join(_stable_repr(v, depth + 1) for v in obj)
        kind = 'list' if isinstance(obj, list) else 'tuple'
        return f'<{kind} [{inner}]>'
    if isinstance(obj, (set, frozenset)):
        inner = ','.join(sorted(_stable_repr(v, depth + 1) for v in obj))
        return f'<set {{{inner}}}>'
    if isinstance(obj, dict):
        inner = ','.join(
            f'{_stable_repr(k, depth + 1)}:{_stable_repr(v, depth + 1)}'
            for k, v in sorted(obj.items(),
                               key=lambda kv: _stable_repr(kv[0], depth + 1)))
        return f'<dict {{{inner}}}>'
    if callable(obj) and hasattr(obj, '__code__'):
        code = obj.__code__
        parts = [f'<func {getattr(obj, "__qualname__", getattr(obj, "__name__", "?"))}',
                 _stable_repr(code, depth + 1)]
        cells = getattr(obj, '__closure__', None)
        if cells:
            cell_reprs = []
            for cell in cells:
                try:
                    cell_reprs.append(_stable_repr(cell.cell_contents, depth + 1))
                except ValueError:
                    cell_reprs.append('<empty>')
            parts.append('closure=' + ','.join(cell_reprs))
        return ' '.join(parts) + '>'
    try:
        text = repr(obj)
    except Exception:
        text = f'<unreprable {type(obj).__module__}.{type(obj).__name__}>'
    return re.sub(r'0x[0-9a-fA-F]+', '0x?', text)


def _make_cache_key(environment, args, kwargs, interface, solver, method):
    """Generate a cache key from the environment function and its configuration.

    The key is stable across processes: function bytecode/constants are
    rendered structurally (no addresses), closures contribute their cell
    contents, and arguments go through :func:`_stable_repr`, so entries
    recorded in one session stay retrievable in the next.
    """
    import hashlib

    parts = []

    if environment is not None and hasattr(environment, '__code__'):
        code = environment.__code__
        parts.append(str(code.co_code))
        parts.append(str(code.co_names))
        parts.append(_stable_repr(code.co_consts))
        parts.append(environment.__name__)
        closure = getattr(environment, '__closure__', None)
        if closure:
            cell_reprs = []
            for cell in closure:
                try:
                    cell_reprs.append(_stable_repr(cell.cell_contents))
                except ValueError:
                    cell_reprs.append('<empty>')
            parts.append('closure=' + ','.join(cell_reprs))

    parts.append(_stable_repr(args))

    non_model_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ('verbose', 'progress', 'report', 'debug',
                                     'should_run', 'cache_key', 'email',
                                     'time_limit', 'cpu_threads', 'absolute_gap',
                                     'relative_gap', 'track_history')}
    sorted_kwargs = sorted(non_model_kwargs.items())
    parts.append(_stable_repr(sorted_kwargs))

    parts.append(str(method))
    parts.append(str(interface))
    parts.append(str(solver))

    key_str = '|'.join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _al_proxy_key(*args):
    """Build a hashable cache key from LinearizationProxy operands."""
    parts = []
    for a in args:
        if hasattr(a, '_native'):
            parts.append(id(a._native))
        else:
            parts.append(repr(a))
    return tuple(parts)


def clear_linearization_cache(key=None, environment=None, method='exact'):
    """Clear the linearization cache.

    Parameters
    ----------
    key : str, optional
        If provided, only clear the cache entry for this key.
    environment : callable, optional
        If provided, clear every entry recorded for this environment
        function.  Matching uses the bytecode/constant metadata stored with
        each entry rather than a recomputed key, so it works regardless of
        the arguments the search was called with.
    method : str, optional
        Method to match (default 'exact').  Entries recorded without a
        method (older entries) are matched either way.
    """
    if key is not None:
        _LINEARIZATION_CACHE.pop(key, None)
    elif environment is not None:
        if not hasattr(environment, '__code__'):
            return
        code = environment.__code__
        target = (environment.__name__, code.co_code, code.co_consts)
        for entry_key, entry in _LINEARIZATION_CACHE.items():
            if not isinstance(entry, dict) or not entry.get('func_code'):
                continue
            stored = (entry.get('func_name'), entry.get('func_code'),
                      entry.get('func_consts'))
            if stored != target:
                continue
            if entry.get('method', method) != method:
                continue
            _LINEARIZATION_CACHE.pop(entry_key, None)
    else:
        _LINEARIZATION_CACHE.clear()


def linearization_cache_stats():
    """Return cache statistics: size, hits, misses, hit rate."""
    return _LINEARIZATION_CACHE.stats()


class LinearizationProxy:
    """
    Wraps a native solver variable or linear expression to intercept
    arithmetic operations and automatically linearize nonlinear terms.
    """

    __slots__ = ('_native', '_model', '_name', '_lb', '_ub')

    def __init__(self, native_var, model, name: str = '',
                 lb: Optional[float] = None, ub: Optional[float] = None):
        object.__setattr__(self, '_native', native_var)
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_lb', lb)
        object.__setattr__(self, '_ub', ub)

    @property
    def native(self):
        return self._native

    # ── Multiplication ──────────────────────────────────────────────────

    def __mul__(self, other):
        if isinstance(other, LinearizationProxy):
            method = getattr(self._model, '_lin_method', 'mccormick')
            if method == 'sos2':
                return self._model._auto_lin_product_sos2(self, other)
            return self._model._auto_lin_product(self, other)
        if isinstance(other, (int, float)):
            if self._lb is None or self._ub is None:
                return self._model._auto_lin_scalar_mul(self._native, other)
            if other >= 0:
                lb, ub = self._lb * other, self._ub * other
            else:
                lb, ub = self._ub * other, self._lb * other
            return self._model._auto_lin_scalar_mul(self._native, other, lb, ub)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            if self._lb is None or self._ub is None:
                return self._model._auto_lin_scalar_mul(self._native, other)
            if other >= 0:
                lb, ub = self._lb * other, self._ub * other
            else:
                lb, ub = self._ub * other, self._lb * other
            return self._model._auto_lin_scalar_mul(self._native, other, lb, ub)
        return NotImplemented

    # ── Addition / Subtraction ──────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, LinearizationProxy):
            lb = None if self._lb is None or other._lb is None else self._lb + other._lb
            ub = None if self._ub is None or other._ub is None else self._ub + other._ub
            return self._model._auto_lin_expr(self._native + other._native, lb, ub)
        if isinstance(other, (int, float)):
            lb = None if self._lb is None else self._lb + other
            ub = None if self._ub is None else self._ub + other
            return self._model._auto_lin_expr(self._native + other, lb, ub)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            lb = None if self._lb is None else other + self._lb
            ub = None if self._ub is None else other + self._ub
            return self._model._auto_lin_expr(other + self._native, lb, ub)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, LinearizationProxy):
            lb = None if self._lb is None or other._ub is None else self._lb - other._ub
            ub = None if self._ub is None or other._lb is None else self._ub - other._lb
            return self._model._auto_lin_expr(self._native - other._native, lb, ub)
        if isinstance(other, (int, float)):
            lb = None if self._lb is None else self._lb - other
            ub = None if self._ub is None else self._ub - other
            return self._model._auto_lin_expr(self._native - other, lb, ub)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            lb = None if self._ub is None else other - self._ub
            ub = None if self._lb is None else other - self._lb
            return self._model._auto_lin_expr(other - self._native, lb, ub)
        return NotImplemented

    def __neg__(self):
        lb = None if self._ub is None else -self._ub
        ub = None if self._lb is None else -self._lb
        return self._model._auto_lin_expr(-self._native, lb, ub)

    # ── Power ───────────────────────────────────────────────────────────

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            n = int(other)
            if n == 0:
                return self._model._auto_lin_expr(self._native * 0 + 1.0, 1.0, 1.0)
            if n == 1:
                return self._model._auto_lin_expr(self._native, self._lb, self._ub)
            method = getattr(self._model, '_lin_method', 'mccormick')
            if method == 'sos2':
                result_proxy = self._model._auto_lin_product_sos2(self, self)
                for _ in range(n - 2):
                    result_proxy = self._model._auto_lin_product_sos2(result_proxy, self)
                return result_proxy
            result_proxy = self._model._auto_lin_product(self, self)
            for _ in range(n - 2):
                result_proxy = self._model._auto_lin_product(result_proxy, self)
            return result_proxy
        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return self._model._auto_lin_func(
                lambda v: other ** v, self._native,
                f"rpow_{id(self)}", self._lb, self._ub)
        return NotImplemented

    # ── Division ────────────────────────────────────────────────────────

    def __truediv__(self, other):
        if isinstance(other, LinearizationProxy):
            den_lb = other._lb
            den_ub = other._ub
            if den_lb is not None and den_ub is not None:
                if den_lb == 0 and den_ub == 1:
                    return self._model._auto_lin_div_binary(self, other)
                num_lb = self._lb
                num_ub = self._ub
                if (den_lb >= 1 and den_ub <= 10 and den_ub == int(den_ub)
                        and num_lb is not None and num_ub is not None):
                    return self._model._auto_lin_div_small_int(
                        self, other, int(den_ub))
                if den_lb > 0 and num_lb is not None and num_ub is not None:
                    if num_ub <= den_ub:
                        return self._model._auto_lin_div_bounded(self, other)
            method = getattr(self._model, '_lin_method', 'mccormick')
            if method == 'taylor':
                return self._model._auto_lin_div_taylor(self, other)
            return self._model._auto_lin_div(self, other)
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return self._model._auto_lin_scalar_mul(self._native, 1.0 / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self, LinearizationProxy):
                den_lb = self._lb
                den_ub = self._ub
                if den_lb is not None and den_ub is not None:
                    if den_lb >= 1 and den_ub <= 20 and den_ub == int(den_ub):
                        return self._model._auto_lin_rdiv_small_int(other, self, int(den_ub))
            method = getattr(self._model, '_lin_method', 'mccormick')
            if method == 'taylor':
                return self._model._auto_lin_rdiv_taylor(other, self)
            return self._model._auto_lin_rdiv(other, self)
        return NotImplemented

    # ── Absolute value ──────────────────────────────────────────────────

    def __abs__(self):
        method = getattr(self._model, '_lin_method', 'mccormick')
        if method == 'sos2':
            return self._model._auto_lin_abs_exact(self)
        return self._model._auto_lin_abs(self)

    # ── Numpy ufunc dispatch ────────────────────────────────────────────

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        proxy_inputs = [isinstance(x, LinearizationProxy) for x in inputs]

        if ufunc in (np.multiply,):
            if sum(proxy_inputs) == 1:
                idx = proxy_inputs.index(True)
                proxy = inputs[idx]
                scalar = inputs[1 - idx]
                if isinstance(scalar, (int, float, np.floating, np.integer)):
                    return self._model._auto_lin_scalar_mul(proxy._native, scalar)
            if all(proxy_inputs) and len(inputs) == 2:
                lin_method = getattr(self._model, '_lin_method', 'mccormick')
                if lin_method == 'sos2':
                    return self._model._auto_lin_product_sos2(inputs[0], inputs[1])
                return self._model._auto_lin_product(inputs[0], inputs[1])
            return NotImplemented

        if ufunc in (np.add,):
            if sum(proxy_inputs) >= 1:
                natives = [x._native if isinstance(x, LinearizationProxy) else x for x in inputs]
                result = natives[0]
                for n in natives[1:]:
                    result = result + n
                return self._model._auto_lin_expr(result)
            return NotImplemented

        if ufunc in (np.subtract,):
            if sum(proxy_inputs) >= 1:
                natives = [x._native if isinstance(x, LinearizationProxy) else x for x in inputs]
                result = natives[0]
                for n in natives[1:]:
                    result = result - n
                return self._model._auto_lin_expr(result)
            return NotImplemented

        if ufunc in (np.negative,):
            if proxy_inputs[0]:
                p = inputs[0]
                lb = None if p._ub is None else -p._ub
                ub = None if p._lb is None else -p._lb
                return self._model._auto_lin_expr(-p._native, lb, ub)
            return NotImplemented

        _UFUNC_MAP = {
            np.sqrt: 'sqrt',
            np.log: 'log',
            np.log2: 'log2',
            np.log10: 'log10',
            np.sin: 'sin',
            np.cos: 'cos',
            np.tan: 'tan',
            np.arcsin: 'arcsin',
            np.arccos: 'arccos',
            np.arctan: 'arctan',
            np.exp: 'exp',
            np.abs: 'abs',
        }

        if ufunc in _UFUNC_MAP:
            func_name = _UFUNC_MAP[ufunc]
            if func_name == 'abs':
                lin_method = getattr(self._model, '_lin_method', 'mccormick')
                if lin_method == 'sos2':
                    return self._model._auto_lin_abs_exact(self)
                return self._model._auto_lin_abs(self)
            proxy_input = inputs[0] if isinstance(inputs[0], LinearizationProxy) else None
            if proxy_input is not None:
                return self._model._auto_lin_func(
                    getattr(np, func_name), proxy_input._native,
                    f"ufunc_{func_name}_{id(proxy_input)}",
                    proxy_input._lb, proxy_input._ub)

        return NotImplemented

    # ── Comparison operators ────────────────────────────────────────────

    def _coerce(self, other):
        import numpy as np
        if isinstance(other, (np.integer,)):
            return int(other)
        if isinstance(other, (np.floating,)):
            return float(other)
        return other

    def __le__(self, other):
        if isinstance(other, LinearizationProxy):
            return self._native <= other._native
        if hasattr(other, '_resolve'):
            other = other._resolve()
        return self._native <= self._coerce(other)

    def __ge__(self, other):
        if isinstance(other, LinearizationProxy):
            return self._native >= other._native
        if hasattr(other, '_resolve'):
            other = other._resolve()
        return self._native >= self._coerce(other)

    def __eq__(self, other):
        if isinstance(other, LinearizationProxy):
            return self._native == other._native
        if hasattr(other, '_resolve'):
            other = other._resolve()
        return self._native == self._coerce(other)

    def __lt__(self, other):
        if isinstance(other, LinearizationProxy):
            return self._native < other._native
        if hasattr(other, '_resolve'):
            other = other._resolve()
        return self._native < self._coerce(other)

    def __gt__(self, other):
        if isinstance(other, LinearizationProxy):
            return self._native > other._native
        if hasattr(other, '_resolve'):
            other = other._resolve()
        return self._native > self._coerce(other)

    def __call__(self):
        return self._model.get_variable(self)

    def __getitem__(self, key):
        native_sub = self._native[key]
        return LinearizationProxy(native_sub, self._model, f"{self._name}[{key}]", self._lb, self._ub)

    def __repr__(self):
        return f"LinProxy({self._name})"

def unwrap_proxies(obj):
    if isinstance(obj, LinearizationProxy):
        return obj.native
    if isinstance(obj, dict):
        return {k: unwrap_proxies(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(unwrap_proxies(item) for item in obj)
    return obj

class LinearizationClass:
    """All linearization methods for the model class."""

    # ── Public linearization API ──────────────────────────────────────


    def lin_piecewise(self, name: str, var, slopes: List[float], intercepts: List[float], breakpoints: List[float]) -> Expression: 
        """
        Implements a piecewise linear function in the context of mathematical programming.

        Parameters:
            slopes (List[float]): A list of slopes for each piece of the function.
            intercepts (List[float]): A list of intercepts for each piece of the function.
            breakpoints (List[float]): A list of breakpoints that define the domain of each piece of the function.

        Returns:
            Expression: The piecewise linear function represented as a sum of linear functions, each multiplied by a binary variable.

        Note:
            The function is represented as a sum of linear functions, each multiplied by a binary variable that indicates whether that piece of the function is active. The constraints ensure that only one piece is active at any point in the domain.
        """
        try:
            self.features['indicators'].append(self.features['indicators'][-1] + 1)
        except:
            self.features['indicators'] = [0]

        has_negative = any(b < 0 for b in breakpoints)
        if has_negative:
            x = self._lin_fvar(name=f"val_{name}_{self.features['indicators'][-1]}", dim=[range(len(breakpoints)-1)])
        else:
            x = self._lin_pvar(name=f"val_{name}_{self.features['indicators'][-1]}",  dim=[range(len(breakpoints)-1)])
        y = self._lin_bvar(name=f"part_{name}_{self.features['indicators'][-1]}", dim=[range(len(breakpoints)-1)])

        for i in range(len(breakpoints) - 1):
            bp_lo = min(breakpoints[i], breakpoints[i+1])
            bp_hi = max(breakpoints[i], breakpoints[i+1])
            self._lin_con(bp_lo * y[i] <= x[i])
            self._lin_con(x[i] <= bp_hi * y[i])

        self._lin_con(sum(y[i] for i in range(len(breakpoints)-1)) == 1)    
        self._lin_con(var==sum(x[i] for i in range(len(breakpoints) - 1)))
        
        return sum(slopes[i] * x[i] + intercepts[i]* y[i] for i in range(len(breakpoints)-1))

    def lin_approx(self, name: str, f: Callable, var: MultidimVariable, bound: Tuple[float, float], num_breakpoints: int) -> Expression:
        
        """
        Implements a piecewise linear approximation of a non-linear function in the context of mathematical programming.

        Parameters:
            f (Callable): The non-linear function to be approximated.
            x (MultidimVariable): The variable of the non-linear function.
            bound (Tuple[float, float]): A tuple of two numbers representing the minimum and maximum values of x.
            num_breakpoints (int): The number of breakpoints to use in the approximation.

        Returns:
            Expression: The piecewise linear approximation of the non-linear function.
        """
        
        breakpoints = np.linspace(bound[0], bound[1], num_breakpoints)        
        slopes = [(f(breakpoints[i+1]) - f(breakpoints[i]))/(breakpoints[i+1]  -  breakpoints[i]) for i in range(len(breakpoints)-1)]          
        intercepts = [f(breakpoints[i]) - slopes[i] * breakpoints[i] for i in range(len(breakpoints)-1)]        

        return self.lin_piecewise(name,var,slopes, intercepts, breakpoints)

    def lin_abs_in_obj(self, expr: Expression, method: int = 0, dir_obj: Optional[str] = None) -> Expression:
        """
        Linearizes an |expr| expression inside the objective function.

        Parameters:
            expr (Expression): The absolute value expression to be linearized.
            method (int): The method to use for linearization.
                - method 0: Uses +2 pvars and +1 constraint (LP, for min only).
                - method 1: Uses +1 pvar, +1 bvar and +4 constraints (MILP, for min or max).
                - method 2: Uses +1 pvar and +1 constraint (LP, for min only).
            dir_obj (str): Deprecated, kept for API compatibility. Method 1 now works for both min and max.

        Returns:
            Expression: The linearized expression.

        Note:
            Method 0 and 2 are LP-relaxations valid only for minimization.
            Method 1 enforces z = |expr| exactly via McCormick envelope (requires binary variable).
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('abs', id(expr), method)
        if key in cache:
            return cache[key]

        if method == 0:
            try:
                self.features['abs_obj_lins'].append(self.features['abs_obj_lins'][-1] + 1)
                self.features['abs_obj_lins'].append(self.features['abs_obj_lins'][-1] + 1)
            except:
                self.features['abs_obj_lins'] = [0, 1]

            z1 = self._lin_pvar(f"abs_obj_lin{self.features['abs_obj_lins'][-1]}")
            z2 = self._lin_pvar(f"abs_obj_lin{self.features['abs_obj_lins'][-2]}")

            self._lin_con(expr == z1 - z2)

            result = z1 + z2
            cache[key] = result
            return result

        if method == 1:
            try:
                self.features['abs_obj_lins'].append(self.features['abs_obj_lins'][-1] + 1)
                self.features['abs_obj_lins'].append(self.features['abs_obj_lins'][-1] + 1)
            except:
                self.features['abs_obj_lins'] = [0, 1]

            z = self._lin_pvar(f"abs_obj_lin{self.features['abs_obj_lins'][-1]}")
            b = self._lin_bvar(f"abs_obj_bin{self.features['abs_obj_lins'][-2]}")

            self._lin_con(z >= expr)
            self._lin_con(z >= -expr)
            self._lin_con(z <= expr + 1e9 * b)
            self._lin_con(z <= -expr + 1e9 * (1 - b))
            cache[key] = z
            return z

        if method == 2:
            try:
                self.features['abs_obj_lins'].append(self.features['abs_obj_lins'][-1] + 1)
            except:
                self.features['abs_obj_lins'] = [0]
            z = self._lin_pvar(f"abs_obj_lin{self.features['abs_obj_lins'][-1]}")
            self._lin_con(expr + z >= 0)
            result = expr + 2 * z
            cache[key] = result
            return result

    def lin_max(self, input_list: List[Expression], type_max: str, ub_max: Union[float, None] = None) -> Expression:
        """
        Linearizes the max function.

        Parameters:
            input_list (List[Expression]): The list of expressions to be linearized.
            type_max (str): The type of variable to use for linearization.
            ub_max (Union[float, None]): The upper bound for the linearized expression (optional).

        Returns:
            Expression: The linearized expression.

        Note:
            The linearization is performed based on the type of variable chosen for linearization and the upper bound, if provided.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('max', tuple(id(x) for x in input_list), type_max, ub_max)
        if key in cache:
            return cache[key]

        if self.features['solution_method'] == 'exact':
            try:
                self.features['max_lins'].append(self.features['max_lins'][-1] + 1)
            except:
                self.features['max_lins'] = [0]

            if type_max == 'bvar':
                z = self._lin_bvar(f"max_lin{self.features['max_lins'][-1]}")
            elif type_max == 'ivar':
                z = self._lin_ivar(f"max_lin{self.features['max_lins'][-1]}")
            elif type_max == 'pvar':
                z = self._lin_pvar(f"max_lin{self.features['max_lins'][-1]}")
            elif type_max == 'fvar':
                z = self._lin_fvar(f"max_lin{self.features['max_lins'][-1]}")

            for item in input_list:
                self._lin_con(z >= item)
            if ub_max is not None:
                self._lin_con(z <= ub_max)
            cache[key] = z
            return z

    def lin_min(self, input_list: List[Expression], type_min: str, lb_min: Union[float, None] = None) -> Expression:
        """
        Linearizes the min function.

        Parameters:
            input_list (List[Expression]): The list of expressions to be linearized.
            type_min (str): The type of variable to use for linearization.
            lb_min (Union[float, None]): The lower bound for the linearized expression (optional).

        Returns:
            Expression: The linearized expression.

        Note:
            The linearization is performed based on the type of variable chosen for linearization and the lower bound, if provided.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('min', tuple(id(x) for x in input_list), type_min, lb_min)
        if key in cache:
            return cache[key]

        try:
            self.features['min_lins'].append(self.features['min_lins'][-1] + 1)
        except:
            self.features['min_lins'] = [0]

        if type_min == 'bvar':
            z = self._lin_bvar(f"min_lin{self.features['min_lins'][-1]}")
        elif type_min == 'ivar':
            z = self._lin_ivar(f"min_lin{self.features['min_lins'][-1]}")
        elif type_min == 'pvar':
            z = self._lin_pvar(f"min_lin{self.features['min_lins'][-1]}")
        elif type_min == 'fvar':
            z = self._lin_fvar(f"min_lin{self.features['min_lins'][-1]}")

        for item in input_list:
            self._lin_con(z <= item)
        if lb_min is not None:
            self._lin_con(z >= lb_min)
        cache[key] = z
        return z

    def lin_prod_bb(self, binary1: MultidimVariable, binary2: MultidimVariable) -> MultidimVariable:
        """
        Linearizes a Binary * Binary product.

        Returns:
            MultidimVariable: The linearized expression.

        Note:
            The linearization requires +3 constraints and +1 positive variable.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('bb', id(binary1), id(binary2))
        if key in cache:
            return cache[key]

        try:
            self.features['bb_lins'].append(self.features['bb_lins'][-1] + 1)
        except:
            self.features['bb_lins'] = [0]

        z = self._lin_pvar(f"bb_lin{self.features['bb_lins'][-1]}")
        self._lin_con([z <= binary1, z <= binary2, z >= binary1 + binary2 - 1])
        cache[key] = z
        return z
    
    def lin_prod_bp(self, binary: MultidimVariable, positive: MultidimVariable, ub_positive: float = 1e9) -> MultidimVariable:
        """
        Linearizes a Binary * Positive product.

        Returns:
            MultidimVariable: The linearized expression.

        Note:
            The linearization requires +3 constraints and +1 positive variable.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('bp', id(binary), id(positive), ub_positive)
        if key in cache:
            return cache[key]

        try:
            self.features['bp_lins'].append(self.features['bp_lins'][-1] + 1)
        except:
            self.features['bp_lins'] = [0]

        z = self._lin_pvar(f"bp_lin{self.features['bp_lins'][-1]}")
        self._lin_con(z <= positive)
        self._lin_con(z <= binary * ub_positive)
        self._lin_con(z >= positive - ub_positive * (1 - binary))
        cache[key] = z
        return z

    def lin_prod_bi(self, binary: MultidimVariable, integer: MultidimVariable, ub_integer: float = 1e9) -> MultidimVariable:
        """
        Linearizes a Binary * Integer product.

        Returns:
            MultidimVariable: The linearized expression.

        Note:
            The linearization requires +3 constraints and +1 positive variable.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('bi', id(binary), id(integer), ub_integer)
        if key in cache:
            return cache[key]

        try:
            self.features['bi_lins'].append(self.features['bi_lins'][-1] + 1)
        except:
            self.features['bi_lins'] = [0]

        z = self._lin_pvar(f"bi_lin{self.features['bi_lins'][-1]}")
        self._lin_con(z <= integer)
        self._lin_con(z <= binary * ub_integer)
        self._lin_con(z >= integer - ub_integer * (1 - binary))
        cache[key] = z
        return z

    def lin_prod_ip(self, integer: MultidimVariable, positive: MultidimVariable, ub_integer: int, ub_positive: float) -> MultidimVariable:
        """
        Linearizes an Integer * Positive product.

        Returns:
            MultidimVariable: The linearized expression.

        Note:
            The linearization requires +1 + 3 * (mt.ceil(mt.log2(ub_integer + 1))) constraints, +
            (mt.ceil(mt.log2(ub_integer + 1))) positive variables, and +
            (mt.ceil(mt.log2(ub_integer + 1))) binary variables.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('ip', id(integer), id(positive), ub_integer, ub_positive)
        if key in cache:
            return cache[key]

        try:
            self.features['ip_lins'].append(self.features['ip_lins'][-1] + 1)
        except:
            self.features['ip_lins'] = [0]

        z = self._lin_pvar(f"ip_lin{self.features['ip_lins'][-1]}", [range(mt.ceil(mt.log2(ub_integer + 1)))])
        x = self._lin_bvar(f"ip_binary_convert{self.features['ip_lins'][-1]}", [range(mt.ceil(mt.log2(ub_integer + 1)))])
        
        self._lin_con(integer == sum(2**i * x[i] for i in range(mt.ceil(mt.log2(ub_integer + 1)))))

        for i in range(mt.ceil(mt.log2(ub_integer + 1))):
            self._lin_con(z[i] <= positive)
            self._lin_con(z[i] <= x[i] * ub_positive)
            self._lin_con(z[i] >= positive - ub_positive * (1 - x[i]))

        result = sum(2**i * z[i] for i in range(mt.ceil(mt.log2(ub_integer + 1))))
        cache[key] = result
        return result

    def lin_prod_ii(self, integer1: MultidimVariable, integer2: MultidimVariable, ub_integer1: int, ub_integer2: int) -> MultidimVariable:
        """
        Linearizes an Integer * Integer product.

        Returns:
            MultidimVariable: The linearized expression.

        Note:
            The linearization requires +1 + 3 * (mt.ceil(mt.log2(ub_integer + 1))) constraints, +
            (mt.ceil(mt.log2(ub_integer + 1))) positive variables, and +
            (mt.ceil(mt.log2(ub_integer + 1))) binary variables.
        """
        cache = self.features.setdefault('_lin_cache', {})
        key = ('ii', id(integer1), id(integer2), ub_integer1, ub_integer2)
        if key in cache:
            return cache[key]

        try:
            self.features['ii_lins'].append(self.features['ii_lins'][-1] + 1)
        except:
            self.features['ii_lins'] = [0]

        z = self._lin_pvar(f"ii_lin{self.features['ii_lins'][-1]}", [range(mt.ceil(mt.log2(ub_integer1 + 1)))])
        x = self._lin_bvar(f"ii_binary_convert{self.features['ii_lins'][-1]}", [range(mt.ceil(mt.log2(ub_integer1 + 1)))])

        self._lin_con(integer1 == sum(2**i * x[i] for i in range(mt.ceil(mt.log2(ub_integer1 + 1)))))

        for i in range(mt.ceil(mt.log2(ub_integer1 + 1))):
            self._lin_con(z[i] <= integer2)
            self._lin_con(z[i] <= x[i] * ub_integer2)
            self._lin_con(z[i] >= integer2 - ub_integer2 * (1 - x[i]))

        result = sum(2**i * z[i] for i in range(mt.ceil(mt.log2(ub_integer1 + 1))))
        cache[key] = result
        return result
    
    def lin_prod_ff(self, var1: MultidimVariable, var2: MultidimVariable, bound1, bound2, num_breakpoints):
        cache = self.features.setdefault('_lin_cache', {})
        key = ('ff', id(var1), id(var2), bound1, bound2, num_breakpoints)
        if key in cache:
            return cache[key]

        try:
            self.features['ff_lins'].append(self.features['ff_lins'][-1] + 1)
        except:
            self.features['ff_lins'] = [0]
        y1 = self._lin_fvar(name=f"ff_lin1_{self.features['ff_lins'][-1]}", bound=bound1)
        y2 = self._lin_fvar(name=f"ff_lin2_{self.features['ff_lins'][-1]}", bound=bound2)
        self._lin_con(y1==0.5*(var1+var2))
        self._lin_con(y2==0.5*(var1-var2))
        result = self.lin_approx(name=f"ff_lin1_{self.features['ff_lins'][-1]}", f=lambda y1:y1**2, var=y1, bound=bound1, num_breakpoints=num_breakpoints)-self.lin_approx(name=f"ff_lin2_{self.features['ff_lins'][-1]}", f=lambda y2:y2**2, var=y2, bound=bound2, num_breakpoints=num_breakpoints)
        cache[key] = result
        return result
    
    def lin_prod_pp(self, var1: MultidimVariable, var2: MultidimVariable, ub_positive1, ub_positive2, num_breakpoints):
        cache = self.features.setdefault('_lin_cache', {})
        key = ('pp', id(var1), id(var2), ub_positive1, ub_positive2, num_breakpoints)
        if key in cache:
            return cache[key]

        try:
            self.features['pp_lins'].append(self.features['pp_lins'][-1] + 1)
        except:
            self.features['pp_lins'] = [0]
        y1 = self._lin_fvar(name=f"pp_lin1_{self.features['pp_lins'][-1]}", bound=(0, ub_positive1))
        y2 = self._lin_fvar(name=f"pp_lin2_{self.features['pp_lins'][-1]}", bound=(-ub_positive2, ub_positive1))
        self._lin_con(y1==0.5*(var1+var2))
        self._lin_con(y2==0.5*(var1-var2))
        result = self.lin_approx(name=f"pp_lin1_{self.features['pp_lins'][-1]}", f=lambda y1:y1**2, var=y1, bound=(0, ub_positive1), num_breakpoints=num_breakpoints)-self.lin_approx(name=f"pp_lin2_{self.features['pp_lins'][-1]}", f=lambda y2:y2**2, var=y2, bound=(-ub_positive2, ub_positive1), num_breakpoints=num_breakpoints)
        cache[key] = result
        return result

    # ── Auto-linearization internals ─────────────────────────────────

    #  Auto-linearization fast-path helpers 

    def _al_fvar(self, name, lb, ub):
        """Fast-path variable creation for auto-linearization. Works with all LP/MIP interfaces."""
        from feloopy.generators.variable_generator import generate_variable
        native = generate_variable(
            self.features['interface_name'], self.model, 'fvar', name, [lb, ub], 0
        )
        self.features['variables'][("fvar", name)] = native
        self.features['dimensions'][name] = 0
        self.features['total_variable_counter'][0] += 1
        self.features['total_variable_counter'][1] += 1
        self.features['free_variable_counter'][0] += 1
        self.features['free_variable_counter'][1] += 1
        self.features['_al_fvar_count'] = self.features.get('_al_fvar_count', 0) + 1
        return native

    def _al_bvar(self, name):
        """Fast-path binary variable creation for auto-linearization. Works with all LP/MIP interfaces."""
        from feloopy.generators.variable_generator import generate_variable
        native = generate_variable(
            self.features['interface_name'], self.model, 'bvar', name, [0, 1], 0
        )
        self.features['variables'][("bvar", name)] = native
        self.features['dimensions'][name] = 0
        self.features['total_variable_counter'][0] += 1
        self.features['total_variable_counter'][1] += 1
        self.features['binary_variable_counter'][0] += 1
        self.features['binary_variable_counter'][1] += 1
        self.features['_al_bvar_count'] = self.features.get('_al_bvar_count', 0) + 1
        return native

    def _al_con(self, expr):
        """Fast-path constraint addition for auto-linearization.
        Directly appends to features without check_constraint_type/set() overhead."""
        self.features['constraint_labels'].append(None)
        self.features['constraints'].append(expr)
        self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
        self.features['constraint_counter'][1] = len(self.features['constraints'])
        self.features['_al_con_count'] = self.features.get('_al_con_count', 0) + 1

    def _al_cons(self, exprs):
        """Batch fast-path constraint addition for auto-linearization.
        Adds multiple constraints in one call with O(1) overhead."""
        labels = self.features['constraint_labels']
        cons = self.features['constraints']
        labels.extend(_AL_NONE_LIST[:len(exprs)])
        cons.extend(exprs)
        self.features['constraint_counter'][0] = len(set(labels))
        self.features['constraint_counter'][1] = len(cons)
        self.features['_al_con_count'] = self.features.get('_al_con_count', 0) + len(exprs)

    # PWL (lin_*) helpers 

    def _lin_pvar(self, name, dim=0, bound=[0, None], bounds=None):
        """Variable creation for lin_* methods. Tracks PWL additions separately."""
        result = self.pvar(name=name, dim=dim, bound=bound)
        self.features['_lin_pvar_count'] = self.features.get('_lin_pvar_count', 0) + 1
        if isinstance(result, dict):
            self.features['_lin_pvar_size'] = self.features.get('_lin_pvar_size', 0) + len(result)
        else:
            self.features['_lin_pvar_size'] = self.features.get('_lin_pvar_size', 0) + 1
        return result

    def _lin_bvar(self, name, dim=0):
        """Binary variable creation for lin_* methods. Tracks PWL additions separately."""
        result = self.bvar(name=name, dim=dim)
        self.features['_lin_bvar_count'] = self.features.get('_lin_bvar_count', 0) + 1
        if isinstance(result, dict):
            self.features['_lin_bvar_size'] = self.features.get('_lin_bvar_size', 0) + len(result)
        else:
            self.features['_lin_bvar_size'] = self.features.get('_lin_bvar_size', 0) + 1
        return result

    def _lin_fvar(self, name, dim=0, bound=[None, None], bounds=None):
        """Free variable creation for lin_* methods. Tracks PWL additions separately."""
        result = self.fvar(name=name, dim=dim, bound=bound)
        self.features['_lin_fvar_count'] = self.features.get('_lin_fvar_count', 0) + 1
        if isinstance(result, dict):
            self.features['_lin_fvar_size'] = self.features.get('_lin_fvar_size', 0) + len(result)
        else:
            self.features['_lin_fvar_size'] = self.features.get('_lin_fvar_size', 0) + 1
        return result

    def _lin_con(self, expr):
        """Constraint addition for lin_* methods. Tracks PWL additions separately."""
        self.features['_lin_generating'] = True
        try:
            self.con(expr)
        finally:
            self.features['_lin_generating'] = False
        self.features['_lin_con_count'] = self.features.get('_lin_con_count', 0) + 1

    def _lin_ivar(self, name, dim=0, bound=None, bounds=None):
        """Integer variable creation for lin_* methods. Tracks PWL additions separately."""
        result = self.ivar(name=name, dim=dim, bound=bound)
        self.features['_lin_ivar_count'] = self.features.get('_lin_ivar_count', 0) + 1
        if isinstance(result, dict):
            self.features['_lin_ivar_size'] = self.features.get('_lin_ivar_size', 0) + len(result)
        else:
            self.features['_lin_ivar_size'] = self.features.get('_lin_ivar_size', 0) + 1
        return result

    # Auto-linearization helpers 

    def _auto_lin_product(self, proxy1, proxy2):
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('prod', *_al_proxy_key(proxy1, proxy2))
        if key in cache:
            return cache[key]

        if (isinstance(proxy1, LinearizationProxy) and isinstance(proxy2, LinearizationProxy)
                and proxy1._native is proxy2._native
                and proxy1._lb is not None and proxy1._ub is not None):
            lo, hi = float(proxy1._lb), float(proxy1._ub)
            if lo > 0:
                result = self._auto_lin_func(lambda x: x * x, proxy1._native,
                                             f"_autolin_sq_{id(proxy1)}",
                                             lb=lo, ub=hi)
                cache[key] = result
                return result

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        xL = proxy1._lb if proxy1._lb is not None else -BIG_M
        xU = proxy1._ub if proxy1._ub is not None else BIG_M
        yL = proxy2._lb if proxy2._lb is not None else -BIG_M
        yU = proxy2._ub if proxy2._ub is not None else BIG_M

        zL = min(xL * yL, xL * yU, xU * yL, xU * yU)
        zU = max(xL * yL, xL * yU, xU * yL, xU * yU)

        z_native = self._al_fvar(f"_autolin_{idx}", zL, zU)

        x_native = proxy1._native
        y_native = proxy2._native

        self._al_cons([
            z_native >= xL * y_native + yL * x_native - xL * yL,
            z_native >= xU * y_native + yU * x_native - xU * yU,
            z_native <= xU * y_native + yL * x_native - xU * yL,
            z_native <= xL * y_native + yU * x_native - xL * yU,
        ])

        result = LinearizationProxy(z_native, self, f"prod_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_scalar_mul(self, var, scalar, lb=None, ub=None):
        cache = self.features.setdefault('_al_cache', {})
        key = ('smul', id(var), scalar, lb, ub)
        if key in cache:
            return cache[key]
        result = LinearizationProxy(scalar * var, self, f"smul_{id(var)}", lb=lb, ub=ub)
        cache[key] = result
        return result

    def _auto_lin_expr(self, expr, lb=None, ub=None):
        cache = self.features.setdefault('_al_cache', {})
        key = ('expr', id(expr), lb, ub)
        if key in cache:
            return cache[key]
        result = LinearizationProxy(expr, self, f"expr_{id(expr)}", lb=lb, ub=ub)
        cache[key] = result
        return result

    def _auto_lin_div(self, proxy_num, proxy_den):
        """Linearize z = num / den using piecewise linear approximation."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('div', *_al_proxy_key(proxy_num, proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        nL = proxy_num._lb if proxy_num._lb is not None else -BIG_M
        nU = proxy_num._ub if proxy_num._ub is not None else BIG_M
        dL = proxy_den._lb if proxy_den._lb is not None else -BIG_M
        dU = proxy_den._ub if proxy_den._ub is not None else BIG_M

        if dL <= 0 <= dU:
            raise ValueError(
                "Division by a variable with bounds containing zero is not "
                "supported for auto-linearization. Tighten the denominator "
                "bounds to exclude zero.")

        if isinstance(proxy_num, LinearizationProxy):
            z1 = self._auto_lin_func(lambda x: 1.0 / x, proxy_den._native,
                                     f"_autolin_{idx}_inv",
                                     lb=dL, ub=dU)
            result = self._auto_lin_product_sos2(proxy_num, z1)
        else:
            scalar = float(proxy_num)
            func = lambda x, s=scalar: s / x
            result = self._auto_lin_func(func, proxy_den._native,
                                       f"_autolin_{idx}_div",
                                       lb=dL, ub=dU)
        cache[key] = result
        return result

    def _auto_lin_rdiv(self, scalar, proxy_den):
        """Linearize z = scalar / den using piecewise linear approximation."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('rdiv', scalar, *_al_proxy_key(proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        dL = proxy_den._lb if proxy_den._lb is not None else -BIG_M
        dU = proxy_den._ub if proxy_den._ub is not None else BIG_M

        if dL <= 0 <= dU:
            raise ValueError(
                "Division by a variable with bounds containing zero is not "
                "supported for auto-linearization.")

        func = lambda x, s=scalar: s / x
        result = self._auto_lin_func(func, proxy_den._native,
                                   f"_autolin_{idx}_rdiv",
                                   lb=dL, ub=dU)
        cache[key] = result
        return result

    def _auto_lin_div_binary(self, proxy_num, proxy_den):
        """Division x / b where b is binary (0 or 1)."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('div_bin', *_al_proxy_key(proxy_num, proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        nL = proxy_num._lb if proxy_num._lb is not None else 0
        nU = proxy_num._ub if proxy_num._ub is not None else BIG_M

        z_native = self._al_fvar(f"_autolin_{idx}", nL, nU)
        b_native = proxy_den._native

        self._al_cons([
            z_native <= nU * b_native,
            z_native >= proxy_num._native - nU * (1 - b_native),
            z_native <= proxy_num._native,
            z_native >= proxy_num._native - nU * (1 - b_native),
        ])

        result = LinearizationProxy(z_native, self, f"div_bin_{idx}", lb=nL, ub=nU)
        cache[key] = result
        return result

    def _auto_lin_div_small_int(self, proxy_num, proxy_den, K):
        """Division x / y where y is positive integer in [1, K]."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('div_int', *_al_proxy_key(proxy_num, proxy_den), K)
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        nL = proxy_num._lb if proxy_num._lb is not None else -BIG_M
        nU = proxy_num._ub if proxy_num._ub is not None else BIG_M

        zL = nL / K
        zU = nU

        z_native = self._al_fvar(f"_autolin_{idx}", zL, zU)

        d = self.bvar(f"_autolin_div_int_d_{idx}", dim=[range(K)])
        d_list = list(d.values()) if isinstance(d, dict) else list(d)
        d_native = [di._native if isinstance(di, LinearizationProxy) else di for di in d_list]

        w = self.pvar(f"_autolin_div_int_w_{idx}", dim=[range(K)], bound=[0, nU])
        w_list = list(w.values()) if isinstance(w, dict) else list(w)
        w_native = [wi._native if isinstance(wi, LinearizationProxy) else wi for wi in w_list]

        cons = [
            sum(d_native[k] for k in range(K)) == 1,
            proxy_den._native == sum((k + 1) * d_native[k] for k in range(K)),
        ]
        for k in range(K):
            cons.append(w_native[k] <= proxy_num._native)
            cons.append(w_native[k] <= nU * d_native[k])
            cons.append(w_native[k] >= proxy_num._native - nU * (1 - d_native[k]))
            cons.append(w_native[k] >= 0)
        cons.append(z_native == sum(w_native[k] / (k + 1) for k in range(K)))

        self._al_cons(cons)

        result = LinearizationProxy(z_native, self, f"div_int_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_div_bounded(self, proxy_num, proxy_den):
        """Division x / y where both positive and x ≤ y → ratio in [0, 1].
        Uses piecewise linear approximation for tightness."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('div_bounded', *_al_proxy_key(proxy_num, proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        nL = proxy_num._lb if proxy_num._lb is not None else 0
        nU = proxy_num._ub if proxy_num._ub is not None else BIG_M
        dL = proxy_den._lb if proxy_den._lb is not None else 1
        dU = proxy_den._ub if proxy_den._ub is not None else BIG_M

        if isinstance(proxy_num, LinearizationProxy):
            z1 = self._auto_lin_func(lambda x: 1.0 / x, proxy_den._native,
                                     f"_autolin_{idx}_inv",
                                     lb=dL, ub=dU)
            result = self._auto_lin_product_sos2(proxy_num, z1)
        else:
            scalar = float(proxy_num)
            func = lambda x, s=scalar: s / x
            result = self._auto_lin_func(func, proxy_den._native,
                                       f"_autolin_{idx}_div",
                                       lb=dL, ub=dU)
        cache[key] = result
        return result

    def _auto_lin_rdiv_small_int(self, scalar, proxy_den, K):
        """Scalar / y where y is positive integer in [1, K]."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('rdiv_int', scalar, *_al_proxy_key(proxy_den), K)
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        zL = scalar / K
        zU = scalar

        z_native = self._al_fvar(f"_autolin_{idx}", min(zL, zU), max(zL, zU))

        d = self.bvar(f"_autolin_rdiv_int_d_{idx}", dim=[range(K)])
        d_list = list(d.values()) if isinstance(d, dict) else list(d)
        d_native = [di._native if isinstance(di, LinearizationProxy) else di for di in d_list]

        self._al_cons([
            sum(d_native[k] for k in range(K)) == 1,
            proxy_den._native == sum((k + 1) * d_native[k] for k in range(K)),
            z_native == sum((scalar / (k + 1)) * d_native[k] for k in range(K)),
        ])

        result = LinearizationProxy(z_native, self, f"rdiv_int_{idx}", lb=min(zL, zU), ub=max(zL, zU))
        cache[key] = result
        return result

    def _auto_lin_abs(self, proxy):
        """Linearize z = |x| using McCormick envelope."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('abs', *_al_proxy_key(proxy))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        xL = proxy._lb if proxy._lb is not None else -BIG_M
        xU = proxy._ub if proxy._ub is not None else BIG_M

        zL = 0.0
        zU = max(abs(xL), abs(xU))

        z_native = self._al_fvar(f"_autolin_{idx}", zL, zU)

        self._al_cons([
            z_native >= proxy._native,
            z_native >= -proxy._native,
        ])

        result = LinearizationProxy(z_native, self, f"abs_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_func(self, func, native_var, name: str,
                       lb: float = None, ub: float = None,
                       num_breakpoints: int = None):
        """Linearize z = func(x) using piecewise linear approximation via lin_approx."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        if num_breakpoints is None:
            num_breakpoints = 20

        if lb is None or ub is None:
            raise ValueError(
                f"Cannot auto-linearize {name}: variable bounds required "
                f"for piecewise linear approximation.")

        prev_flag = self.features.get('auto_linearize', False)
        self.features['auto_linearize'] = False
        self.features['_auto_lin_generating'] = True
        try:
            z = self.lin_approx(name, func, native_var, (lb, ub), num_breakpoints)
        finally:
            self.features['auto_linearize'] = prev_flag
            self.features['_auto_lin_generating'] = False

        samples = np.linspace(lb, ub, 200)
        f_vals = [func(v) for v in samples]
        f_lb = min(f_vals)
        f_ub = max(f_vals)
        return LinearizationProxy(z, self, name, lb=f_lb, ub=f_ub)

    def _auto_lin_product_sos2(self, proxy1, proxy2):
        """Product z = x*y via McCormick envelope with tighter bounds."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('sos2', *_al_proxy_key(proxy1, proxy2))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        xL = proxy1._lb if proxy1._lb is not None else -BIG_M
        xU = proxy1._ub if proxy1._ub is not None else BIG_M
        yL = proxy2._lb if proxy2._lb is not None else -BIG_M
        yU = proxy2._ub if proxy2._ub is not None else BIG_M

        zL = min(xL * yL, xL * yU, xU * yL, xU * yU)
        zU = max(xL * yL, xL * yU, xU * yL, xU * yU)

        z_native = self._al_fvar(f"_autolin_{idx}", zL, zU)
        x_native = proxy1._native
        y_native = proxy2._native

        self._al_cons([
            z_native >= xL * y_native + yL * x_native - xL * yL,
            z_native >= xU * y_native + yU * x_native - xU * yU,
            z_native <= xU * y_native + yL * x_native - xU * yL,
            z_native <= xL * y_native + yU * x_native - xL * yU,
        ])

        result = LinearizationProxy(z_native, self, f"sos2_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_div_taylor(self, proxy_num, proxy_den):
        """Division z = num/den via first-order Taylor expansion.

        Expands f(num,den) = num/den around (n0, d0) = midpoints:
          z ≈ (1/d0)*num - (n0/d0^2)*den + n0/d0

        Purely linear expression no auxiliary variables needed.
        """

        cache = self.features.setdefault('_al_cache', {})
        key = ('tdiv', *_al_proxy_key(proxy_num, proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        nL = proxy_num._lb if proxy_num._lb is not None else -BIG_M
        nU = proxy_num._ub if proxy_num._ub is not None else BIG_M
        dL = proxy_den._lb if proxy_den._lb is not None else -BIG_M
        dU = proxy_den._ub if proxy_den._ub is not None else BIG_M

        if dL <= 0 <= dU:
            raise ValueError(
                "Division by a variable with bounds containing zero is not "
                "supported for auto-linearization.")

        n0 = 0.5 * (nL + nU)
        d0 = 0.5 * (dL + dU)
        inv_d0 = 1.0 / d0
        inv_d0_sq = 1.0 / (d0 * d0)

        zL = min(nL / dL, nL / dU, nU / dL, nU / dU)
        zU = max(nL / dL, nL / dU, nU / dL, nU / dU)

        taylor_expr = (inv_d0 * proxy_num._native
                       - (n0 * inv_d0_sq) * proxy_den._native
                       + n0 * inv_d0)
        result = LinearizationProxy(taylor_expr, self, f"tdiv_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_rdiv_taylor(self, scalar, proxy_den):
        """Scalar division z = scalar/den via first-order Taylor expansion."""

        cache = self.features.setdefault('_al_cache', {})
        key = ('trdiv', scalar, *_al_proxy_key(proxy_den))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        dL = proxy_den._lb if proxy_den._lb is not None else -BIG_M
        dU = proxy_den._ub if proxy_den._ub is not None else BIG_M

        if dL <= 0 <= dU:
            raise ValueError(
                "Division by a variable with bounds containing zero is not "
                "supported for auto-linearization.")

        d0 = 0.5 * (dL + dU)
        inv_d0 = 1.0 / d0
        inv_d0_sq = 1.0 / (d0 * d0)

        zL = min(scalar / dL, scalar / dU)
        zU = max(scalar / dL, scalar / dU)

        taylor_expr = (-(scalar * inv_d0_sq) * proxy_den._native
                       + 2.0 * scalar * inv_d0)
        result = LinearizationProxy(taylor_expr, self, f"trdiv_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _auto_lin_abs_exact(self, proxy):
        """Exact absolute value z = |x| via binary variable (MILP)."""
        self.features['_auto_lin_active'] = True
        self._snapshot_original()

        cache = self.features.setdefault('_al_cache', {})
        key = ('abs_exact', *_al_proxy_key(proxy))
        if key in cache:
            return cache[key]

        try:
            self.features['_auto_lin_counter'] += 1
        except KeyError:
            self.features['_auto_lin_counter'] = 0
        idx = self.features['_auto_lin_counter']

        xL = proxy._lb if proxy._lb is not None else -BIG_M
        xU = proxy._ub if proxy._ub is not None else BIG_M

        zL = 0.0
        zU = max(abs(xL), abs(xU))

        z_native = self._al_fvar(f"_autolin_{idx}", zL, zU)
        b_native = self._al_bvar(f"_autolin_abs_b_{idx}")

        self._al_cons([
            z_native >= proxy._native,
            z_native >= -proxy._native,
            z_native <= proxy._native + zU * (1 - b_native),
            z_native <= -proxy._native + zU * b_native,
        ])

        result = LinearizationProxy(z_native, self, f"abs_{idx}", lb=zL, ub=zU)
        cache[key] = result
        return result

    def _analyze_expressions(self):
        """Analyze objective and constraint expressions for linearity and degree.

        Returns:
            tuple: (is_linear, is_quadratic, has_nonlinear)
        """
        from ..helpers.problem_detect import analyze_expressions
        return analyze_expressions(self.features)

    def _snapshot_original(self):
        """Capture original variable/constraint counts before auto-linearization adds aux vars."""
        if '_original_variable_count' not in self.features:
            self.features['_original_variable_count'] = self.features['total_variable_counter'][1]
            self.features['_original_user_con_count'] = self.features.get('_user_con_count', 0)
            self.features['_original_bvar_count'] = self.features['binary_variable_counter'][0]
            self.features['_original_bvar_size'] = self.features['binary_variable_counter'][1]
            self.features['_original_ivar_count'] = self.features['integer_variable_counter'][0]
            self.features['_original_ivar_size'] = self.features['integer_variable_counter'][1]
            self.features['_original_pvar_count'] = self.features['positive_variable_counter'][0]
            self.features['_original_pvar_size'] = self.features['positive_variable_counter'][1]
            self.features['_original_fvar_count'] = self.features['free_variable_counter'][0]
            self.features['_original_fvar_size'] = self.features['free_variable_counter'][1]
            self.features['_original_con_count'] = self.features['constraint_counter'][0]
            self.features['_original_con_size'] = self.features['constraint_counter'][1]
            self.features['_al_bvar_count'] = 0
            self.features['_al_ivar_count'] = 0
            self.features['_al_pvar_count'] = 0
            self.features['_al_fvar_count'] = 0
            self.features['_al_con_count'] = 0
            self.features['_lin_bvar_count'] = 0
            self.features['_lin_pvar_count'] = 0
            self.features['_lin_fvar_count'] = 0
            self.features['_lin_ivar_count'] = 0
            self.features['_lin_con_count'] = 0

    def _refresh_snapshot_for_sol(self):
        """Refresh original counts in sol() after all user code has run.
        
        Subtracts both auto-linearization and PWL additions from current totals.
        Original = user-level variables/constraints only.
        Linearized = user + PWL + auto-linearization.
        """
        _al_bvar = self.features.get('_al_bvar_count', 0)
        _al_ivar = self.features.get('_al_ivar_count', 0)
        _al_pvar = self.features.get('_al_pvar_count', 0)
        _al_fvar = self.features.get('_al_fvar_count', 0)
        _lin_bvar_count = self.features.get('_lin_bvar_count', 0)
        _lin_bvar_size = self.features.get('_lin_bvar_size', 0)
        _lin_ivar_count = self.features.get('_lin_ivar_count', 0)
        _lin_ivar_size = self.features.get('_lin_ivar_size', 0)
        _lin_pvar_count = self.features.get('_lin_pvar_count', 0)
        _lin_pvar_size = self.features.get('_lin_pvar_size', 0)
        _lin_fvar_count = self.features.get('_lin_fvar_count', 0)
        _lin_fvar_size = self.features.get('_lin_fvar_size', 0)
        self.features['_original_bvar_count'] = self.features['binary_variable_counter'][0] - _al_bvar - _lin_bvar_count
        self.features['_original_bvar_size'] = self.features['binary_variable_counter'][1] - _al_bvar - _lin_bvar_size
        self.features['_original_ivar_count'] = self.features['integer_variable_counter'][0] - _al_ivar - _lin_ivar_count
        self.features['_original_ivar_size'] = self.features['integer_variable_counter'][1] - _al_ivar - _lin_ivar_size
        self.features['_original_pvar_count'] = self.features['positive_variable_counter'][0] - _al_pvar - _lin_pvar_count
        self.features['_original_pvar_size'] = self.features['positive_variable_counter'][1] - _al_pvar - _lin_pvar_size
        self.features['_original_fvar_count'] = self.features['free_variable_counter'][0] - _al_fvar - _lin_fvar_count
        self.features['_original_fvar_size'] = self.features['free_variable_counter'][1] - _al_fvar - _lin_fvar_size
        self.features['_original_con_count'] = self.features.get('_user_con_count', 0)
        self.features['_original_con_size'] = self.features.get('_user_con_count', 0)


class _AutomaticLinearExpression:
    def __init__(self, constant=0.0, terms=None, owner=None):
        self.constant = constant
        self.terms = dict(terms or {})
        self.owner = owner

    @staticmethod
    def from_value(value):
        if isinstance(value, _AutomaticLinearExpression):
            return value
        return _AutomaticLinearExpression(constant=value)

    def _combine(self, other, scale=1.0):
        other = self.from_value(other)
        terms = dict(self.terms)
        for key, coefficient in other.terms.items():
            terms[key] = terms.get(key, 0.0) + scale * coefficient
        return _AutomaticLinearExpression(
            self.constant + scale * other.constant, terms,
            self.owner or other.owner)

    def __add__(self, other):
        return self._combine(other)

    def __radd__(self, other):
        return self.from_value(other)._combine(self)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        return self.from_value(other)._combine(self, -1.0)

    def __mul__(self, other):
        if isinstance(other, _AutomaticLinearExpression):
            if self.owner is not None and self.owner is other.owner:
                return self.owner.linearize_square(self, other)
            raise ValueError("Automatic Benders requires linear expressions")
        return _AutomaticLinearExpression(
            self.constant * other,
            {key: coefficient * other for key, coefficient in self.terms.items()},
            self.owner)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (1.0 / other)

    def __neg__(self):
        return self * -1.0

    def _relation(self, other, sense):
        return _AutomaticLinearConstraint(self - other, sense)

    def __le__(self, other):
        return self._relation(other, '<=')

    def __ge__(self, other):
        return self._relation(other, '>=')

    def __eq__(self, other):
        return self._relation(other, '==')

    def evaluate(self, variables):
        expression = self.constant
        for key, coefficient in self.terms.items():
            expression += coefficient * variables[key]
        return expression


class _AutomaticLinearVariable(_AutomaticLinearExpression):
    def __init__(self, key, owner):
        super().__init__(terms={key: 1.0}, owner=owner)
        self.key = key


class _AutomaticLinearConstraint:
    def __init__(self, expression, sense):
        self.expression = expression
        self.sense = sense


class _AutomaticVariableArray(dict):
    def sum(self):
        return sum(self.values(), _AutomaticLinearExpression())
