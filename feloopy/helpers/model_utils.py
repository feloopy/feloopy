# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Private helpers backing the classes defined in ``feloopy.feloopy``.
"""

import math as mt
import pickle
import threading

import numpy as np

_CHEAP_PICKLE_TYPES = (type(None), bool, int, float, complex, str, bytes,
                       bytearray)

_PROBE_STATE = threading.local()


def _can_pickle(value):
    """True when ``value`` survives a ``pickle.dumps`` probe.
    """
    if isinstance(value, _CHEAP_PICKLE_TYPES):
        return True
    if isinstance(value, np.ndarray) and value.dtype != object:
        return True
    st = getattr(_PROBE_STATE, 'st', None)
    if st is None:
        st = _PROBE_STATE.st = {'depth': 0, 'memo': {}}
    key = id(value)
    entry = st['memo'].get(key)
    if entry is not None:
        return entry[1]
    st['depth'] += 1

    st['memo'][key] = (value, True)
    try:
        pickle.dumps(value)
        return True
    except Exception:
        st['memo'][key] = (value, False)
        return False
    finally:
        st['depth'] -= 1
        if st['depth'] == 0:
            st['memo'].clear()


def _salvage_value(value, stringify, depth=0):
    """Pickle-safe stand-in for a value already known to be unpicklable.
    """
    if depth > 6:
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not _can_pickle(k):
                try:
                    k = str(k)
                except Exception:
                    continue
            out[k] = v if _can_pickle(v) else _salvage_value(v, stringify, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        salvaged = [v if _can_pickle(v)
                    else _salvage_value(v, stringify, depth + 1)
                    for v in value]
        return salvaged if isinstance(value, list) else tuple(salvaged)
    if stringify:
        try:
            return str(value)
        except Exception:
            return None
    return None


def _sanitize_features(features, solutions=None):
    """Return ``(picklable_copy, changed)`` for a model ``features`` dict.
    """
    out = dict(features)
    changed = False

    for attr in ('model_object', 'model_object_before_solve',
                 '_staged_assignments'):
        value = out.get(attr)
        if value is not None and not _can_pickle(value):
            out[attr] = None
            changed = True

    variables = out.get('variables')
    if isinstance(variables, dict) and not _can_pickle(variables):
        variables = dict(variables)
        for key, handle in variables.items():
            if not _can_pickle(handle):
                var_name = key[1] if isinstance(key, tuple) and len(key) > 1 else key
                variables[key] = solutions.get(var_name) if isinstance(solutions, dict) else None
        out['variables'] = variables
        changed = True

    for attr in ('objectives', 'constraints'):
        value = out.get(attr)
        if isinstance(value, (list, tuple)) and not _can_pickle(value):
            out[attr] = _salvage_value(value, stringify=True)
            changed = True

    for key, value in list(out.items()):
        if value is None or _can_pickle(value):
            continue
        out[key] = _salvage_value(value, stringify=False)
        changed = True

    return out, changed


def _constraint_violation(value):
    """Non-negative violation of a heuristic constraint value.
    """
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(float)
    if arr.dtype == bool:
        return np.logical_not(arr).astype(float)
    return np.maximum(arr, 0)


def _flatten_constraint(value):
    """Yield the scalar/array leaves of a (possibly nested) constraint entry."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_constraint(item)
    else:
        yield value


def _constraint_violations(entry):
    """Flat vector of non-negative violations for one stored constraint.
    """
    if isinstance(entry, (list, tuple)):
        elements = []
        for item in entry:
            elements.extend(_flatten_constraint(item))
    else:
        elements = [entry]
    out = []
    for element in elements:
        if element is None:
            continue
        if isinstance(element, (bool, np.bool_)):
            out.append(np.array([0.0 if element else 1.0]))
            continue
        arr = np.asarray(element)
        if arr.dtype == bool:
            out.append(np.logical_not(arr).astype(float).ravel())
        else:
            out.append(np.maximum(arr.astype(float), 0).ravel())
    if not out:
        return np.zeros(0, dtype=float)
    return np.concatenate(out)


def _is_solver_expression(value):
    """True for solver-side expression objects that must be evaluated after
    the solve rather than captured at build time (x[i] + y[j], LinExpr, ...)."""
    if isinstance(value, (int, float, bool, str, bytes, np.ndarray, np.number,
                          list, tuple, dict, set, type(None))):
        return False
    if hasattr(value, '_resolve'):
        # DataRef-style objects resolve against the live dataset
        return False
    if hasattr(value, 'highs') and hasattr(value, 'index'):
        # a bare solver variable (highs_var) still needs the primal solution
        return True
    cls = type(value).__name__
    return (hasattr(value, 'evaluate') or hasattr(value, 'getValue')
            or cls.endswith(('Expression', 'Expr', 'LinExpr', 'QuadExpr',
                             'AffExpr', 'Proxy')))


def _evaluate_term_expression(expr, em):
    """Evaluate a deferred term expression against the latest solution.

    Returns None when no interface-specific evaluation path works, so the
    report can show an honest '-' instead of a stale/garbage value.
    """
    # highs: primal column values indexed like expr.idxs
    try:
        if hasattr(expr, 'evaluate'):
            mo = getattr(em, 'model', None)
            if mo is not None and hasattr(mo, 'getSolution'):
                col = getattr(mo.getSolution(), 'col_value', None)
                if col is not None:
                    return expr.evaluate(np.asarray(col, dtype=float))
    except Exception:
        pass
    # highs: a bare variable -> its own primal value
    try:
        if hasattr(expr, 'index') and hasattr(expr, 'highs'):
            mo = getattr(em, 'model', None)
            if mo is not None and hasattr(mo, 'getSolution'):
                col = getattr(mo.getSolution(), 'col_value', None)
                if col is not None:
                    return col[expr.index]
    except Exception:
        pass
    # gurobi-style LinExpr.getValue()
    try:
        if hasattr(expr, 'getValue'):
            return expr.getValue()
    except Exception:
        pass
    # pulp-style .value / .value()
    try:
        v = getattr(expr, 'value', None)
        if v is not None:
            v = v() if callable(v) else v
            if v is not None:
                return v
    except Exception:
        pass
    try:
        return float(expr)
    except Exception:
        return None


def _slot_index(dims, args):
    """Flat slot of ``args`` inside a variable's element slice.
    """
    if isinstance(dims, (set, frozenset)):
        _members = list(dims)
        for _key in (tuple(args), args[0] if len(args) == 1 else None):
            if _key is None:
                continue
            try:
                return _members.index(_key)
            except ValueError:
                continue
        raise KeyError("index %r not in dimension %r" % (args, dims))
    return sum(args[k] * mt.prod(len(dims[j]) for j in range(k + 1, len(dims)))
               for k in range(len(dims)))
