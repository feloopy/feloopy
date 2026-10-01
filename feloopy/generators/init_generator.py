# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Warm-start plumbing for exact methods (``search(init=...)``, ``vstart``,
``tstart``).
"""

import warnings

import numpy as np


# --------------------------------------------------------------------------- #
# value / container helpers
# --------------------------------------------------------------------------- #

def _as_float(value, name=None):
    """Coerce a scalar-ish warm-start value to a plain ``float``."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != 1:
            raise ValueError(
                "warm start for scalar variable %r expects a single number but "
                "got %d values; pass init as a dict keyed by variable name "
                "instead" % (name, arr.size))
        return float(arr[0])
    return float(value)


def _container_items(var_obj, dim=None):
    """Return ``[(key, scalar_variable), ...]`` for an array variable.

    Every variable container shape the variable generators produce is handled:

    * plain ``dict`` (most interfaces) and dict subclasses (docplex/COPT
      ``tupledict``),
    * objects exposing ``keys()``/``__getitem__`` (Pyomo ``IndexedVar``,
      docplex containers),
    * numpy-like tensors (Gurobi ``MVar``),
    * xarray/DataArray style containers (linopy ``VariableArray``),
    * plain sequences,
    * positional indexers with no container protocol at all (JuMP
      ``JumpVar``, which only understands ``var[index]``) -- these are walked
      with the index set declared in *dim*, exactly like ``model.tstart``.

    Returns ``None`` when *var_obj* is a single scalar variable.
    """
    if var_obj is None:
        return None

    if isinstance(var_obj, dict):
        return list(var_obj.items())

    keys = getattr(var_obj, 'keys', None)
    if callable(keys):
        try:
            names = list(keys())
        except Exception:
            names = None
        if names:
            try:
                return [(k, var_obj[k]) for k in names]
            except Exception:
                pass

    shape = getattr(var_obj, 'shape', None)
    if (isinstance(shape, tuple) and len(shape) > 0
            and all(isinstance(s, (int, np.integer)) for s in shape)):
        try:
            import itertools as _it
            return [(idx, var_obj[idx])
                    for idx in _it.product(*[range(int(s)) for s in shape])]
        except Exception:
            pass

    values = getattr(var_obj, 'values', None)
    if values is not None and not isinstance(var_obj, np.ndarray):
        try:
            flat = np.asarray(values, dtype=object).ravel()
            return [(i, v) for i, v in enumerate(flat)]
        except Exception:
            pass

    if isinstance(var_obj, (list, tuple, np.ndarray)):
        return [(i, v) for i, v in enumerate(var_obj)]

    if dim:
        index_keys = _index_keys(dim)
        if index_keys:
            try:
                items = [(k, var_obj[k]) for k in index_keys]
            except Exception:
                items = None
            if items is not None:
                return items

    return None


def _index_keys(dim):
    """Index set(s) of *dim* in the order ``model.tstart`` walks them."""
    from ..operators.fix_operators import fix_dims

    fixed = fix_dims(dim)
    if isinstance(fixed, set):
        return list(fixed)
    if isinstance(fixed, (list, tuple)):
        if len(fixed) == 1:
            try:
                return list(fixed[0])
            except TypeError:
                return None
        try:
            import itertools as _it
            return list(_it.product(*fixed))
        except TypeError:
            return None
    return None


def _expand_assignments(features, init_d):
    """Yield ``(variable_object, scalar_value)`` pairs from a ``{name: value}``.

    Array variables are matched against the flattened init value in container
    order (row-major / C order, matching how variable dicts are built), so
    partial or fully-flattened values are accepted for every container shape.
    """
    dimensions = features.get('dimensions') or {}

    for key, var_obj in features['variables'].items():
        name = key[1] if isinstance(key, tuple) and len(key) > 1 else key
        if name not in init_d:
            continue

        value = init_d[name]
        dim = dimensions.get(name, 0)

        if not dim:
            yield var_obj, _as_float(value, name)
            continue

        items = _container_items(var_obj, dim)
        if items is None:
            yield var_obj, _as_float(value, name)
            continue

        try:
            flat = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            raise ValueError(
                "warm start value for variable %r is not numeric: %r"
                % (name, value))

        if flat.size < len(items):
            raise ValueError(
                "warm start for variable %r expects %d values but got %d"
                % (name, len(items), flat.size))

        for i, (_key, scalar_var) in enumerate(items):
            yield scalar_var, float(flat[i])


# --------------------------------------------------------------------------- #
# interface registry
# --------------------------------------------------------------------------- #

# interface -> init module (under ``feloopy.generators.init``)
_INIT_MODULES = {
    'pulp': 'pulp',
    'casadi': 'casadi',
    'pyomo': 'pyomo',
    'gurobi': 'gurobi',
    'cplex': 'cplex',
    'gekko': 'gekko',
    'copt': 'copt',
    'xpress': 'xpress',
    'highs': 'highs',
    'scip': 'scip',
    'ortools': 'ortools',
    'ortools_cp': 'ortools_cp',
    'mip': 'mip',
    'gams': 'gamspy',
    'uno': 'uno',
    'jump': 'jump',
    'cplex_cp': 'cplex_cp',
}

# interfaces whose start must be applied *after* the constraints have been
# added to the native model (or after the generator has materialized it).
_LATE_ONLY = frozenset({
    'highs', 'ortools', 'ortools_cp', 'cplex', 'xpress', 'copt', 'casadi',
    'scip', 'pyoptinterface', 'mip', 'cplex_cp',
})

# interfaces whose generator builds the start itself from features
# (see the *_solution_generator patches).
_FEATURE_START = frozenset({'uno', 'jump'})


def _base_interface(features):
    iface = features.get('interface_name') or ''
    return iface.split('.', 1)[0]


def _init_module_name(features):
    """Return the init-generator module name for *features*, or ``None``."""
    iface = features.get('interface_name') or ''
    if 'pyoptinterface' in iface:
        return 'pyoptinterface'
    return _INIT_MODULES.get(iface)


def is_supported(features):
    """True when the interface has a working warm-start implementation."""
    return _init_module_name(features) is not None


def _emit(message, stacklevel=3):
    """Raise a warning even when a solver module installed a blanket
    ``warnings.filterwarnings('ignore')`` (cvxpy, pulp, ... do so at import
    time, which would otherwise silence the one warning that matters here)."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)


def warn_unsupported(features, init):
    """Tell the user their ``init`` was ignored instead of dropping it."""
    iface = features.get('interface_name') or 'unknown'
    _emit(
        "warm start was requested (init=... or vstart/tstart) but the '%s' "
        "interface has no warm-start support; the solve starts from scratch. "
        "Supported exact interfaces: %s"
        % (iface, ', '.join(sorted(set(_INIT_MODULES) | {'pyoptinterface'}))),
        stacklevel=4,
    )


def _warn_apply_failure(features, iface, exc):
    _emit(
        "could not apply the warm start on interface '%s': %s: %s"
        % (features.get('interface_name', iface), type(exc).__name__, exc),
        stacklevel=4,
    )


def warn_start_dropped(solver_name, exc=None):
    """Say out loud that a solver plugin refused the start we prepared."""
    detail = "" if exc is None else " (%s: %s)" % (type(exc).__name__, exc)
    _emit(
        "warm start was requested but the '%s' solver plugin does not accept "
        "one%s; the solve starts from scratch." % (solver_name, detail),
        stacklevel=3,
    )


def _load(iface_module):
    """Import ``feloopy.generators.init.<iface_module>_init_generator``."""
    import importlib
    return importlib.import_module(
        '.init.%s_init_generator' % iface_module, 'feloopy.generators')


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #

def _apply_one(features, variable, value, fix):
    """Dispatch a single warm-start/fix assignment to the interface."""
    module_name = _init_module_name(features)
    if module_name is None:
        if not features.get('_init_unsupported_warned'):
            features['_init_unsupported_warned'] = True
            warn_unsupported(features, value)
        return None

    try:
        generator = _load(module_name)
    except ImportError as exc:
        if not features.get('_init_module_failed'):
            features['_init_module_failed'] = True
            _warn_apply_failure(features, module_name, exc)
        return None

    return generator.set_init_value(
        features=features, variable=variable, value=value, fix=fix)


def generate_init(features, variable, value, fix, record=True):
    """Apply (and remember) one warm-start assignment.

    ``fix=True`` is applied immediately: it changes variable bounds, which
    have to be in place before the rest of the model is built. Plain warm
    starts are recorded so :func:`flush_init` can re-apply them at the right
    moment, and applied straight away as well when the interface does not
    need a late application.
    """
    if fix:
        _apply_one(features, variable, value, fix=True)
        return

    if _is_late_only(features):
        # the variable object is handed to us by vstart/tstart, so remember it
        # until the solution generator can apply it to a fully built model
        if record:
            features.setdefault('_staged_assignments', []).append(
                (variable, float(value)))
        return

    _apply(features, variable, value, fix=False)


def _apply(features, variable, value, fix):
    try:
        _apply_one(features, variable, value, fix)
    except Exception as exc:      # a broken start must never break the solve
        _warn_apply_failure(features, _init_module_name(features) or '?', exc)


def _is_late_only(features):
    return _base_interface(features) in _LATE_ONLY


def stage_init_values(features, init_dicts):
    """Record ``init=`` dictionaries so they are applied at solve time."""
    if not init_dicts:
        return
    staged = features.setdefault('_staged_init', [])
    staged.extend(dict(d) for d in init_dicts)


def generate_init_values(features, init_dicts, fix=False):
    """Apply ``init`` dictionaries (batch).

    Array variables are expanded into scalar assignments first; interfaces
    with a native batch API (HiGHS, SCIP, Xpress, COPT) get one call with all
    assignments, the rest fall back to the per-variable setter.
    """
    if not init_dicts:
        return

    per_dict = [list(_expand_assignments(features, init_d))
                for init_d in init_dicts]
    per_dict = [assignments for assignments in per_dict if assignments]
    if not per_dict:
        return

    iface = _base_interface(features)
    batched = iface in ('highs', 'scip', 'xpress', 'copt', 'pyoptinterface')

    if batched:
        try:
            generator = _load(_init_module_name(features))
        except ImportError as exc:
            generator = None
            _warn_apply_failure(features, iface, exc)
        if generator is not None and hasattr(generator, 'set_init_values'):
            for assignments in per_dict:
                try:
                    generator.set_init_values(features, assignments, fix)
                except Exception as exc:
                    _warn_apply_failure(features, iface, exc)
            return

    for assignments in per_dict:
        for variable, value in assignments:
            # applied straight away: this only runs at flush time, when the
            # native model is (or is about to be) fully built. Going through
            # generate_init() here would re-stage instead of apply for the
            # late-only interfaces, which are exactly the ones that need it.
            _apply(features, variable, value, fix=fix)


def flush_init(features, force=False):
    """Apply everything staged for this model. Safe to call repeatedly.

    Without ``force=True`` the interfaces listed in ``_LATE_ONLY`` are
    skipped: their solution generator calls this again with ``force=True``
    once the constraints are in place.
    """
    if not force and _is_late_only(features):
        return
    _flush(features)


def _flush(features):
    for variable, value in (features.get('_staged_assignments') or ()):
        _apply(features, variable, value, fix=False)

    staged = features.get('_staged_init')
    if staged:
        generate_init_values(features, staged, fix=False)

    flush_pending_init(features)

    # interfaces that accumulate assignments instead of applying them one by
    # one commit them to the native model here (cplex, xpress, copt, ortools,
    # cplex_cp, ...)
    module_name = _init_module_name(features)
    if module_name:
        try:
            generator = _load(module_name)
        except ImportError as exc:
            generator = None
            _warn_apply_failure(features, module_name, exc)
        finalize = getattr(generator, 'finalize', None)
        if callable(finalize):
            try:
                finalize(features)
            except Exception as exc:
                _warn_apply_failure(features, module_name, exc)


def flush_pending_init(features):
    """Flush single-variable warm starts staged by the per-solver modules.

    Called before solve so values staged by vstart/tstart are applied even
    when ``search(init=...)`` is not used.
    """
    iface = _base_interface(features)
    if iface == 'scip':
        from .init import scip_init_generator
        try:
            scip_init_generator.flush_pending(features)
        except Exception as exc:
            _warn_apply_failure(features, iface, exc)
    elif iface == 'highs':
        from .init import highs_init_generator
        try:
            highs_init_generator.flush_pending(features)
        except Exception as exc:
            _warn_apply_failure(features, iface, exc)
