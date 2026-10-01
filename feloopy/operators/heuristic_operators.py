# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.



from ..helpers.empty import *
from ..helpers.error import *
import numpy as np
import math as mt
import inspect
import ast

from .arraybase import _ArrayBase


class NumpyVariable(_ArrayBase):
    """A heuristic decision variable.
    """

    
def generate_heuristic_variable(features, type, name, variable_dim, variable_bound, agent, no_agents):
    """Build a heuristic variable; set-dim declarations (one slot per index
    tuple, e.g. ``d = pvar(dim={(i,v,t,p) ...})``) are sized to ``len(set)``
    and returned with a tuple->slot key map so ``d[i,v,t,p]`` resolves."""

    _set_keys = None
    if isinstance(variable_dim, (set, frozenset)):
        _set_keys = list(variable_dim)
        # treat the set as a 1-D axis over its elements: same random/bound
        # arithmetic and the same agent slice, without the bogus reshape
        variable_dim = [range(len(_set_keys))]
    out = _generate_heuristic_variable_raw(
        features, type, name, variable_dim, variable_bound, agent, no_agents)
    if isinstance(out, NumpyVariable):
        if _set_keys is not None:
            out._key_map = {k: i for i, k in enumerate(_set_keys)}
        # records whether this variable carries the leading population axis, so
        # __getitem__ can tell the vectorized layout from the plain one
        out._vec = bool(features['vectorized'])
    return out


def _generate_heuristic_variable_raw(features, type, name, variable_dim, variable_bound, agent, no_agents):

    if no_agents==None:
        no_agents=100

    if features['agent_status'] == 'idle':

        if features['vectorized']:

            if variable_dim == 0:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(no_agents,1)*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(no_agents,1)*(variable_bound[1]-variable_bound[0])).astype(int))
                if type == 'svar':
                    raise VariableDimError("Dimension is set to be 0 or not defined for a sequential variable.")
                
            else:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0])).astype(int) )
                if type == 'svar':          
                    return NumpyVariable(np.argsort(np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim])), axis=1))

        else:
            
           

            if variable_dim == 0:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand()*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand()*(variable_bound[1]-variable_bound[0])).astype(int))
                if type == 'svar':
                    raise VariableDimError("Dimension is set to be 0 or not defined for a sequential variable.")
            
            else:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(*tuple([len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(*tuple([len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0])).astype(int) )
                if type == 'svar':          
                    return NumpyVariable(np.argsort(np.random.rand(*tuple([len(dims) for dims in variable_dim]))))
    else:

        spread = features['variable_spread'][name]

        if features['vectorized']:
            if variable_dim == 0:
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))
                
                elif type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]))
                else:
                    return NumpyVariable(np.argsort(agent[:, spread[0]:spread[1]]))
            else:

                if type == 'bvar' or type == 'ivar':

                    var = NumpyVariable(np.round(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))

                    return NumpyVariable(np.reshape(var, [var.shape[0]]+[len(dims) for dims in variable_dim]))

                elif type == 'pvar' or type == 'fvar':

                    var = variable_bound[0] + agent[:, spread[0]:spread[1]
                                                    ] * (variable_bound[1] - variable_bound[0])

                    return NumpyVariable(np.reshape(var, [var.shape[0]]+[len(dims) for dims in variable_dim]))

                else:

                    return NumpyVariable(np.argsort(agent[:, spread[0]:spread[1]]))
        else:

            if variable_dim == 0:

                if type == 'bvar' or type == 'ivar':
                    
                    return NumpyVariable(np.round(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))

                elif type == 'pvar' or type == 'fvar':

                    return NumpyVariable(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]))

                else:

                    return NumpyVariable(np.argsort(agent[spread[0]:spread[1]]))

            else:

                if type == 'bvar' or type == 'ivar':
                    
                    return NumpyVariable(np.reshape(np.round(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])), [len(dims) for dims in variable_dim]).astype(int))

                elif type == 'pvar' or type == 'fvar':

                    return NumpyVariable(np.reshape(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]), [len(dims) for dims in variable_dim]))

                else:

                    return NumpyVariable(np.argsort(agent[spread[0]:spread[1]]))
                
def get_return_names(fn):
    src = inspect.getsource(fn)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    ret = stmt.value
                    if isinstance(ret, ast.Tuple):
                        return [ast.unparse(el).strip() for el in ret.elts]
                    else:
                        return [ast.unparse(ret).strip()]
    return []


def to_plain_python(value):
    """Recursively convert decoder outputs to plain Python so reports and
    ``m.get`` return standard types: numpy scalars become ``int``/``float``,
    arrays become nested lists, containers are recursed.  Never raises —
    exotic objects pass through unchanged."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, np.generic):
                k = k.item()
            out[k] = to_plain_python(v)
        return out
    if isinstance(value, tuple):
        return tuple(to_plain_python(v) for v in value)
    if isinstance(value, list):
        return [to_plain_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()  # nested lists, python scalars throughout
    if isinstance(value, np.generic):
        return value.item()
    return value

def get_in_out(fn, /, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        inputs = args[0]
    elif not args and kwargs:
        inputs = kwargs
    else:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments

    result = fn(**inputs)

    if isinstance(result, dict):
        outputs = result
    else:
        names = get_return_names(fn)
        values = result if isinstance(result, tuple) else (result,)
        if len(names) != len(values):
            raise ValueError(
                f"{fn.__name__!r} returned {len(values)} value(s) "
                f"but {len(names)} name(s) were found: {names!r}"
            )
        outputs = dict(zip(names, values))

    return inputs, outputs


def infer_bounds_from_constraints(env_fn, args, kwargs, features, interface, pop_size, auto_linearize=False, n_samples=30):
    """
    Infers feasible bounds for heuristic variables that have no user-specified bounds.
    """
    auto_vars = features.get('_auto_bound_vars', [])
    if not auto_vars:
        return {}
    cc = features.get('constraint_counter', [0])
    if isinstance(cc, (list, tuple)) and cc[0] == 0:
        return {}

    known_max_range = 0
    for vn in features['variable_bound']:
        if vn not in auto_vars:
            b = features['variable_bound'][vn]
            if b[0] is not None and b[1] is not None:
                known_max_range = max(known_max_range, b[1] - b[0])
    if known_max_range <= 0:
        known_max_range = 100.0

    from ..feloopy import model as _Model

    def _check_feasibility(scale):
        auto_scales = {}
        for vn in auto_vars:
            vtype = features['variable_type'].get(vn, 'pvar')
            if vtype == 'fvar':
                auto_scales[vn] = [-scale, scale]
            else:
                auto_scales[vn] = [0, scale]
        feasible_count = 0
        n_per_scale = max(n_samples // 3, 10)
        for _ in range(n_per_scale):
            try:
                m = _Model(method='heuristic', name='_bp', interface=interface, agent=['idle'], no_agents=1, auto_linearize=auto_linearize)
                m.features['_auto_bound_overrides'] = dict(auto_scales)
                # The probe pre-declares known-bound variables and then re-runs
                # the environment, which declares them again; allow the shadow.
                m.features['_allow_duplicate_names'] = True
                for name in features['variable_type']:
                    if name in auto_vars:
                        continue
                    vtype = features['variable_type'][name]
                    bound = list(features['variable_bound'][name])
                    if any(b is None for b in bound):
                        continue
                    vdim = features.get('variable_dim', {}).get(name, 0)
                    if vtype == 'pvar':
                        m.pvar(name=name, dim=vdim, bound=bound)
                    elif vtype == 'fvar':
                        m.fvar(name=name, dim=vdim, bound=bound)
                    elif vtype == 'ivar':
                        m.ivar(name=name, dim=vdim, bound=bound)
                    elif vtype == 'bvar':
                        m.bvar(name=name, dim=vdim, bound=bound)
                result = env_fn(m, *args, **kwargs)
                if result is None:
                    result = m
                constraints = result.features.get('constraints', [])
                is_feasible = True
                for c in constraints:
                    c_arr = np.asarray(c).flatten()
                    if np.any(c_arr > 0):
                        is_feasible = False
                        break
                if is_feasible:
                    feasible_count += 1
            except Exception:
                continue
        return feasible_count

    lo, hi = 0.0, known_max_range
    best = lo

    # Single binary search: _check_feasibility applies the same scale to ALL auto variables
    for _ in range(12):
        if hi - lo < 1e-6:
            break
        mid = (lo + hi) / 2.0
        fc = _check_feasibility(mid)
        if fc > 0:
            best = mid
            lo = mid
        else:
            hi = mid

    # Validate the inferred bound
    if best > 0:
        margin = 1.2
        vc = _check_feasibility(best * margin)
        if vc == 0:
            margin = 1.05
            vc = _check_feasibility(best * margin)
            if vc == 0:
                margin = 1.0

    inferred = {}
    for var_name in auto_vars:
        vtype = features['variable_type'].get(var_name, 'pvar')
        if best > 0:
            if vtype == 'pvar':
                inferred[var_name] = [0, best * margin]
            elif vtype == 'fvar':
                inferred[var_name] = [-best * margin, best * margin]
            elif vtype == 'ivar':
                inferred[var_name] = [0, int(best * margin) + 1]

    return inferred
