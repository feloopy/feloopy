import inspect

import numpy as np

from .enums import (
    SequentialError, PolicyType, _VALID_METHODS,
    _SDM_SEARCH_OPTIONS, _SDM_HEURISTIC_OPTIONS,
    _SDM_DEFAULT_STAGE_EPOCH, _SDM_DEFAULT_STAGE_POP_SIZE,
)
from .proxy import _VarProxy, _replay
from .solvers import _solve_model, _extract_decision

try:
    from ....helpers.solver_params import _HEURISTIC_IFACES
except Exception:  # pragma: no cover - helpers always ship with feloopy
    _HEURISTIC_IFACES = frozenset()


_MODEL_FIRST_NAMES = ('m', 'model', 'mdl', 'mod')


def _stage_signature(fn):
    """Return ``(required_positional_count, first_param_name, open)``.

    ``open`` is True when the callable accepts an arbitrary number of
    positional arguments, in which case nothing can be inferred.
    """
    try:
        spec = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, None, True
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL
           for p in spec.parameters.values()):
        return None, None, True
    params = [p for p in spec.parameters.values()
              if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in params
                   if p.default is inspect.Parameter.empty)
    return required, (params[0].name if params else None), False


def _stage_needs_model(fn, stage_args):
    """True for the heuristic ``build_model(m, *stage_args)`` signature."""
    required, first_name, open_sig = _stage_signature(fn)
    if open_sig:
        return False
    return required > len(stage_args) or first_name in _MODEL_FIRST_NAMES


def _build_stage(fn, stage_args):
    """Call a state-first stage builder, ``build_model(*stage_args)``."""
    required, _, open_sig = _stage_signature(fn)
    if not open_sig and required > len(stage_args):
        raise SequentialError(
            f"{getattr(fn, '__name__', fn)} requires {required} positional "
            f"arguments but the stage builder is called with "
            f"{len(stage_args)}. Use the exact signature "
            "build_model(state, ...) or construct the policy with "
            "method='heuristic' to use build_model(m, state, ...).")
    return fn(*stage_args)


def _call_with_model(fn, m, stage_args):
    """Invoke a model-first builder, tolerating the legacy arity."""
    try:
        spec = inspect.signature(fn)
        params = [p for p in spec.parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        open_sig = any(p.kind is inspect.Parameter.VAR_POSITIONAL
                       for p in spec.parameters.values())
    except (TypeError, ValueError):
        params, open_sig = None, True
    if not open_sig and params is not None and len(params) == len(stage_args):
        # legacy form taking only the model and the last stage argument
        # (DLA: build_model(m, horizon_data))
        return fn(m, stage_args[-1])
    return fn(m, *stage_args)


def _solve_heuristic_stage(policy, stage_args):
    """Run a model-first stage builder through a heuristic stage search.

    The search hands the builder its own agent-populated model, which is
    what ``build_model(m, state, theta) -> m`` expects: a model created
    here would carry no search agent for the solver to evaluate.  Stage
    directions are picked up from the built model itself.
    """
    import feloopy as flp
    iface = (policy.interface
             if policy.interface in _HEURISTIC_IFACES else 'feloopy')
    fn = policy.build_model_fn

    def _env(m):
        built = _call_with_model(fn, m, stage_args)
        return built if hasattr(built, 'features') else m

    r = flp.search(
        _env, method='heuristic', interface=iface,
        solver=policy.solver or 'orig-de',
        options=_stage_options(policy.solver_options, heuristic=True),
        verbose=False, repeat=1, should_run=True)
    return _extract_decision(r, policy.var_names)


def _stage_options(options, heuristic=False):
    """Drop sequential-search bookkeeping before options reach a solver.

    Heuristic stages also get a small default budget when the policy set
    none (any ``epoch``/``pop_size`` alias counts as "set"), because the
    global heuristic default would otherwise be paid T x N times per
    simulation -- see ``_SDM_DEFAULT_STAGE_EPOCH``.
    """
    opts = {k: v for k, v in (options or {}).items()
            if k not in _SDM_SEARCH_OPTIONS}
    if not heuristic:
        return {k: v for k, v in opts.items()
                if k not in _SDM_HEURISTIC_OPTIONS}
    from ....generators.solution.option_utils import (
        get_epoch_value, get_pop_size_value)
    if get_epoch_value(opts) is None:
        opts['epoch'] = _SDM_DEFAULT_STAGE_EPOCH
    if get_pop_size_value(opts) is None:
        opts['pop_size'] = _SDM_DEFAULT_STAGE_POP_SIZE
    return opts


def _solve_built(m, policy):
    """Solve one built stage model with the interface that built it.

    The built model, not the search-level interface, decides how it is
    solved: a stage handed back by a heuristic builder (mealpy, ...) goes
    through the heuristic search, an exact stage solves exactly, and a
    solver name is only forwarded when the policy was configured for that
    same interface (so a heuristic algorithm name never reaches highs).
    """
    features = getattr(m, 'features', None) or {}
    model_iface = features.get('interface_name') or policy.interface
    heuristic = model_iface in _HEURISTIC_IFACES
    options = _stage_options(policy.solver_options, heuristic=heuristic)
    directions = features.get('directions') or policy._get_directions()
    if heuristic:
        solved = _solve_model(
            m, method='heuristic', interface=model_iface,
            solver=policy.solver or 'orig-de', solver_options=options,
            directions=directions)
    else:
        solver = policy.solver if policy.interface == model_iface else None
        solved = _solve_model(
            m, method='exact', interface=model_iface, solver=solver,
            solver_options=options, directions=directions,
            obj_operators=policy.obj_operators,
            uncertainty_set_constraints=policy.uncertainty_set_constraints)
    return _extract_decision(solved, policy.var_names)


def _state_with_exo(state, ctx):
    """State dict merged with the exogenous realisation, as builders see it."""
    merged = CFA._state_to_dict(state)
    exo = ctx.get('exogenous', {})
    if isinstance(exo, dict):
        merged.update(exo)
    return merged


class PFA:
    """Class 1 -- Policy Function Approximation (Powell 2019, Ch. 12).

    X_t = pi(S_t, theta)

    An analytical function mapping the state variable S_t directly to a
    feasible decision X_t.  No optimisation model is solved; the policy
    itself encodes the decision rule.  Parameters theta are tuned via
    policy search (derivative-free, gradient-based, or policy-gradient).

    Common instantiations:
      - Lookup tables
      - Parametric functions (linear, polynomial, radial basis)
      - Neural networks (for high-dimensional state spaces)
      - Greedy / threshold rules

    The user supplies:
        policy_fn(state, theta) -> decision array
    """

    def __init__(self, policy_fn, theta0=None, name='pfa',
                 tpar_info=None):
        if not callable(policy_fn):
            raise SequentialError("PFA: policy_fn must be callable")
        self.policy_fn = policy_fn
        self.theta = (np.asarray(theta0, dtype=float)
                      if theta0 is not None else np.array([]))
        self.name = name
        self._type = PolicyType.PFA
        self._tpar_info = tpar_info or {}

    def _theta_to_dict(self, theta_vec):
        """Convert flat theta vector to named dict using tpar_info."""
        if not self._tpar_info:
            return theta_vec
        result = {}
        offset = 0
        for name, info in self._tpar_info.items():
            dim = info['dim']
            result[name] = theta_vec[offset:offset + dim]
            if dim == 1:
                result[name] = result[name][0]
            offset += dim
        return result

    def __call__(self, state, **ctx):
        theta_dict = self._theta_to_dict(self.theta)
        return self.policy_fn(state, theta_dict)


class CFA:
    """Class 2 -- Cost Function Approximation (Powell 2019, Ch. 13).

    X_t = argmin { C(x, S_t, theta) : x in X(S_t, theta) }
           x

    A parameterised optimisation model.  The user supplies a callable
    that, given (state, theta), builds and returns a feloopy model whose
    objective captures a cost function (not the true objective) that is
    parameterised by theta.  The framework solves this model at each
    time step to obtain the decision.

    Common instantiations:
      - Upper confidence bounding (UCB) for bandits
      - Parameterised recourse / constraint models
      - Any feloopy model whose objective or constraints depend on theta

    The user supplies:
        For exact/uncertain/constraint:
            build_model(state, theta) -> feloopy model instance
        For heuristic:
            build_model(m, state, theta) -> m
    """

    @staticmethod
    def _state_to_dict(state):
        if isinstance(state, dict):
            return dict(state)
        if isinstance(state, np.ndarray):
            return {i: v for i, v in enumerate(state.flatten())}
        if isinstance(state, (list, tuple)):
            return {i: v for i, v in enumerate(state)}
        return dict(state)

    def __init__(self, build_model_fn, theta0=None, var_names=None,
                 method='exact', interface=None, solver=None,
                 solver_options=None, obj_operators=None,
                 uncertainty_set_constraints=None, name='cfa'):
        if not callable(build_model_fn):
            raise SequentialError("CFA: build_model_fn must be callable")
        if method not in _VALID_METHODS:
            raise SequentialError(
                f"CFA: method={method!r} must be one of {_VALID_METHODS}")
        self.build_model_fn = build_model_fn
        self.theta = (np.asarray(theta0, dtype=float)
                      if theta0 is not None else np.array([]))
        self.var_names = var_names
        self.method = method
        self.interface = interface
        self.solver = solver
        self.solver_options = solver_options or {}
        self.obj_operators = obj_operators
        self.uncertainty_set_constraints = uncertainty_set_constraints
        self.name = name
        self._type = PolicyType.CFA

    def _get_directions(self):
        dirs = self.solver_options.get('directions', None)
        if dirs is None:
            dirs = getattr(self, '_directions', None)
        if dirs is None:
            return ['min']
        return list(dirs)

    def __call__(self, state, **ctx):
        method = self.method
        if method == 'heuristic':
            return self._call_heuristic(state, **ctx)
        return self._call_exact(state, **ctx)

    def _call_exact(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        stage = (merged, self.theta)
        if _stage_needs_model(self.build_model_fn, stage):
            raise SequentialError(
                "CFA: build_model(m, state, theta) is the heuristic "
                "signature; construct the policy with method='heuristic'.")
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)

    def _call_heuristic(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        stage = (merged, self.theta)
        if _stage_needs_model(self.build_model_fn, stage):
            return _solve_heuristic_stage(self, stage)
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)


class VFA:
    """Class 3 -- Value Function Approximation (Powell 2019, Ch. 14-18).

    X_t = argmin { c(x, S_t) + V_bar(S^M(S_t, x, W), theta) }
           x

    A one-step lookahead policy where the immediate cost c(x, S_t) is
    augmented by a parameterised approximation V_bar of the downstream
    (cost-to-go) value function.

    The user supplies:
        build_model(state, theta, value_fn) -> feloopy model
        value_fn(next_state, theta) -> scalar approximate cost-to-go
    """

    def __init__(self, build_model_fn, value_fn=None, theta0=None,
                 var_names=None, method='exact', interface=None,
                 solver=None, solver_options=None, obj_operators=None,
                 uncertainty_set_constraints=None, name='vfa'):
        if not callable(build_model_fn):
            raise SequentialError("VFA: build_model_fn must be callable")
        if value_fn is not None and not callable(value_fn):
            raise SequentialError("VFA: value_fn must be callable or None")
        if method not in _VALID_METHODS:
            raise SequentialError(
                f"VFA: method={method!r} must be one of {_VALID_METHODS}")
        self.build_model_fn = build_model_fn
        self.value_fn = value_fn or (lambda state, theta: 0.0)
        self.theta = (np.asarray(theta0, dtype=float)
                      if theta0 is not None else np.array([]))
        self.var_names = var_names
        self.method = method
        self.interface = interface
        self.solver = solver
        self.solver_options = solver_options or {}
        self.obj_operators = obj_operators
        self.uncertainty_set_constraints = uncertainty_set_constraints
        self.name = name
        self._type = PolicyType.VFA

    def _get_directions(self):
        dirs = self.solver_options.get('directions', None)
        if dirs is None:
            dirs = getattr(self, '_directions', None)
        if dirs is None:
            return ['min']
        return list(dirs)

    def __call__(self, state, **ctx):
        method = self.method
        if method == 'heuristic':
            return self._call_heuristic(state, **ctx)
        return self._call_exact(state, **ctx)

    def _call_exact(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        stage = (merged, self.theta, self.value_fn)
        if _stage_needs_model(self.build_model_fn, stage):
            raise SequentialError(
                "VFA: build_model(m, state, theta, value_fn) is the "
                "heuristic signature; construct the policy with "
                "method='heuristic'.")
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)

    def _call_heuristic(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        stage = (merged, self.theta, self.value_fn)
        if _stage_needs_model(self.build_model_fn, stage):
            return _solve_heuristic_stage(self, stage)
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)

    @staticmethod
    def _state_to_dict(state):
        return CFA._state_to_dict(state)


class DLA:
    """Class 4 -- Direct Lookahead Approximation (Powell 2019, Ch. 19).

    X_t = argmin { sum_{tau=t}^{t+H-1} c(x_tau, S_tau) }
           x

    A multi-period lookahead model built as a single feloopy optimisation
    problem spanning the planning horizon H.

    The user supplies:
        For exact/uncertain/constraint:
            build_model(state, horizon_data) -> feloopy model
        For heuristic:
            build_model(m, horizon_data) -> m
    """

    def __init__(self, build_model_fn, horizon=1, scenarios=None,
                 var_names=None, method='exact', interface=None,
                 solver=None, solver_options=None, obj_operators=None,
                 uncertainty_set_constraints=None, name='dla'):
        if not callable(build_model_fn):
            raise SequentialError("DLA: build_model_fn must be callable")
        if method not in _VALID_METHODS:
            raise SequentialError(
                f"DLA: method={method!r} must be one of {_VALID_METHODS}")
        self.build_model_fn = build_model_fn
        self.horizon = horizon
        self.scenarios = scenarios
        self.var_names = var_names
        self.method = method
        self.interface = interface
        self.solver = solver
        self.solver_options = solver_options or {}
        self.obj_operators = obj_operators
        self.uncertainty_set_constraints = uncertainty_set_constraints
        self.name = name
        self._type = PolicyType.DLA

    def _get_directions(self):
        dirs = self.solver_options.get('directions', None)
        if dirs is None:
            dirs = getattr(self, '_directions', None)
        if dirs is None:
            return ['min']
        return list(dirs)

    def __call__(self, state, **ctx):
        method = self.method
        if method == 'heuristic':
            return self._call_heuristic(state, **ctx)
        return self._call_exact(state, **ctx)

    def _call_exact(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        horizon_data = {
            'horizon': self.horizon,
            'scenarios': self.scenarios,
            'state': merged,
        }
        stage = (merged, horizon_data)
        if _stage_needs_model(self.build_model_fn, stage):
            raise SequentialError(
                "DLA: build_model(m, horizon_data) is the heuristic "
                "signature; construct the policy with method='heuristic'.")
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)

    def _call_heuristic(self, state, **ctx):
        merged = _state_with_exo(state, ctx)
        horizon_data = {
            'horizon': self.horizon,
            'scenarios': self.scenarios,
            'state': merged,
        }
        stage = (merged, horizon_data)
        if _stage_needs_model(self.build_model_fn, stage):
            return _solve_heuristic_stage(self, stage)
        m = _build_stage(self.build_model_fn, stage)
        if isinstance(m, np.ndarray):
            return m
        if hasattr(m, 'features'):
            return _solve_built(m, self)
        return np.asarray(m)
