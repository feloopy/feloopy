import numpy as np

from .enums import SequentialError, PolicyType, _log_message
from .proxy import _VarProxy
from .trace import Trace
from .result import SimulationResult, OptimizationResult


def _mealpy_algo_class(solver_name='orig-de'):
    """Resolve a mealpy algorithm class from its feloopy solver name.

    ``from mealpy import ORIG_DE`` only worked on old mealpy releases; the
    class now lives under ``mealpy.evolutionary_based.DE.OriginalDE``.  The
    shared ``module_mappings`` table is the same one the mealpy model
    generator uses, so the optimizer and the solver stay in sync.
    """
    try:
        from ....generators.model.mealpy_model_generator import module_mappings
        mapping = (module_mappings.get(solver_name)
                   or module_mappings.get('orig-de'))
        module_name, class_name, model_name = mapping
        module = __import__(module_name, fromlist=[class_name])
        return getattr(getattr(module, class_name), model_name)
    except Exception:
        # last resort: pre-3.0 mealpy exported the classes at top level
        import mealpy
        for candidate in (solver_name, 'orig-de', 'de'):
            cls = getattr(mealpy, candidate.upper().replace('-', '_'), None)
            if cls is not None:
                return cls
        raise SequentialError(
            f"Cannot resolve mealpy algorithm {solver_name!r}.")


class SequentialDecisionProblem:
    """General-purpose sequential decision-making framework.

    Wraps a user-defined feloopy model into a multi-stage simulator
    following Powell's unified framework for sequential decisions.

    Powell's framework (Reinforcement Learning and Stochastic
    Optimization, 2022) decomposes any sequential problem into five
    components:

      1. State variables S_t        -- information available at time t
      2. Decision variables X_t     -- what we control
      3. Exogenous information W_t  -- uncertain realisation
      4. Transition function        -- S_{t+1} = S^M(S_t, X_t, W_{t+1})
      5. Objective function         -- minimise/maximise cumulative cost

    The user defines:
      - environment:  def model(m, state, t): return m   (feloopy model)
      - transition:   S_{t+1} = f(S_t, X_t, W_{t+1})
      - cost/reward:  c_t = c(S_t, X_t, W_t)
      - exogenous:    W_t ~ some process (optional)

    Usage::

        sdp = SequentialDecisionProblem()
        sdp.environment(my_model)
        sdp.transition(lambda s, x, w, t: s + x - w)
        sdp.cost(lambda s, x, w, t: abs(x[0] - s[0]))
        sdp.exogenous(lambda t, rng: rng.normal(0, 1))

        policy = PFA(lambda s, theta: [theta[0] * s[0]], theta0=[0.5])
        result = sdp.simulate(policy, S0=[5.0], T=20, N=100, seed=1)
    """

    def __init__(self):
        """Initialise an empty sequential decision problem.

        Use ``environment``, ``transition``, ``cost``/``reward``, and
        ``exogenous`` to define the problem, then call ``simulate()``.
        """
        self._env_fn = None
        self._transition_fn = None
        self._cost_fn = None
        self._exogenous_fn = None
        self._terminal_cost_fn = None
        self._reward_fn = None
        self._feasibility_fn = None
        self._direction = 'min'
        self._sdp_vars = {}

    def environment(self, fn):
        """Set the feloopy model builder: def model(m, state, t) -> m"""
        self._env_fn = fn
        return self

    def transition(self, fn):
        """Set S_{t+1} = fn(state, decision, exogenous, t)"""
        self._transition_fn = fn
        return self

    def cost(self, fn):
        """Set stage cost: c_t = fn(state, decision, exogenous, t)"""
        self._cost_fn = fn
        return self

    def reward(self, fn):
        """Set stage reward: r_t = fn(state, decision, exogenous, t)"""
        self._reward_fn = fn
        self._direction = 'max'
        return self

    def exogenous(self, fn):
        """Set exogenous process: W_t = fn(t, rng)"""
        self._exogenous_fn = fn
        return self

    def terminal_cost(self, fn):
        """Set terminal cost: V_T = fn(final_state)"""
        self._terminal_cost_fn = fn
        return self

    def feasibility(self, fn):
        """Set feasibility check: feasible = fn(state, decision, exogenous, t)"""
        self._feasibility_fn = fn
        return self

    def _validate(self):
        if self._env_fn is None:
            if self._transition_fn is None:
                raise SequentialError(
                    "No transition function. Call sdp.transition(fn).")
            if self._cost_fn is None and self._reward_fn is None:
                raise SequentialError(
                    "No cost or reward function. "
                    "Call sdp.cost(fn) or sdp.reward(fn).")
            return
        if self._transition_fn is None:
            raise SequentialError(
                "No transition function. Call sdp.transition(fn).")
        if self._cost_fn is None and self._reward_fn is None:
            raise SequentialError(
                "No cost or reward function. "
                "Call sdp.cost(fn) or sdp.reward(fn).")

    def simulate(self, policy, S0, T, N=1, seed=None, discount=1.0,
                 show_log=False, save_trace=True, **sim_kwargs):
        """Run Monte Carlo simulation of the sequential process.

        Parameters
        ----------
        policy : PFA | CFA | VFA | DLA
            The decision policy.
        S0 : array-like or dict
            Initial state vector S_0, or dict of {var_name: array}.
        T : int
            Number of stages (planning horizon per replication).
        N : int
            Number of Monte Carlo replications.
        seed : int or None
            Random seed for reproducibility.
        discount : float
            Discount factor gamma in [0, 1].
        show_log : bool
            Print progress every ~10% of simulations.
        save_trace : bool
            Store per-step traces (states, decisions, costs).
        **sim_kwargs
            Extra context forwarded to policy calls.

        Returns
        -------
        SimulationResult
        """
        self._validate()
        rng = np.random.default_rng(seed)

        use_dict_state = bool(self._sdp_vars)

        if use_dict_state:
            if isinstance(S0, dict):
                state0_dict = {k: np.asarray(v, dtype=float)
                               for k, v in S0.items()}
            else:
                state0_dict = {}
                for name, var_info in self._sdp_vars.items():
                    if var_info['type'] == 'state':
                        state0_dict[name] = var_info['initial'].copy()
        else:
            S0 = np.asarray(S0, dtype=float)

        all_total_costs = np.zeros(N)
        all_discounted_costs = np.zeros(N)
        all_n_infeasible = np.zeros(N, dtype=int)
        traces = [] if save_trace else None

        for n in range(N):
            if use_dict_state:
                state = {k: v.copy() for k, v in state0_dict.items()}
            else:
                state = S0.copy()
            total = 0.0
            disc_total = 0.0
            n_infeasible = 0
            trace = Trace() if save_trace else None

            for t in range(T):
                if self._exogenous_fn is not None:
                    w = self._exogenous_fn(t, rng)
                    if use_dict_state and not isinstance(w, dict):
                        w = {'_exo': np.atleast_1d(w)}
                    elif not isinstance(w, dict) and np.ndim(w) == 0:
                        # a scalar realisation must still be indexable as
                        # w[0] inside user transition/cost functions, the
                        # same way the dict-state path wraps it
                        w = np.atleast_1d(w)
                else:
                    w = {} if use_dict_state else None

                if use_dict_state:
                    ctx = {'t': t, 'exogenous': w, **sim_kwargs}
                    decision = policy(state, **ctx)
                    if not isinstance(decision, dict):
                        dp_names = [v['proxy']._name
                                    for v in self._sdp_vars.values()
                                    if v['type'] == 'dp']
                        if dp_names:
                            arr = np.atleast_1d(np.asarray(decision, dtype=float))
                            decision = {name: arr[i:i+1].flatten()
                                        for i, name in enumerate(dp_names)}
                else:
                    ctx = {'t': t, 'exogenous': w, **sim_kwargs}
                    decision = policy(state, **ctx)

                feasible = True
                if decision is None:
                    feasible = False
                elif self._feasibility_fn is not None:
                    feasible = bool(self._feasibility_fn(
                        state, decision, w, t))
                elif policy._type in (PolicyType.CFA, PolicyType.VFA,
                                      PolicyType.DLA):
                    feasible = decision is not None and len(decision) > 0

                if not feasible:
                    n_infeasible += 1

                if self._cost_fn is not None:
                    step_cost = float(self._cost_fn(state, decision, w, t))
                else:
                    step_cost = -float(self._reward_fn(
                        state, decision, w, t))

                step_disc = (discount ** t) * step_cost
                total += step_cost
                disc_total += step_disc

                obj_val = step_cost
                if save_trace:
                    trace.record(state, decision, obj_val, step_cost,
                                 feasible, w, t)

                state = self._transition_fn(state, decision, w, t)
                if use_dict_state:
                    if not isinstance(state, dict):
                        state = dict(zip(
                            (name for name, v in self._sdp_vars.items()
                             if v['type'] == 'state'),
                            np.atleast_2d(state)))
                else:
                    state = np.asarray(state, dtype=float)

            if self._terminal_cost_fn is not None:
                term = float(self._terminal_cost_fn(state))
                total += term
                disc_total += (discount ** T) * term

            all_total_costs[n] = total
            all_discounted_costs[n] = disc_total
            all_n_infeasible[n] = n_infeasible
            if save_trace:
                traces.append(trace)

            if show_log and (n + 1) % max(1, N // 10) == 0:
                _log_message(
                    f"Sim {n + 1}/{N}  "
                    f"cost={total:.4f}  disc_cost={disc_total:.4f}")

        return SimulationResult(
            total_costs=all_total_costs,
            discounted_costs=all_discounted_costs,
            n_infeasible=all_n_infeasible,
            traces=traces,
            T=T,
            N=N,
            discount=discount,
            policy_name=policy.name,
            policy_type=policy._type.value,
        )

    def evaluate(self, policy, S0, T, N=100, seed=None, **kw):
        """Shorthand: simulate and return stats dict."""
        r = self.simulate(policy, S0, T, N, seed=seed, **kw)
        return r.stats

    def compare(self, policies, S0, T, N=100, seed=None, show_log=False):
        """Evaluate multiple policies and return a comparison table."""
        results = {}
        for p in policies:
            if show_log:
                _log_message(f"Policy: {p.name} ({p._type.value})")
            r = self.simulate(p, S0, T, N, seed=seed, show_log=False)
            results[p.name] = r.stats
        return results

    def optimize(self, policy, S0, T, N_eval=50, max_iters=50,
                 seed=None, theta_bounds=None, show_log=False,
                 solver='orig-de', interface='mealpy', solver_options=None,
                 obj_operators=None):
        """Optimise policy theta via feloopy's simulation-optimisation."""
        import feloopy as flp

        if not hasattr(policy, 'theta'):
            raise SequentialError(
                "optimize requires a policy with tunable theta "
                "(PFA, CFA, or VFA).")
        if len(policy.theta) == 0:
            raise SequentialError("policy.theta is empty; nothing to optimise.")

        n_params = len(policy.theta)
        if theta_bounds is None:
            theta_bounds = [[-100.0, 100.0]] * n_params
        elif len(theta_bounds) != n_params:
            raise SequentialError(
                f"theta_bounds has {len(theta_bounds)} entries but "
                f"theta has {n_params} dimensions.")

        _orig_theta = policy.theta.copy()
        _direction = self._direction
        _convergence = []
        _n_evals = [0]

        def _objective(theta_vec):
            policy.theta = np.asarray(theta_vec, dtype=float)
            _n_evals[0] += 1
            try:
                r = self.simulate(
                    policy, S0, T, N=N_eval, seed=seed, save_trace=False)
                val = (r.stats['mean_cost'] if _direction == 'min'
                       else -r.stats['mean_cost'])
                _convergence.append(val)
                return val
            except Exception as e:
                return 1e12 if _direction == 'min' else -1e12

        _init_val = _objective(_orig_theta)

        _lb = np.array([b[0] for b in theta_bounds], dtype=float)
        _ub = np.array([b[1] for b in theta_bounds], dtype=float)
        _opt_theta = [None]
        _opt_val = [float('inf')]

        _opts = dict(solver_options or {})
        if 'epoch' not in _opts:
            _opts['epoch'] = max_iters

        def _agent_to_theta(agent):
            a = np.asarray(agent, dtype=float).ravel()[:n_params]
            return _lb + a * (_ub - _lb)

        def _fitness_fn(agent):
            theta_vec = _agent_to_theta(agent)
            return _objective(theta_vec)

        _iface = (interface or 'feloopy').lower()
        _solv = (solver or 'ga').lower()

        if _iface == 'feloopy':
            from ...heuristic.GA import GA
            _ga = GA(
                f=n_params, d=['min'],
                s=_opts.get('epoch', max_iters),
                t=_opts.get('pop_size', 10),
                sc=0,
                cr=_opts.get('crossover_rate', 0.7),
                mu=_opts.get('mutation_rate', 0.02),
                sfl=0.5, sfu=1.0,
            )
            def _ga_fitness(pop):
                for idx in range(pop.shape[0]):
                    val = _fitness_fn(pop[idx, :n_params])
                    if _opt_theta[0] is None or val < _opt_val[0]:
                        _opt_val[0] = val
                        _opt_theta[0] = _agent_to_theta(pop[idx, :n_params]).copy()
                    pop[idx, -1] = val
                return pop
            _ga.run(_ga_fitness)

        elif _iface == 'mealpy':
            from mealpy import FloatVar, SCA, PSO, ABC, GA as MealpyGA
            _orig_de = _mealpy_algo_class(_solv if _solv in (
                'orig-de', 'de') else 'orig-de')
            _mealpy_map = {
                'orig-de': _orig_de, 'de': _orig_de,
                'sca': SCA, 'pso': PSO, 'abc': ABC,
                'ga': MealpyGA,
            }
            _algo_cls = _mealpy_map.get(_solv, _orig_de)
            _pop_size = _opts.get('pop_size', 10)
            _epoch = _opts.get('epoch', max_iters)
            _problem = {
                'obj_func': _fitness_fn,
                'bounds': FloatVar(lb=(0.0,) * n_params, ub=(1.0,) * n_params),
                'minmax': 'min',
                'log_to': None,
                'save_population': False,
            }
            _algo = _algo_cls(epoch=_epoch, pop_size=_pop_size)
            _g_best = _algo.solve(_problem)
            best_agent = np.asarray(_g_best.solution).ravel()[:n_params]
            _opt_theta[0] = _agent_to_theta(best_agent).copy()
            _opt_val[0] = _g_best.target.fitness

        elif _iface == 'niapy':
            from niapy.problems import Problem
            from niapy.task import Task
            from niapy.algorithms.basic import (
                DifferentialEvolution, ParticleSwarmOptimization,
                GeneticAlgorithm, GreyWolfOptimizer,
            )
            _niapy_map = {
                'de': DifferentialEvolution, 'differential-evolution': DifferentialEvolution,
                'pso': ParticleSwarmOptimization, 'particle-swarm': ParticleSwarmOptimization,
                'ga': GeneticAlgorithm, 'genetic': GeneticAlgorithm,
                'gwo': GreyWolfOptimizer, 'grey-wolf': GreyWolfOptimizer,
            }
            _algo_cls = _niapy_map.get(_solv, DifferentialEvolution)
            _epoch = _opts.get('epoch', max_iters)

            class _NiaProblem(Problem):
                def _evaluate(self, x):
                    return _fitness_fn(np.array(x))

            _task = Task(problem=_NiaProblem(n_params), max_iters=_epoch)
            _algo = _algo_cls()
            _best_agent, _best_reward = _algo.run(_task)
            _opt_theta[0] = _agent_to_theta(_best_agent).copy()
            _opt_val[0] = _best_reward

        elif _iface == 'indago':
            import indago
            _indago_name_map = {
                'de': 'DE', 'pso': 'PSO', 'gwo': 'GWO',
                'abc': 'ABC', 'fwa': 'FWA', 'ba': 'BA',
                'efo': 'EFO', 'mrfo': 'MRFO', 'nm': 'NM',
                'msgd': 'MSGD', 'rs': 'RS', 'hbo': 'HBO', 'crs': 'CRS',
            }
            _opt_name = _indago_name_map.get(_solv, _solv.upper())
            if _opt_name not in indago.optimizers_dict:
                raise SequentialError(f"Unknown indago optimizer: {_opt_name}")
            _optimizer = indago.optimizers_dict[_opt_name]()
            _optimizer.evaluation_function = lambda X: float(_fitness_fn(X))
            _optimizer.lb = np.zeros(n_params)
            _optimizer.ub = np.ones(n_params)
            _optimizer.monitoring = 'none'
            _epoch = _opts.get('epoch', max_iters)
            _optimizer.max_evaluations = _epoch * _opts.get('pop_size', 10)
            _result = _optimizer.optimize()
            _opt_theta[0] = _agent_to_theta(np.array(_result.X)).copy()
            _opt_val[0] = float(_result.f)

        elif _iface == 'pygad':
            import pygad
            _pop_size = _opts.get('pop_size', 10)
            _epoch = _opts.get('epoch', max_iters)

            def _pygad_fitness(ga_instance, sol, solution_idx):
                return _fitness_fn(sol)

            _ga = pygad.GA(
                num_generations=_epoch,
                num_parents_mating=max(2, _pop_size // 2),
                sol_per_pop=_pop_size,
                num_genes=n_params,
                fitness_func=_pygad_fitness,
                gene_space=[{'low': 0, 'high': 1}] * n_params,
                suppress_warnings=True,
                fitness_func_type='min',
            )
            _ga.run()
            _best_agent, _best_reward, _ = _ga.best_solution()
            _opt_theta[0] = _agent_to_theta(_best_agent).copy()
            _opt_val[0] = float(_best_reward)

        elif _iface == 'scipy':
            from scipy.optimize import minimize as sp_minimize
            _method_map = {
                'nelder-mead': 'Nelder-Mead', 'powell': 'Powell',
                'cobyla': 'COBYLA', 'slsqp': 'SLSQP',
                'l-bfgs-b': 'L-BFGS-B', 'lbfgsb': 'L-BFGS-B',
            }
            _sp_method = _method_map.get(_solv, _solv.upper())
            _x0 = np.full(n_params, 0.5)

            _bounds_only = _sp_method in ('Nelder-Mead', 'Powell', 'COBYLA')
            if _bounds_only:
                _res = sp_minimize(_fitness_fn, _x0, method=_sp_method)
            else:
                _sp_bounds = [(0, 1)] * n_params
                _res = sp_minimize(_fitness_fn, _x0, method=_sp_method, bounds=_sp_bounds)
            _opt_theta[0] = _agent_to_theta(_res.x).copy()
            _opt_val[0] = _res.fun

        else:
            raise SequentialError(
                f"Unknown interface '{_iface}' for optimize. "
                f"Supported: feloopy, mealpy, niapy, indago, pygad, scipy.")

        best_theta = _opt_theta[0] if _opt_theta[0] is not None \
            else _orig_theta
        policy.theta = best_theta.copy()
        best_r = self.simulate(
            policy, S0, T, N=N_eval, seed=seed, save_trace=False)

        return OptimizationResult(
            best_theta=best_theta,
            best_stats=best_r.stats,
            init_theta=_orig_theta,
            init_stats={'mean_cost': _init_val if _direction == 'min'
                        else -_init_val},
            convergence=_convergence,
            n_evaluations=_n_evals[0],
            solver=solver,
            direction=_direction,
        )
