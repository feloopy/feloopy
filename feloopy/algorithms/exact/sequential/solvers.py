import numpy as np


def _solve_model(m, method='exact', interface=None, solver=None,
                 solver_options=None, directions=None,
                 obj_operators=None, uncertainty_set_constraints=None):
    """Dispatch a feloopy model to the appropriate solver backend."""
    import feloopy as flp

    if directions is None:
        directions = m.features.get('directions', ['min'])

    if method == 'heuristic':
        _interface = interface or 'mealpy'
        kwargs = dict(directions=directions, solver=solver)
        if solver_options:
            kwargs['options'] = solver_options
        r = flp.search(
            lambda inner_m: m,
            method='heuristic',
            interface=_interface,
            **kwargs,
            verbose=False,
            repeat=1,
            should_run=True,
        )
        return r

    else:
        kwargs = dict(directions=directions, solver=solver)
        if solver_options:
            kwargs['solver_options'] = solver_options
        if obj_operators:
            kwargs['obj_operators'] = obj_operators
        if uncertainty_set_constraints:
            kwargs['uncertainty_set_constraints'] = uncertainty_set_constraints
        m.sol(**kwargs)
        return m


def _extract_decision(m, var_names=None):
    """Extract decision variable values from a solved model or search result."""
    if hasattr(m, 'solutions') and isinstance(m.solutions, dict):
        vals = []
        if var_names is not None:
            for vn in var_names:
                if vn in m.solutions:
                    vals.append(np.asarray(m.solutions[vn]).ravel())
        else:
            for vn, v in m.solutions.items():
                vals.append(np.asarray(v).ravel())
        return np.concatenate(vals) if len(vals) > 1 else (
            vals[0] if vals else np.array([]))

    if var_names is not None:
        vals = []
        for vn in var_names:
            vals.append(np.asarray(m.get(vn)).ravel())
        return np.concatenate(vals) if len(vals) > 1 else vals[0]

    vals = []
    for (typ, vn) in m.features.get('variables', {}).keys():
        if typ == 'evar':
            continue
        vals.append(np.asarray(m.get(vn)).ravel())
    return np.concatenate(vals) if len(vals) > 1 else (
        vals[0] if vals else np.array([]))


def sol_sequential(
    environment,
    transition,
    cost=None,
    reward=None,
    exogenous=None,
    terminal_cost=None,
    policy=None,
    S0=None,
    T=10,
    N=100,
    seed=None,
    discount=1.0,
    show_log=False,
    save_trace=True,
    compare=None,
    search_theta=False,
    search_iters=20,
    search_eval=50,
    opt_solver='orig-de',
    opt_interface='mealpy',
    opt_solver_options=None,
    obj_operators=None,
    **sim_kwargs,
):
    """High-level entry point for sequential decision making."""
    from .enums import SequentialError
    from .core import SequentialDecisionProblem

    sdp = SequentialDecisionProblem()
    sdp.environment(environment)
    sdp.transition(transition)
    if cost is not None:
        sdp.cost(cost)
    if reward is not None:
        sdp.reward(reward)
    if exogenous is not None:
        sdp.exogenous(exogenous)
    if terminal_cost is not None:
        sdp.terminal_cost(terminal_cost)

    if policy is None:
        raise SequentialError("policy is required.")

    if S0 is None:
        raise SequentialError("S0 (initial state) is required.")

    if search_theta and hasattr(policy, 'theta') and len(policy.theta) > 0:
        # optimize() returns an OptimizationResult and also writes the best
        # theta back onto the policy (which the simulation below reuses)
        _opt = sdp.optimize(
            policy, S0, T, N_eval=search_eval, max_iters=search_iters,
            seed=seed, show_log=show_log, solver=opt_solver,
            interface=opt_interface, solver_options=opt_solver_options,
            obj_operators=obj_operators)
        if show_log:
            from .enums import _log_message
            _log_message(f"Best theta: {_opt.best_theta}")

    if compare is not None:
        all_policies = [policy] + list(compare)
        return sdp.compare(all_policies, S0, T, N=N, seed=seed,
                           show_log=show_log)

    return sdp.simulate(
        policy, S0, T, N=N, seed=seed, discount=discount,
        show_log=show_log, save_trace=save_trace, **sim_kwargs)


def sdm(environment=None, policy='pfa', theta0=None,
        S0=None, T=10, N=100, discount=1.0, seed=None,
        optimize=False, theta_bounds=None, max_iters=50,
        solver='orig-de', options=None,
        name='sdm_model', directions=None, **kwargs):
    """Convenience wrapper: flp.sdm(...) is equivalent to flp.search(..., method='sequential')."""
    import feloopy as flp
    return flp.search(
        environment=environment,
        method='sequential',
        name=name,
        directions=directions,
        verbose=kwargs.pop('verbose', False),
        policy=policy,
        theta0=theta0,
        S0=S0,
        T=T,
        N=N,
        discount=discount,
        seed=seed,
        save_trace=kwargs.pop('save_trace', True),
        optimize=optimize,
        theta_bounds=theta_bounds,
        max_iters=max_iters,
        solver=solver,
        options=options,
        **kwargs,
    )
