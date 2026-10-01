import numpy as np

from .enums import (
    MultiObjectiveError, _DIR_MAP,
    _validate_directions, _validate_approach, _validate_weights,
)
from .utils import _generate_payoff_table, revise_pareto
from . import ecm, nwsm


def sol_multi(
    instance,
    directions=None,
    solver_name=None,
    solver_options=None,
    objective_id=0,
    email=None,
    debug=False,
    time_limit=None,
    cpu_threads=None,
    absolute_gap=None,
    relative_gap=None,
    show_log=False,
    save_log=False,
    save_model=False,
    max_iterations=None,
    approach_options=None,
    weights=None,
    save_vars=False,
    obj_operators=None,
    uncertainty_set_constraints=None):

    from ....generators import solution_generator
    from ....helpers.mo_progress import MOProgress, NullMOProgress

    if approach_options is None:
        approach_options = {}
    if solver_options is None:
        solver_options = {}
    if weights is None:
        weights = []

    _validate_approach(objective_id)

    solver_opts = dict(
        solver_name=solver_name,
        solver_options=solver_options,
        debug_mode=debug,
        time_limit=time_limit,
        thread_count=cpu_threads,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        log=False,
        write_model_file=save_model,
        save_solver_log=save_log,
        email_address=email,
        max_iterations=max_iterations,
        obj_operators=obj_operators or [],
        uncertainty_set_constraints=uncertainty_set_constraints or [],
    )

    _validate_directions(directions)

    if type(objective_id) != str:
        m = instance()
        m.features['objective_being_optimized'] = objective_id
        m.features['solver_name'] = solver_name
        m.features['solver_options'] = solver_options
        if m.features['directions'][objective_id] is None:
            m.features['directions'][objective_id] = directions[objective_id]
            for i in range(len(m.features['objectives'])):
                if i != objective_id:
                    del m.features['directions'][i]
                    del directions[i]
                    del m.features['objectives'][i]
            objective_id = 0
            m.features['objective_counter'] = [1, 1]
        else:
            for i in range(len(m.features['directions'])):
                m.features['directions'][i] = directions[i]
        m.features['log'] = False
        m.features['model_object_before_solve'] = m
        m.features['debug_mode'] = False
        m.solution = solution_generator.generate_solution(m.features)

    M = len(directions)

    if objective_id == 'nwsm':
        _validate_weights(weights, M)

    intervals = approach_options.get('intervals', 10)

    if show_log:
        progress = MOProgress(objective_id, M, directions, intervals)
        progress.start()
    else:
        progress = NullMOProgress()

    try:
        payoff = None

        if approach_options.get('payoff_method', 'separated') == 'separated':
            payoff = _generate_payoff_table(
                instance, M, directions, solver_opts, solution_generator, progress)

        if payoff is None:
            raise MultiObjectiveError(
                "payoff table was not generated. "
                "Ensure approach_options['payoff_method'] is 'separated'.")

        if objective_id == 'ecm':
            pareto, variables, ogrs = ecm.solve(
                instance, payoff, directions, M, solver_opts, approach_options,
                solution_generator, save_vars, progress)
        elif objective_id == 'nwsm':
            pareto, variables, ogrs = nwsm.solve(
                instance, payoff, directions, M, solver_opts, approach_options,
                solution_generator, save_vars, progress, weights)
        else:
            raise MultiObjectiveError(f"Unknown objective_id: {objective_id!r}")

        pareto, variables = revise_pareto(_DIR_MAP, directions, pareto, variables)

        if pareto.shape[0] >= 2:
            dir_mult = np.array([1 if d == 'max' else -1 for d in directions])
            pareto_flipped = pareto * dir_mult
            corr = np.corrcoef(pareto_flipped.T)
            _M = corr.shape[0]
            if _M > 1:
                mask = ~np.eye(_M, dtype=bool)
                mean_corr = np.mean(corr[mask])
                conflict_metric = 0.0 if np.isnan(mean_corr) else float((1 - mean_corr) / 2)
            else:
                conflict_metric = 0.0
        else:
            corr = np.eye(pareto.shape[1])
            conflict_metric = 0.0

        if show_log:
            progress.finish(pareto.shape[0], conflict_metric)
            progress.stop()

        return pareto, payoff, corr, conflict_metric, variables, ogrs

    except Exception:
        if show_log:
            progress.stop()
        raise
