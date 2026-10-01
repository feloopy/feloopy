# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import pyomo.environ as pyomo_interface
import timeit
import os

from ...helpers.solver_executables import (
    ensure_solver_on_path,
    inject_solver_paths,
    missing_solver_message,
)

inject_solver_paths()

pyomo_offline_solver_selector = {
    'baron': 'baron',
    'bonmin': 'bonmin',
    'cbc': 'cbc',
    'conopt': 'conopt',
    'couenne': 'couenne',
    'cplex': 'cplex',
    'cplex_direct': 'cplex_direct',
    'cplex-direct': 'cplex_direct',
    'cplex_persistent': 'cplex_persistent',
    'cplex-persistent': 'cplex_persistent',
    'cyipopt': 'cyipopt',
    'gams': 'gams',
    'highs': 'highs',
    'asl': 'asl',
    'gdpopt': 'gdpopt',
    'gdpopt.gloa': 'gdpopt.gloa',
    'gdpopt.lbb': 'gdpopt.lbb',
    'gdpopt.loa': 'gdpopt.loa',
    'gdpopt.ric': 'gdpopt.ric',
    'glpk': 'glpk',
    'gurobi': 'gurobi',
    'gurobi_direct': 'gurobi_direct',
    'gurobi-direct': 'gurobi_direct',
    'gurobi_persistent': 'gurobi_persistent',
    'gurobi-persistent': 'gurobi_persistent',
    'ipopt': 'ipopt',
    'mindtpy': 'mindtpy',
    'mosek': 'mosek',
    'mosek_direct': 'mosek_direct',
    'mosek-direct': 'mosek_direct',
    'mosek_persistent': 'mosek_persistent',
    'mosek-persistent': 'mosek_persistent',
    'mpec_minlp': 'mpec_minlp',
    'mpec-minlp': 'mpec_minlp',
    'mpec_nlp': 'mpec_nlp',
    'mpec-nlp': 'mpec_nlp',
    'multistart': 'multistart',
    'path': 'path',
    'scip': 'scip',
    'trustregion': 'trustregion',
    'xpress': 'xpress',
    'xpress_direct': 'xpress_direct',
    'xpress-direct': 'xpress_direct',
    'xpress_persistent': 'xpress_persistent',
    'xpress-persistent': 'xpress_persistent'
}

pyomo_online_solver_selector = {
    'bonmin_online': 'bonmin',
    'cbc_online': 'cbc',
    'conopt_online': 'conopt',
    'couenne_online': 'couenne',
    'cplex_online': 'cplex',
    'filmint_online': 'filmint',
    'filter_online': 'filter',
    'ipopt_online': 'ipopt',
    'knitro_online': 'knitro',
    'l-bfgs-b_online': 'l-bfgs-variable_bound',
    'lancelot_online': 'lancelot',
    'lgo_online': 'lgo',
    'loqo_online': 'loqo',
    'minlp_online': 'minlp',
    'minos_online': 'minos',
    'minto_online': 'minto',
    'mosek_online': 'mosek',
    'octeract_online': 'octeract',
    'ooqp_online': 'ooqp',
    'path_online': 'path',
    'raposa_online': 'raposa',
    'snopt_online': 'snopt'
}


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    directions = features['directions']
    constraint_labels = features['constraint_labels']
    debug = features['debug_mode']
    time_limit = features['time_limit']
    absolute_gap = features['absolute_gap']
    relative_gap = features['relative_gap']
    thread_count = features['thread_count']
    solver_name = features['solver_name']
    objective_id = features['objective_being_optimized']
    log = features['log']
    save = features['save_solver_log']
    save_model = features['write_model_file']
    email = features['email_address']
    max_iterations = features['max_iterations']
    solver_options = features['solver_options']

    if log:
        tee = True
    else:
        tee = False

    if max_iterations != None:
        "None"

    match debug:

        case False | True:

            if model_object.component('OBJ') is not None:
                model_object.del_component('OBJ')

            match directions[objective_id]:

                case "min":
                    model_object.OBJ = pyomo_interface.Objective(
                        expr=model_objectives[objective_id], sense=pyomo_interface.minimize)

                case "max":
                    model_object.OBJ = pyomo_interface.Objective(
                        expr=model_objectives[objective_id], sense=pyomo_interface.maximize)

            has_int = features.get('integer_variable_counter', [0])[0] > 0
            has_bin = features.get('binary_variable_counter', [0])[0] > 0
            is_mip = has_int or has_bin

            if len(model_constraints)!=0:
                all_none = all(label is None for label in constraint_labels)
                if all_none:
                    if model_object.component('constraint') is not None:
                        model_object.del_component('constraint')
                    model_object.constraint = pyomo_interface.ConstraintList()
                    for element in model_constraints:
                        model_object.constraint.add(expr=element)
                else:
                    if not is_mip:
                        if model_object.component('dual') is not None:
                            model_object.del_component('dual')
                        model_object.dual = pyomo_interface.Suffix(direction=pyomo_interface.Suffix.IMPORT)
                    if model_object.component('c') is not None:
                        model_object.del_component('c')
                    model_object.c = pyomo_interface.Constraint(pyomo_interface.Any)
                    counter=0
                    auto_idx = 0
                    for element in model_constraints:
                        label = constraint_labels[counter]
                        if label is None:
                            label = f'_auto_constraint_{auto_idx}'
                            auto_idx += 1
                        model_object.c[label] = element
                        counter+=1

            if 'online' not in solver_name:

                if solver_name not in pyomo_offline_solver_selector.keys():

                    raise RuntimeError(
                        "Using solver '%s' is not supported by 'pyomo'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

                # First explicit use: fetch a free binary once if it is
                # missing (set FELOOPY_NO_AUTO_DOWNLOAD=1 to opt out).
                # Commercial and pip-backed solvers are never downloaded
                # implicitly — those only get an install hint below.
                from ...helpers.solver_provisioning import (
                    ensure_solver,
                    is_provisionable,
                )
                from ...helpers.solver_guides import resolve_commercial

                executable = ensure_solver(solver_name)
                if executable:
                    ensure_solver_on_path(executable)
                elif is_provisionable(solver_name):
                    raise RuntimeError(missing_solver_message(solver_name))
                else:
                    inject_solver_paths((solver_name,))

                if solver_name == 'cbc':
                    solver_manager = pyomo_interface.SolverFactory(
                        pyomo_offline_solver_selector[solver_name], solver_io='nl')
                else:
                    solver_manager = pyomo_interface.SolverFactory(
                        pyomo_offline_solver_selector[solver_name])

                # Backend missing? (binding not installed, no license, ...)
                # Provisionable/commercial names get feloopy's install hint;
                # everything else (gdpopt, mindtpy, ...) keeps pyomo's own
                # error path.
                try:
                    backend_available = solver_manager.available(exception_flag=False)
                except Exception:
                    backend_available = True  # unknown plugin: let pyomo report it
                if not backend_available and (
                        is_provisionable(solver_name)
                        or resolve_commercial(solver_name)):
                    raise RuntimeError(missing_solver_message(solver_name))

                if thread_count != None:

                    solver_manager.options['threads'] = thread_count

                if time_limit != None:

                    solver_manager.options['timelimit'] = time_limit

                if relative_gap != None:

                    solver_manager.options['mipgap'] = relative_gap

                if solver_name.endswith('-persistent') or solver_name.endswith('_persistent'):
                    solver_manager.set_instance(model_object)

                gdpopt_algorithm = None
                if solver_name in ('gdpopt',):
                    solver_options = dict(solver_options)
                    gdpopt_algorithm = solver_options.pop('algorithm', 'LOA')

                mindtpy_kwargs = {}
                if solver_name == 'mindtpy':
                    if 'mip_solver' not in solver_options:
                        solver_options = dict(solver_options)
                        for cand in (
                            'gurobi', 'cplex', 'cbc', 'glpk', 'gams',
                            'gurobi_persistent', 'cplex_persistent',
                            'appsi_cplex', 'appsi_gurobi', 'appsi_highs',
                        ):
                            try:
                                if pyomo_interface.SolverFactory(cand).available(exception_flag=False):
                                    solver_options['mip_solver'] = cand
                                    break
                            except Exception:
                                continue
                    mindtpy_kwargs = dict(solver_options)
                    solver_options = {}

                warmstart_flag = bool(features.get('_pyomo_warmstart'))
                if warmstart_flag:
                    _capable = getattr(solver_manager, 'warm_start_capable', None)
                    if callable(_capable):
                        try:
                            if not _capable():
                                warmstart_flag = False
                                from ..init_generator import warn_start_dropped
                                warn_start_dropped(solver_name)
                        except Exception:
                            pass

                def _try_solve(model, *args, **kwargs):
                    if warmstart_flag and 'warmstart' not in kwargs:
                        try:
                            return solver_manager.solve(
                                model, *args, warmstart=True, **kwargs)
                        except (TypeError, ValueError) as e:
                            if 'warmstart' not in str(e):
                                raise

                            from ..init_generator import warn_start_dropped
                            warn_start_dropped(solver_name, e)
                    try:
                        return solver_manager.solve(model, *args, **kwargs)
                    except Exception as e:
                        from pyomo.contrib.solver.common.util import NoFeasibleSolutionError, NoDualsError
                        if isinstance(e, NoDualsError):
                            try:
                                return solver_manager.solve(model, *args, load_solutions=False, **kwargs)
                            except Exception:
                                pass
                        if not isinstance(e, (NoFeasibleSolutionError, NoDualsError)):
                            raise
                        from pyomo.opt import SolverResults
                        results = SolverResults()
                        results.solver.termination_condition = 'infeasible'
                        return results

                if solver_name == 'trustregion':
                    dof_vars = list(model_object.component_data_objects(
                        pyomo_interface.Var, active=True))
                    time_solve_begin = timeit.default_timer()
                    result = _try_solve(model_object, dof_vars, tee=tee)
                    time_solve_end = timeit.default_timer()

                elif solver_name == 'multistart':
                    time_solve_begin = timeit.default_timer()
                    result = _try_solve(model_object)
                    time_solve_end = timeit.default_timer()

                elif mindtpy_kwargs:
                    time_solve_begin = timeit.default_timer()
                    result = _try_solve(model_object, tee=tee, **mindtpy_kwargs)
                    time_solve_end = timeit.default_timer()

                elif gdpopt_algorithm is not None:
                    time_solve_begin = timeit.default_timer()
                    if len(solver_options) == 0:
                        result = _try_solve(model_object, tee=tee, algorithm=gdpopt_algorithm)
                    else:
                        result = _try_solve(model_object, tee=tee, algorithm=gdpopt_algorithm, options=solver_options)
                    time_solve_end = timeit.default_timer()

                elif len(solver_options) == 0:

                    time_solve_begin = timeit.default_timer()
                    result = _try_solve(model_object, tee=tee)
                    time_solve_end = timeit.default_timer()

                else:

                    time_solve_begin = timeit.default_timer()
                    result = _try_solve(model_object, tee=tee, options=solver_options)
                    time_solve_end = timeit.default_timer()

            else:

                if solver_name not in pyomo_online_solver_selector.keys():

                    raise RuntimeError(
                        "Using solver '%s' is not supported by 'pyomo'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

                os.environ['NEOS_EMAIL'] = email
                solver_manager = pyomo_interface.SolverManagerFactory('neos')
                time_solve_begin = timeit.default_timer()
                result = solver_manager.solve(
                    model_object, solver=pyomo_online_solver_selector[solver_name], tee=tee)
                time_solve_end = timeit.default_timer()

            generated_solution = result, [time_solve_begin, time_solve_end]

    return generated_solution
