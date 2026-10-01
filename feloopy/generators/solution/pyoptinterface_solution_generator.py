# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyoptinterface as poi
import timeit

poi_solver_selector = {
    'highs': 'highs',
    'copt': 'copt',
    'gurobi': 'gurobi',
    'mosek': 'mosek',
}

SENSE_MAP = {
    '<=': poi.Leq,
    '>=': poi.Geq,
    '==': poi.Eq,
    '<': poi.Leq,
    '>': poi.Geq,
}


def generate_solution(features):

    model_object = features['model_object_before_solve']
    model_objectives = features['objectives']
    model_constraints = features['constraints']
    constraint_labels = features['constraint_labels']
    directions = features['directions']
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
    callback = features.get('callback', None)

    if solver_name not in poi_solver_selector.keys():
        raise RuntimeError(
            "Using solver '%s' is not supported by 'pyoptinterface'! \nPossible fixes: \n1) Check the solver name. \n2) Use another interface. \n" % (solver_name))

    if log:
        model_object.set_model_attribute(poi.ModelAttribute.Silent, False)
    else:
        model_object.set_model_attribute(poi.ModelAttribute.Silent, True)

    if time_limit is not None:
        model_object.set_model_attribute(poi.ModelAttribute.TimeLimitSec, time_limit)

    if thread_count is not None:
        try:
            model_object.set_model_attribute(poi.ModelAttribute.NumberOfThreads, thread_count)
        except Exception:
            pass

    if relative_gap is not None:
        model_object.set_model_attribute(poi.ModelAttribute.RelativeGap, relative_gap)

    if len(solver_options) != 0:
        for key, value in solver_options.items():
            if key.startswith("---"):
                continue
            if value is None:
                continue
            try:
                attr = getattr(poi.ModelAttribute, key, None)
                if attr is not None:
                    model_object.set_model_attribute(attr, value)
            except Exception:
                pass

    match debug:

        case False | True:

            poi_constraint_indices = {}
            poi_constraint_meta = {}

            for i, constraint in enumerate(model_constraints):
                label = constraint_labels[i] if i < len(constraint_labels) else None
                if hasattr(constraint, 'lhs') and hasattr(constraint, 'rhs') and hasattr(constraint, 'sense'):
                    cidx = model_object.add_linear_constraint(constraint.lhs, constraint.sense, constraint.rhs)
                    meta = {'sense': constraint.sense, 'rhs': constraint.rhs}
                elif isinstance(constraint, (list, tuple)) and len(constraint) >= 3:
                    lhs, sense, rhs = constraint[0], constraint[1], constraint[2]
                    epsilon = constraint[3] if len(constraint) >= 4 else 0
                    if sense == '<' and epsilon > 0:
                        sense = '<='
                        rhs = rhs - epsilon
                    elif sense == '>' and epsilon > 0:
                        sense = '>='
                        rhs = rhs + epsilon
                    if isinstance(sense, str) and sense in SENSE_MAP:
                        sense = SENSE_MAP[sense]
                    cidx = model_object.add_linear_constraint(lhs, sense, rhs)
                    meta = {'sense': sense, 'rhs': rhs}
                else:
                    continue
                if label is not None:
                    poi_constraint_indices[label] = cidx
                    poi_constraint_meta[label] = meta

            model_object._feloopy_constraint_indices = poi_constraint_indices
            model_object._feloopy_constraint_meta = poi_constraint_meta

            match directions[objective_id]:

                case 'min':
                    model_object.set_objective(model_objectives[objective_id], poi.ObjectiveSense.Minimize)

                case 'max':
                    model_object.set_objective(model_objectives[objective_id], poi.ObjectiveSense.Maximize)

            if save_model is not False:
                try:
                    model_object.write(str(save_model))
                except Exception:
                    pass

            from ..init_generator import flush_init
            flush_init(features, force=True)

            time_solve_begin = timeit.default_timer()
            model_object.optimize()
            time_solve_end = timeit.default_timer()

            poi_variable_indices = {}
            try:
                nv = model_object.number_of_variables()
                for i in range(nv):
                    v = poi.VariableIndex(i)
                    vn = model_object.get_variable_name(v)
                    poi_variable_indices[vn] = v
            except Exception:
                pass
            model_object._feloopy_variable_indices = poi_variable_indices

            generated_solution = [None, [time_solve_begin, time_solve_end]]

    return generated_solution
