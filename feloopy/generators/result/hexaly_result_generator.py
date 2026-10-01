# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from hexaly.optimizer import HxSolutionStatus

def Get(model_object, result, input1, input2=None, input=None):

    input1 = input1[0]
    optimizer = (input or {}).get('hexaly_optimizer')

    match input1:

        case 'variable':
            if input2 is None:
                return None
            target = str(input2)
            try:
                expr = model_object.get_expression(target)
                return expr.value
            except Exception:
                pass
            return None

        case 'status':
            if optimizer is None:
                return 'unknown'
            sol_status = optimizer.solution.get_status()
            status_map = {
                HxSolutionStatus.OPTIMAL: 'optimal',
                HxSolutionStatus.FEASIBLE: 'feasible',
                HxSolutionStatus.INFEASIBLE: 'infeasible',
                HxSolutionStatus.INCONSISTENT: 'inconsistent',
            }
            return status_map.get(sol_status, str(sol_status))

        case 'objective':
            try:
                return model_object.objective.value
            except Exception:
                pass
            return None

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            if optimizer is None:
                return None
            try:
                return optimizer.solution.get_objective_bound()
            except Exception:
                pass
            return None

        case 'dual':
            return None

        case 'slack':
            return None
