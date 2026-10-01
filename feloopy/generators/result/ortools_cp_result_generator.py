# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


from ortools.sat.python import cp_model

_status_code_to_name = {
    int(cp_model.UNKNOWN): "unknown",
    int(cp_model.MODEL_INVALID): "model_invalid",
    int(cp_model.INFEASIBLE): "infeasible",
    int(cp_model.FEASIBLE): "feasible",
    int(cp_model.OPTIMAL): "optimal",
}


def _is_solved(result):
    status = int(result[0][0])
    return status in (int(cp_model.OPTIMAL), int(cp_model.FEASIBLE))


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            if not _is_solved(result):
                return None

            solver = result[0][1]
            if isinstance(input2, dict):
                values = {}
                for key, var in input2.items():
                    try:
                        if hasattr(var, 'StartExpr'):
                            values[key] = {
                                'start': solver.Value(var.StartExpr()),
                                'end': solver.Value(var.EndExpr()),
                                'size': solver.Value(var.SizeExpr()),
                            }
                        else:
                            values[key] = solver.Value(var)
                    except Exception:
                        values[key] = None
                return values
            try:
                if hasattr(input2, 'StartExpr'):
                    return {
                        'start': solver.Value(input2.StartExpr()),
                        'end': solver.Value(input2.EndExpr()),
                        'size': solver.Value(input2.SizeExpr()),
                    }
                return solver.Value(input2)
            except Exception:
                return None

        case 'status':

            return _status_code_to_name.get(int(result[0][0]), "not_solved")

        case 'objective':

            if not _is_solved(result):
                return None

            try:
                return result[0][1].ObjectiveValue()
            except Exception:
                return None

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            if not _is_solved(result):
                return None

            try:
                return result[0][1].BestObjectiveBound()
            except Exception:
                return None

        case 'num_branches':

            try:
                return result[0][1].NumBranches()
            except Exception:
                return 0

        case 'num_conflicts':

            try:
                return result[0][1].NumConflicts()
            except Exception:
                return 0

        case 'wall_time':

            try:
                return result[0][1].WallTime()
            except Exception:
                return 0.0

        case 'dual':

            return None

        case 'slack':

            return None
