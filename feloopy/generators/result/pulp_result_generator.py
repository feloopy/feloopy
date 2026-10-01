# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import pulp as pulp_interface

from ..pulp_compat import PULP_V4, get_constraint


def Get(model_object, result, input1, input2=None, input=None):

    if input and PULP_V4:
        model_object = input.get('model_object_before_solve', model_object)

    directions = +1 if input1[1][input1[2]] == 'min' else -1
    input1 = input1[0]

    match input1:

        case 'variable':

            if isinstance(input2, str):
                for v in model_object.variables():
                    if v.name == input2:
                        return v.varValue
                return None
            return input2.varValue

        case 'status':

            if PULP_V4:
                if result.status == pulp_interface.LpSolveStatus.GapLimit:
                    return 'Optimal'
                return result.status_str
            return pulp_interface.LpStatus[result[0]]

        case 'objective':

            if PULP_V4:
                try:
                    return directions * result.objective
                except TypeError:
                    return None
            return directions*pulp_interface.value(model_object.objective)

        case 'time':

            if PULP_V4:
                return result.time
            return (result[1][1]-result[1][0])

        case 'bound':

            if PULP_V4:
                return result.best_bound
            return None

        case 'dual':

            try:
                constraint = get_constraint(model_object, input2)
                if constraint is not None:
                    sense = constraint.sense
                    pi_val = constraint.pi

                    if sense == pulp_interface.LpConstraintEQ:
                        return pi_val

                    elif sense == pulp_interface.LpConstraintLE and directions==1:
                        return -1*abs(pi_val)

                    elif sense == pulp_interface.LpConstraintLE and directions==-1:
                        return 1*abs(pi_val)

                    elif sense == pulp_interface.LpConstraintGE and directions==1:
                        return abs(pi_val)

                    elif sense == pulp_interface.LpConstraintGE and directions==-1:
                        return -1*abs(pi_val)
                return None
            except Exception:
                return None

        case 'slack':

            constraint = get_constraint(model_object, input2)
            if constraint is not None:
                return abs(constraint.slack)

        case 'rc':
            if isinstance(input2, str):
                for v in model_object.variables():
                    if v.name == input2:
                        return directions * v.dj
                return None
            return directions*input2.dj
