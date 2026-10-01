# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


import picos as picos_interface


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            return input2.value

        case 'status':

            return result[0][0].claimedStatus

        case 'objective':

            return model_object.value

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return None

        case 'dual':
            constraint_dict = result[0][1]
            if isinstance(input2, str) and input2 in constraint_dict:
                con_obj = constraint_dict[input2]
                try:
                    return con_obj.dual
                except Exception:
                    return None
            elif hasattr(input2, '_id'):
                try:
                    return model_object.constraints[input2._id].dual
                except Exception:
                    return None
            return None

        case 'slack':
            constraint_dict = result[0][1]
            if isinstance(input2, str) and input2 in constraint_dict:
                con_obj = constraint_dict[input2]
                try:
                    return con_obj.slack
                except Exception:
                    return None
            elif hasattr(input2, '_id'):
                try:
                    return model_object.constraints[input2._id].slack
                except Exception:
                    return None
            return None

