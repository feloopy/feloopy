# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mip as mip_interface


def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':

            return input2.x

        case 'status':

            return result[0]

        case 'objective':

            return model_object.objective_value

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return model_object.objective_bound

        case 'dual':

            return None

        case 'slack':

            return None
