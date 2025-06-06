# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def Get(model_object, result, input1, input2=None):

    input1 = input1[0]

    match input1:

        case 'variable':
            return result[0].value(input2)

        case 'status':
            return model_object.return_status()

        case 'objective':
            return result[0].value(model_object.f)

        case 'time':
            return (result[1][1] - result[1][0])