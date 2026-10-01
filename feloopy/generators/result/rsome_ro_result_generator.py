# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def Get(model_object, result, input1, input2=None):

    directions = +1 if input1[1][input1[2]] == 'min' else -1
    input1 = input1[0]

    match input1:

        case 'variable':
            if hasattr(input2, 'get'):
                return input2.get()
            if isinstance(input2, str):
                var_name = input2.split('[')[0] if '[' in input2 else input2
                for (typ, name), var_obj in model_object.features.get('variables', {}).items():
                    if name == var_name:
                        full_val = var_obj.get()
                        if '[' in input2:
                            idx = input2.split('[')[1].rstrip(']')
                            try:
                                return full_val[int(idx)]
                            except (ValueError, TypeError):
                                return full_val[idx]
                        return full_val
                return model_object.get()
            return input2.get()
            
        case 'status':

            try:
                model_object.get()
                return 'optimal*'
            except:
                return 'not optimal or nothing found'

        case 'objective':

            return model_object.get()

        case 'time':

            return (result[1][1]-result[1][0])

        case 'bound':

            return None

        case 'dual':

            return None

        case 'slack':

            return None
