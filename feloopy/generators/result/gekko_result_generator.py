# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import json
import os
import gekko as gekko_interface

gekko_status_dict = {0: "unknown", 1: "optimal", 2: "optimal", 3: "feasible"}


def _read_results(model_object):
    try:
        results_path = os.path.join(model_object.path, 'results.json')
        with open(results_path) as f:
            return json.load(f)
    except Exception:
        return {}


def Get(model_object, result, input1, input2=None):

    infeasible = getattr(model_object, '_feloopy_infeasible', False)

    if infeasible:
        match input1[0]:
            case 'status':
                return 'infeasible'
            case 'objective':
                return None
            case 'time':
                return (result[1][1] - result[1][0]) if result else 0
            case 'variable':
                return None
            case 'dual' | 'rc' | 'bound':
                return None
            case 'slack':
                return None

    directions = +1 if input1[1][input1[2]] == 'min' else -1

    input1 = input1[0]

    match input1:

        case 'variable':

            if isinstance(input2, str):
                for v in model_object._variables:
                    if v.name == input2 or v.name == f'int_{input2}':
                        return v.value[0]
                underscore_name = input2
                for i in range(len(input2)):
                    if input2[i:].isdigit():
                        underscore_name = input2[:i] + '_' + input2[i:]
                        break
                for v in model_object._variables:
                    if v.name == underscore_name or v.name == f'int_{underscore_name}':
                        return v.value[0]
                return None
            try:
                return input2.value[0]
            except Exception:
                return input2

        case 'status':

            return gekko_status_dict.get(model_object.options.SOLVESTATUS, 'unknown')

        case 'objective':

            return directions * model_object.options.objfcnval

        case 'time':

            return (result[1][1] - result[1][0])

        case 'bound':

            return None

        case 'dual':

            return None

        case 'slack':

            data = _read_results(model_object)
            if not data:
                return None

            constraint_labels = getattr(model_object, '_constraint_labels', [])

            if isinstance(input2, str):
                if input2 in constraint_labels:
                    idx = constraint_labels.index(input2)
                    slk_key = f'slk_{idx + 1}'
                    if slk_key in data:
                        return data[slk_key][0]
                    return 0.0

            return None

        case 'rc':

            return None
