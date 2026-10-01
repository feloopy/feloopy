# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def _normalize_name(name):
    return (name.replace('[', '').replace(']', '')
                .replace('(', '').replace(')', '').replace(' ', ''))


def Get(model_object, result, input1, input2=None, input=None):
    input1 = input1[0]

    scip_model = result[0]

    match input1:
        case 'variable':
            if isinstance(input2, str):
                try:
                    for v in scip_model.getVars():
                        if v.name == input2:
                            return scip_model.getVal(v)
                    target = _normalize_name(input2)
                    if target:
                        for v in scip_model.getVars():
                            if _normalize_name(v.name) == target:
                                return scip_model.getVal(v)
                    return None
                except Exception:
                    return None
            elif hasattr(input2, 'name'):
                try:
                    for v in scip_model.getVars():
                        if v.name == input2.name:
                            return scip_model.getVal(v)
                    return None
                except Exception:
                    return None
            elif isinstance(input2, int):
                try:
                    vars_list = scip_model.getVars()
                    if input2 < len(vars_list):
                        return scip_model.getVal(vars_list[input2])
                    return None
                except Exception:
                    return None
            return None

        case 'status':
            try:
                status = scip_model.getStatus()
                status_map = {
                    'optimal': 'optimal',
                    'feasible': 'feasible',
                    'infeasible': 'infeasible',
                    'unbounded': 'unbounded',
                    'timelimit': 'time limit',
                    'nodelimit': 'node limit',
                    'totalnodelimit': 'node limit',
                    'stallnodelimit': 'stall node limit',
                    'memorylimit': 'memory limit',
                    'gaplimit': 'gap limit',
                    'primalfeasible': 'feasible',
                    'dualfeasible': 'feasible',
                    'unknown': 'unknown',
                    'solving': 'solving',
                }
                return status_map.get(status.lower(), status.lower())
            except Exception:
                return 'unknown'

        case 'objective':
            try:
                return scip_model.getObjVal()
            except Exception:
                return None

        case 'time':
            return (result[1][1] - result[1][0])

        case 'bound':
            try:
                return scip_model.getDualbound()
            except Exception:
                return None

        case 'iis':
            return []

        case 'slack':
            if input2 is None:
                return None
            try:
                cache = (input or {}).get('_scip_slack_cache', {})
                target = str(input2)
                if target in cache:
                    return cache[target]
            except Exception:
                pass
            return 0.0

        case 'dual':
            if input2 is None:
                return None
            try:
                cache = (input or {}).get('_scip_dual_cache', {})
                target = str(input2)
                if target in cache:
                    return cache[target]
            except Exception:
                pass
            return 0.0

        case 'rc':
            return None
