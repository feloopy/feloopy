# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from docplex.mp.model import Model as CPLEXMODEL


def generate_model(features):

    model_name = features.get('model_name', 'feloopy_model')

    env = features.get('env', None)
    if env is not None:
        model_object = CPLEXMODEL(name=model_name, cplex_parameters=env)
    else:
        model_object = CPLEXMODEL(name=model_name)

    model_params = features.get('model_params', {})
    for key, value in model_params.items():
        try:
            model_object.parameters.__setattr__(key, value)
        except Exception:
            pass

    return model_object
