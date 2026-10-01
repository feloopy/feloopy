# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import xpress as xpress_interface


def generate_model(features):

    model_name = features.get('model_name', 'feloopy_model')

    env = features.get('env', None)
    if env is not None:
        model_object = xpress_interface.problem(name=model_name)
    else:
        model_object = xpress_interface.problem(name=model_name)

    model_params = features.get('model_params', {})
    for key, value in model_params.items():
        try:
            model_object.controls.__setattr__(key, value)
        except Exception:
            pass

    return model_object
