# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def _get_model(features):
    return features.get('model_object_before_solve') or features.get('model_object')


def set_init_value(features, variable, value, fix):
    model_object = _get_model(features)
    if model_object is None:
        return
    if fix:
        model_object.subject_to(variable == value)
    else:
        model_object.set_initial(variable, value)
