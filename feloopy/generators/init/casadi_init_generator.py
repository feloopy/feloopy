# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def set_init_value(features, variable, value, fix):
    model_object = features['model_object_before_solve']
    if fix:
        model_object.subject_to(variable == value)
    else:
        model_object.set_initial(variable, value)