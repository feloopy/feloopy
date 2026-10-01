# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import importlib


def generate_model(features):

    interface_name = features.get('interface_name', 'pyoptinterface.highs')
    if interface_name == 'pyoptinterface':
        interface_name = 'pyoptinterface.highs'

    module = importlib.import_module(interface_name)
    Model = getattr(module, 'Model')

    model_object = Model()

    model_params = features.get('model_params', {})
    for key, value in model_params.items():
        try:
            attr = getattr(__import__('pyoptinterface', fromlist=['ModelAttribute']).ModelAttribute, key, None)
            if attr is not None:
                model_object.set_model_attribute(attr, value)
        except Exception:
            pass

    return model_object
