# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from hexaly.optimizer import HexalyOptimizer

def generate_model(features):

    optimizer = HexalyOptimizer()
    model = optimizer.model
    features['hexaly_optimizer'] = optimizer
    return model
