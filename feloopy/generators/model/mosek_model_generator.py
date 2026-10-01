# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mosek

INF = 1e30

def generate_model(features):

    env = mosek.Env()
    task = env.Task(0, 0)
    features['mosek_env'] = env
    return task
