# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
import numpy as np

def sets(*args):
    if len(args)==1:
        return args[0]
    else:
        return it.product(*args)
    
def array(*args):
    return np.array(*args)