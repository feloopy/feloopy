# Copyright (c) 2022-2024, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it

def sets(*args):
    if len(args)==1:
        return args[0]
    else:
        return it.product(*args)