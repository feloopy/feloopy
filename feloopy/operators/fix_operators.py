# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def fix_dims(dim):

    if dim == 0:
        return dim

    if isinstance(dim, set):
        return dim

    if not isinstance(dim, (list, tuple)):
        return dim

    if len(dim) >= 1:
        if isinstance(dim[0], set):
            pass
        elif isinstance(dim[0], str):
            return set(dim)
        else:
            dim = [range(d) if not isinstance(d, range) else d for d in dim]
    
    return dim
