# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

class MultiObjectivityError(Exception):
    pass

class VariableDimError(Exception):
    pass

class DirectionError(Exception):
    pass

class ConstantConstraintError(ValueError):
    """A constant constraint evaluated to False: the model is infeasible.

    Subclasses ``ValueError`` so plain ``except ValueError`` still matches.
    """
    pass
