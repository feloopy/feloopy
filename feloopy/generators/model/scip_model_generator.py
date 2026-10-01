# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def generate_model(features):

    try:
        import pyscipopt as scip
    except ImportError:
        raise ImportError(
            "The 'pyscipopt' package is required for SCIP interface. "
            "Install it with: pip install PySCIPOpt"
        )

    model = scip.Model()
    return model
