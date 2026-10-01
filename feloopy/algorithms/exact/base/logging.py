# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Shared logging utilities for decomposition algorithms."""


def make_logger(prefix, show_log):
    """Return a logging function that prints with *prefix* when enabled.

    Parameters
    ----------
    prefix : str
        Tag shown in brackets, e.g. ``'Benders'``.
    show_log : bool
        When *False* the returned function is a no-op.

    Returns
    -------
    log : callable
        ``log(msg)`` prints ``[{prefix}] {msg}`` to stdout.
        Detailed per-iteration diagnostics (messages starting with two
        spaces) are suppressed for a clean tabular progress table;
        only the header / iteration rows are shown.
    """
    def _log(msg):
        if not show_log:
            return
        # keep only the iteration table and high-level status.
        if isinstance(msg, str) and msg.startswith("  "):
            stripped = msg.lstrip()
            if stripped.startswith(("Solving subproblem",
                                   "Subproblem",
                                   "IIS nogood",
                                   "Re-solving",
                                   "Artificial-var",
                                   "Gurobi",
                                   "Optimality cut",
                                   "Initial x_bar")):
                return
        print(f"{msg}", flush=True)
    return _log
