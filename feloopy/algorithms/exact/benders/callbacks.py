# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from .enums import BendersStatus

class _BendersEvent:
    ON_START = 'on_start'
    ON_ITERATION_START = 'on_iteration_start'
    ON_MASTER_BUILD = 'on_master_build'
    ON_BEFORE_MASTER_SOLVE = 'on_before_master_solve'
    ON_AFTER_MASTER_SOLVED = 'on_after_master_solved'
    ON_MASTER_SOLVED = 'on_master_solved'
    ON_BEFORE_SUB_SOLVE = 'on_before_sub_solve'
    ON_SUB_SOLVE = 'on_sub_solve'
    ON_AFTER_SUB_SOLVED = 'on_after_sub_solved'
    ON_SUB_SOLVED = 'on_sub_solved'
    ON_OPTI_CUT_GENERATED = 'on_opti_cut_generated'
    ON_FEAS_CUT_GENERATED = 'on_feas_cut_generated'
    ON_CUT_GENERATED = 'on_cut_generated'
    ON_OPTI_CUT_ADDED = 'on_opti_cut_added'
    ON_FEAS_CUT_ADDED = 'on_feas_cut_added'
    ON_CUT_ADDED = 'on_cut_added'
    ON_NEW_LOWER_BOUND = 'on_new_lower_bound'
    ON_NEW_UPPER_BOUND = 'on_new_upper_bound'
    ON_ITERATION_END = 'on_iteration_end'
    ON_END = 'on_end'


class BendersCallback:
    """Base class for Benders decomposition callbacks.

    Override any of the event methods to hook into the Benders process.
    Return ``BendersCallback.TERMINATE`` to stop early.

    Context attributes:
        iteration   : int        -- current iteration number
        lb          : float      -- current lower bound
        ub          : float      -- current upper bound
        master_obj  : float      -- master problem objective value
        sub_results : list       -- subproblem results
        current_cuts: list       -- cuts added so far
        x_bar       : dict       -- current complicating variable values
        status      : BendersStatus -- current solver status
        where       : str        -- 'incumbent' or 'node' (for BnC)
    """

    TERMINATE = True
    PROCEED = False

    def on_start(self, context): return self.PROCEED
    def on_iteration_start(self, context): return self.PROCEED
    def on_master_build(self, context): return self.PROCEED
    def on_before_master_solve(self, context): return self.PROCEED
    def on_after_master_solved(self, context): return self.PROCEED
    def on_master_solved(self, context): return self.PROCEED
    def on_before_sub_solve(self, context): return self.PROCEED
    def on_sub_solve(self, context): return self.PROCEED
    def on_after_sub_solved(self, context): return self.PROCEED
    def on_sub_solved(self, context): return self.PROCEED
    def on_opti_cut_generated(self, context): return self.PROCEED
    def on_feas_cut_generated(self, context): return self.PROCEED
    def on_cut_generated(self, context): return self.PROCEED
    def on_opti_cut_added(self, context): return self.PROCEED
    def on_feas_cut_added(self, context): return self.PROCEED
    def on_cut_added(self, context): return self.PROCEED
    def on_new_lower_bound(self, context): return self.PROCEED
    def on_new_upper_bound(self, context): return self.PROCEED
    def on_iteration_end(self, context): return self.PROCEED
    def on_end(self, context): return self.PROCEED


class BendersContext:
    """Read-only context passed to callbacks during the Benders process.

    Attributes
    ----------
    iteration : int
        Current iteration number.
    lb : float
        Current lower bound.
    ub : float
        Current upper bound.
    master_obj : float or None
        Master problem objective value.
    sub_results : list
        Subproblem results from the current iteration.
    current_cuts : list
        Cuts added so far.
    x_bar : dict
        Current complicating variable values.
    status : BendersStatus
        Current solver status.
    where : str
        ``'incumbent'`` or ``'node'`` (for Branch-and-Check).
    master_problem : object or None
        Reference to the master problem object.
    sub_problem : object or None
        Reference to the subproblem object.
    benders : object or None
        Reference to the BendersDecomposition instance.
    """

    def __init__(self):
        self.iteration = 0
        self.lb = float('-inf')
        self.ub = float('inf')
        self.master_obj = None
        self.sub_results = []
        self.current_cuts = []
        self.x_bar = {}
        self.status = BendersStatus.UNSOLVED
        self.where = 'incumbent'
        self.master_problem = None
        self.sub_problem = None
        self.benders = None


class _CallbackManager:
    def __init__(self):
        self._callbacks = []

    def register(self, callback):
        if isinstance(callback, type) and issubclass(callback, BendersCallback):
            callback = callback()
        if callable(callback) and not isinstance(callback, BendersCallback):
            class _FnCallback(BendersCallback):
                pass
            cb = _FnCallback()
            for event_name in dir(cb):
                if event_name.startswith('on_') and event_name != 'on_start':
                    setattr(cb, event_name, callback)
            callback = cb
        if isinstance(callback, BendersCallback):
            self._callbacks.append(callback)

    def trigger(self, event, context):
        for cb in self._callbacks:
            handler = getattr(cb, event, None)
            if handler and handler != getattr(BendersCallback, event, None):
                result = handler(context)
                if result is BendersCallback.TERMINATE or result is True:
                    return True
        return False
