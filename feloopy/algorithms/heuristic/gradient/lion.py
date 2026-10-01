# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import warnings as wn
import numpy as np

wn.filterwarnings("ignore")

class LION:

    def __init__(self, f: int, d: list, s: int, t: int, lb: np.ndarray, ub: np.ndarray, lr: float, b1: float, b2: float, **kwargs):

        self.d = np.asarray([1 if item == 'max' else -1 for item in d])
        self.r = 0 if len(d) == 1 else len(d)
        self.f = f
        self.s = s
        self.t = t
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lb = np.array([lb])
        self.ub = np.array([ub])
        self.scale = (self.ub - self.lb)[0]
        self.features_cols = [0, self.f]
        self.grads_cols = [self.f, 2*self.f]
        self.status_col = [-2]
        self.reward_col = [-1]
        self.single_objective_tot = self.f + self.f + 1 + 1
        self.solve = self.run

    def run(self, evaluate):
        self.evaluate = evaluate
        self.initialize()
        for self.it_no in range(0, self.s):
            self.update()
            self.vary()
            if self.early_termination():
                break
        return self.report()

    def initialize(self):
        self.m = np.zeros(shape=[self.f])
        self.it = 0
        if self.r == 0:
            self.pi = np.random.rand(1, self.single_objective_tot)
            self.pi[:, :self.f] = self.lb + self.pi[:, :self.f]*(self.ub-self.lb)
            init = getattr(self, 'init_solutions', None)
            if init is not None:
                init = np.atleast_2d(init)
                self.pi[0, :self.f] = self.lb + init[0, :self.f]*(self.ub-self.lb)
            self.pi[:, self.reward_col] = - np.inf * self.d
            self.pi[:, self.status_col] = 0
            self.best_index = -1*(1+self.d[0])//2
            self.bad_status = -1
        self.best = self.pi[-1].copy()

    def _compute_gradient(self):
        h = 1e-7
        x = self.best[self.features_cols[0]:self.features_cols[1]].copy()
        grad = np.zeros(self.f)
        for i in range(self.f):
            x_plus = x.copy(); x_plus[i] += h
            x_minus = x.copy(); x_minus[i] -= h
            pp = np.zeros((1, self.single_objective_tot))
            pp[0, :self.f] = (x_plus - self.lb[0]) / (self.ub[0] - self.lb[0])
            f_plus = self.evaluate(pp)[0, self.reward_col[0]]
            pm = np.zeros((1, self.single_objective_tot))
            pm[0, :self.f] = (x_minus - self.lb[0]) / (self.ub[0] - self.lb[0])
            f_minus = self.evaluate(pm)[0, self.reward_col[0]]
            grad[i] = (f_plus - f_minus) / (2 * h)
        self.best[self.grads_cols[0]:self.grads_cols[1]] = grad

    def update(self):
        self.pi[:, :self.f] = (self.pi[:, :self.f]-self.lb)/(self.ub-self.lb)
        self.pi = self.evaluate(self.pi)
        self.pi[:, :self.f] = self.lb + self.pi[:, :self.f]*(self.ub-self.lb)
        self.best = self.pi[0].copy()
        self._compute_gradient()

    def vary(self):
        self.it += 1
        self.current_gradient = self.best[self.grads_cols[0]:self.grads_cols[1]]
        self.m = (1 - self.b2) * self.current_gradient + self.b2 * self.m
        update = np.sign(self.b1 * self.m + (1 - self.b1) * self.current_gradient)
        self.pi[:, :self.f] = self.pi[:, :self.f] - self.d[0] * self.lr * self.scale * update
        self.pi[:, :self.f] = np.clip(self.pi[:, :self.f], self.lb, self.ub)

    def early_termination(self):
        if np.linalg.norm(self.current_gradient) < 10e-6:
            return True
        else:
            return False

    def report(self):
        if self.r == 0:
            return (self.best[self.features_cols[0]:self.features_cols[1]]-self.lb[0])/(self.ub[0]-self.lb[0]), self.best[self.reward_col[0]], self.best[self.status_col]
