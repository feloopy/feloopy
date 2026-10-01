"""
Expression Functions Module

This module provides mathematical atomic functions accessible through the
model object's `m.` syntax. Uses cvxpy when available for the cvxpy interface,
numpy for evaluation, and standard Python operators for other optimization interfaces.

Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

import numpy as np


class CvxpExpressionClass:
    """Mixin class that exposes mathematical functions via m. syntax."""

    def _is_cvxpy(self):
        return self.features.get('interface_name', '') == 'cvxpy'

    def _import_cvxpy(self):
        import cvxpy
        return cvxpy

    def _is_numpy(self, x):
        return isinstance(x, (np.ndarray, np.generic))

    def _is_opt_var(self, x):
        return not isinstance(x, (int, float, np.ndarray, np.generic, bool))

    # ── Elementwise functions ───────────────────────────────────────────

    def exp(self, x):
        """exp(x) — exponential. Uses cvxpy atom when available, numpy otherwise."""
        if self._is_cvxpy():
            return self._import_cvxpy().exp(x)
        if self._is_numpy(x):
            return np.exp(x)
        if self._is_opt_var(x):
            import sympy
            return sympy.exp(x)
        return np.exp(float(x))

    def log1p(self, x):
        """log1p(x) — log(x + 1). Uses cvxpy atom when available, numpy otherwise."""
        if self._is_cvxpy():
            return self._import_cvxpy().log1p(x)
        if self._is_numpy(x):
            return np.log1p(x)
        if self._is_opt_var(x):
            import sympy
            return sympy.log(x + 1)
        return np.log1p(float(x))

    def log2(self, x):
        """log2(x) — log base 2."""
        if self._is_cvxpy():
            cvxpy = self._import_cvxpy()
            return cvxpy.log(x) / np.log(2)
        if self._is_numpy(x):
            return np.log2(x)
        if self._is_opt_var(x):
            import sympy
            return sympy.log(x) / sympy.log(2)
        return np.log2(float(x))

    def square(self, x):
        """square(x) — x^2."""
        if self._is_cvxpy():
            return self._import_cvxpy().square(x)
        if self._is_numpy(x):
            return np.square(x)
        return x ** 2

    def power(self, x, p):
        """power(x, p) — x^p."""
        if self._is_cvxpy():
            return self._import_cvxpy().power(x, p)
        if self._is_numpy(x):
            return np.power(x, p)
        return x ** p

    def pos(self, x):
        """pos(x) — max(x, 0)."""
        if self._is_cvxpy():
            return self._import_cvxpy().pos(x)
        if self._is_numpy(x):
            return np.maximum(x, 0)
        if self._is_opt_var(x):
            z = self.pvar(f"_pos{self._next_ind()}")
            self.con(z >= x)
            self.con(z >= 0)
            return z
        return max(float(x), 0)

    def neg(self, x):
        """neg(x) — max(-x, 0)."""
        if self._is_cvxpy():
            return self._import_cvxpy().neg(x)
        if self._is_numpy(x):
            return np.maximum(-x, 0)
        if self._is_opt_var(x):
            z = self.pvar(f"_neg{self._next_ind()}")
            self.con(z >= -x)
            self.con(z >= 0)
            return z
        return max(-float(x), 0)

    def inv_pos(self, x):
        """inv_pos(x) — 1/x (x > 0)."""
        if self._is_cvxpy():
            return self._import_cvxpy().inv_pos(x)
        if self._is_numpy(x):
            return 1.0 / x
        return 1 / x

    def entr(self, x):
        """entr(x) — -x*log(x), concave."""
        if self._is_cvxpy():
            return self._import_cvxpy().entr(x)
        if self._is_numpy(x):
            result = np.zeros_like(x, dtype=float)
            mask = x > 0
            result[mask] = -x[mask] * np.log(x[mask])
            return result
        import sympy
        return -x * sympy.log(x)

    def logistic(self, x):
        """logistic(x) — log(1 + exp(x))."""
        if self._is_cvxpy():
            return self._import_cvxpy().logistic(x)
        if self._is_numpy(x):
            return np.log(1 + np.exp(x))
        import sympy
        return sympy.log(1 + sympy.exp(x))

    def huber(self, x, M=1):
        """huber(x, M) — Huber loss."""
        if self._is_cvxpy():
            return self._import_cvxpy().huber(x, M=M)
        if self._is_numpy(x):
            abs_x = np.abs(x)
            return np.where(abs_x <= M, abs_x ** 2, 2 * M * abs_x - M ** 2)
        if self._is_opt_var(x):
            t = self.pvar(f"_huber{self._next_ind()}")
            self.con(t >= x)
            self.con(t >= -x)
            s = self.pvar(f"_huber_s{self._next_ind()}")
            self.con(s >= 0)
            self.con(s >= t - M)
            return M ** 2 + 2 * M * s - s ** 2
        abs_x = abs(float(x))
        return abs_x ** 2 if abs_x <= M else 2 * M * abs_x - M ** 2

    def xexp(self, x):
        """xexp(x) — x * exp(x)."""
        if self._is_cvxpy():
            return self._import_cvxpy().xexp(x)
        if self._is_numpy(x):
            return x * np.exp(x)
        import sympy
        return x * sympy.exp(x)

    def scalene(self, x, alpha, beta):
        """scalene(x, alpha, beta) — alpha*pos(x) + beta*neg(x)."""
        if self._is_cvxpy():
            return self._import_cvxpy().scalene(x, alpha, beta)
        if self._is_numpy(x):
            return alpha * np.maximum(x, 0) + beta * np.maximum(-x, 0)
        return alpha * self.pos(x) + beta * self.neg(x)

    def multiply(self, c, x):
        """multiply(c, x) — elementwise multiplication."""
        if self._is_cvxpy():
            return self._import_cvxpy().multiply(c, x)
        return c * x

    def minimum(self, *args):
        """minimum(x, y, ...) — elementwise minimum."""
        if self._is_cvxpy():
            return self._import_cvxpy().minimum(*args)
        if len(args) == 0:
            return 0
        result = args[0]
        for a in args[1:]:
            if self._is_opt_var(result) or self._is_opt_var(a):
                z = self.pvar(f"_min{self._next_ind()}")
                self.con(z <= result)
                self.con(z <= a)
                result = z
            elif self._is_numpy(result) or self._is_numpy(a):
                result = np.minimum(result, a)
            else:
                result = min(float(result), float(a))
        return result

    def maximum(self, *args):
        """maximum(x, y, ...) — elementwise maximum."""
        if self._is_cvxpy():
            return self._import_cvxpy().maximum(*args)
        if len(args) == 0:
            return 0
        result = args[0]
        for a in args[1:]:
            if self._is_opt_var(result) or self._is_opt_var(a):
                z = self.pvar(f"_max{self._next_ind()}")
                self.con(z >= result)
                self.con(z >= a)
                result = z
            elif self._is_numpy(result) or self._is_numpy(a):
                result = np.maximum(result, a)
            else:
                result = max(float(result), float(a))
        return result

    # ── Aggregation ─────────────────────────────────────────────────────

    def mean(self, x, axis=None, keepdims=False):
        """mean(x) — arithmetic mean."""
        if self._is_cvxpy():
            return self._import_cvxpy().mean(x, axis=axis, keepdims=keepdims)
        if self._is_numpy(x):
            return np.mean(x, axis=axis, keepdims=keepdims)
        return sum(x) / len(x)

    def sum_squares(self, x):
        """sum_squares(x) — sum of squared elements."""
        if self._is_cvxpy():
            return self._import_cvxpy().sum_squares(x)
        if self._is_numpy(x):
            return np.sum(x ** 2)
        return sum(xi ** 2 for xi in x)

    def sum_largest(self, x, k):
        """sum_largest(x, k) — sum of k largest elements."""
        if self._is_cvxpy():
            return self._import_cvxpy().sum_largest(x, k)
        if self._is_numpy(x):
            sorted_x = np.sort(x)[::-1]
            return np.sum(sorted_x[:k])
        sorted_x = sorted(x, reverse=True)
        return sum(sorted_x[:k])

    def sum_smallest(self, x, k):
        """sum_smallest(x, k) — sum of k smallest elements."""
        if self._is_cvxpy():
            return self._import_cvxpy().sum_smallest(x, k)
        if self._is_numpy(x):
            sorted_x = np.sort(x)
            return np.sum(sorted_x[:k])
        sorted_x = sorted(x)
        return sum(sorted_x[:k])

    def log_sum_exp(self, x, axis=None, keepdims=False):
        """log_sum_exp(x) — log of sum of exponentials."""
        if self._is_cvxpy():
            return self._import_cvxpy().log_sum_exp(x, axis=axis, keepdims=keepdims)
        if self._is_numpy(x):
            max_x = np.max(x, axis=axis, keepdims=True)
            return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=keepdims))
        import sympy
        return sympy.log(sum(sympy.exp(xi) for xi in x))

    # ── Norm / quadratic form ───────────────────────────────────────────

    def pnorm(self, x, p):
        """pnorm(x, p) — p-norm."""
        if self._is_cvxpy():
            return self._import_cvxpy().pnorm(x, p)
        if self._is_numpy(x):
            return np.linalg.norm(x, ord=p)
        if p == 1:
            return sum(abs(xi) for xi in x)
        elif p == 2:
            return self.con_norm_of(
                self.pvar(f"_pnorm{self._next_ind()}"), x, norm_type=2
            )
        return sum(abs(xi) ** p for xi in x) ** (1.0 / p)

    def quad_form(self, x, P):
        """quad_form(x, P) — x^T P x."""
        if self._is_cvxpy():
            return self._import_cvxpy().quad_form(x, P)
        if self._is_numpy(x) and self._is_numpy(P):
            return x @ P @ x
        return sum(x[i] * P[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))

    def quad_over_lin(self, x, y):
        """quad_over_lin(x, y) — (sum x_i^2) / y."""
        if self._is_cvxpy():
            return self._import_cvxpy().quad_over_lin(x, y)
        if self._is_numpy(x):
            return np.sum(x ** 2) / y
        return sum(xi ** 2 for xi in x) / y

    def matrix_frac(self, x, P):
        """matrix_frac(x, P) — x^T P^{-1} x."""
        if self._is_cvxpy():
            return self._import_cvxpy().matrix_frac(x, P)
        if self._is_numpy(x) and self._is_numpy(P):
            return x @ np.linalg.inv(P) @ x
        import sympy
        P_inv = sympy.Matrix(P).inv()
        x_vec = sympy.Matrix(x)
        return (x_vec.T * P_inv * x_vec)[0, 0]

    def mixed_norm(self, X, p, q):
        """mixed_norm(X, p, q) — || X ||_{p,q}."""
        if self._is_cvxpy():
            return self._import_cvxpy().mixed_norm(X, p, q)
        if self._is_numpy(X):
            return np.linalg.norm([np.linalg.norm(row, ord=p) for row in X], ord=q)
        return 0

    # ── Statistical / special ───────────────────────────────────────────

    def geo_mean(self, x, p=None):
        """geo_mean(x, p) — geometric mean."""
        if self._is_cvxpy():
            cvxpy = self._import_cvxpy()
            return cvxpy.geo_mean(x) if p is None else cvxpy.geo_mean(x, p)
        if self._is_numpy(x):
            return np.prod(x) ** (1.0 / len(x))
        import sympy
        return sympy.prod(x) ** (sympy.Rational(1, len(x)))

    def harmonic_mean(self, x):
        """harmonic_mean(x) — harmonic mean."""
        if self._is_cvxpy():
            return self._import_cvxpy().harmonic_mean(x)
        if self._is_numpy(x):
            return len(x) / np.sum(1.0 / x)
        return len(x) / sum(1.0 / xi for xi in x)

    def tv(self, x):
        """tv(x) — total variation."""
        if self._is_cvxpy():
            return self._import_cvxpy().tv(x)
        if self._is_numpy(x):
            return np.sum(np.abs(np.diff(x)))
        return sum(abs(x[i + 1] - x[i]) for i in range(len(x) - 1))

    def kl_div(self, x, y):
        """kl_div(x, y) — KL divergence."""
        if self._is_cvxpy():
            return self._import_cvxpy().kl_div(x, y)
        if self._is_numpy(x) and self._is_numpy(y):
            mask = x > 0
            return np.sum(x[mask] * np.log(x[mask] / y[mask]))
        import sympy
        return sum(xi * sympy.log(xi / yi) for xi, yi in zip(x, y) if xi > 0)

    def rel_entr(self, x, y):
        """rel_entr(x, y) — relative entropy x*log(x/y)."""
        if self._is_cvxpy():
            return self._import_cvxpy().rel_entr(x, y)
        if self._is_numpy(x) and self._is_numpy(y):
            return np.sum(x * np.log(x / y))
        import sympy
        return sum(xi * sympy.log(xi / yi) for xi, yi in zip(x, y))

    def log_normcdf(self, x):
        """log_normcdf(x) — approximate log normal CDF."""
        if self._is_cvxpy():
            return self._import_cvxpy().log_normcdf(x)
        from scipy.stats import norm
        if self._is_numpy(x):
            return np.log(norm.cdf(x))
        import sympy
        return sympy.log(sympy.erfc(-x / sympy.sqrt(2)) / 2)

    def dotsort(self, X, W):
        """dotsort(X, W) — dot product of sorted vectors."""
        if self._is_cvxpy():
            return self._import_cvxpy().dotsort(X, W)
        if self._is_numpy(X) and self._is_numpy(W):
            return np.sum(np.sort(X) * np.sort(W))
        sorted_X = sorted(X)
        sorted_W = sorted(W)
        return sum(xi * wi for xi, wi in zip(sorted_X, sorted_W))

    def ptp(self, X, axis=None, keepdims=False):
        """ptp(X) — peak-to-peak (max - min)."""
        if self._is_cvxpy():
            return self._import_cvxpy().ptp(X, axis=axis, keepdims=keepdims)
        if self._is_numpy(X):
            return np.ptp(X, axis=axis, keepdims=keepdims)
        return max(X) - min(X)

    def std(self, X, axis=None, ddof=0, keepdims=False):
        """std(X) — standard deviation."""
        if self._is_cvxpy():
            return self._import_cvxpy().std(X, axis=axis, ddof=ddof, keepdims=keepdims)
        if self._is_numpy(X):
            return np.std(X, axis=axis, ddof=ddof, keepdims=keepdims)
        import statistics
        return statistics.stdev(X)

    def var(self, X, axis=None, ddof=0, keepdims=False):
        """var(X) — variance."""
        if self._is_cvxpy():
            return self._import_cvxpy().var(X, axis=axis, ddof=ddof, keepdims=keepdims)
        if self._is_numpy(X):
            return np.var(X, axis=axis, ddof=ddof, keepdims=keepdims)
        import statistics
        return statistics.variance(X)

    # ── Matrix functions ────────────────────────────────────────────────

    def log_det(self, X):
        """log_det(X) — log of determinant (X positive definite)."""
        if self._is_cvxpy():
            return self._import_cvxpy().log_det(X)
        if self._is_numpy(X):
            sign, logdet = np.linalg.slogdet(X)
            return logdet
        import sympy
        return sympy.log(sympy.Matrix(X).det())

    def tr_inv(self, X):
        """tr_inv(X) — trace of inverse."""
        if self._is_cvxpy():
            return self._import_cvxpy().tr_inv(X)
        if self._is_numpy(X):
            return np.trace(np.linalg.inv(X))
        import sympy
        return sympy.Matrix(X).inv().trace()

    def lambda_max(self, X):
        """lambda_max(X) — maximum eigenvalue."""
        if self._is_cvxpy():
            return self._import_cvxpy().lambda_max(X)
        if self._is_numpy(X):
            return np.max(np.linalg.eigvalsh(X))
        import sympy
        return max(sympy.Matrix(X).eigenvals().keys())

    def lambda_min(self, X):
        """lambda_min(X) — minimum eigenvalue."""
        if self._is_cvxpy():
            return self._import_cvxpy().lambda_min(X)
        if self._is_numpy(X):
            return np.min(np.linalg.eigvalsh(X))
        import sympy
        return min(sympy.Matrix(X).eigenvals().keys())

    def von_neumann_entr(self, X):
        """von_neumann_entr(X) — von Neumann entropy."""
        if self._is_cvxpy():
            return self._import_cvxpy().von_neumann_entr(X)
        if self._is_numpy(X):
            eigvals = np.linalg.eigvalsh(X)
            eigvals = eigvals[eigvals > 0]
            return -np.sum(eigvals * np.log(eigvals))
        return 0

    # ── Vector / matrix construction ────────────────────────────────────

    def vstack(self, X_list):
        """vstack([X1, ...]) — vertical stack."""
        if self._is_cvxpy():
            return self._import_cvxpy().vstack(X_list)
        if all(self._is_numpy(x) for x in X_list):
            return np.vstack(X_list)
        return [list(x) for x in X_list]

    def hstack(self, X_list):
        """hstack([X1, ...]) — horizontal stack."""
        if self._is_cvxpy():
            return self._import_cvxpy().hstack(X_list)
        if all(self._is_numpy(x) for x in X_list):
            return np.hstack(X_list)
        result = []
        for x in X_list:
            result.extend(list(x))
        return result

    def vec(self, X):
        """vec(X) — flatten matrix to vector."""
        if self._is_cvxpy():
            return self._import_cvxpy().vec(X)
        if self._is_numpy(X):
            return X.flatten()
        return [X]

    def reshape(self, X, shape, order='F'):
        """reshape(X, shape, order) — reshape expression."""
        if self._is_cvxpy():
            return self._import_cvxpy().reshape(X, shape, order=order)
        if self._is_numpy(X):
            return X.reshape(shape, order=order)
        return X

    def convolve(self, c, x):
        """convolve(c, x) — convolution."""
        if self._is_cvxpy():
            return self._import_cvxpy().convolve(c, x)
        if self._is_numpy(c) and self._is_numpy(x):
            return np.convolve(c, x)
        n = len(c) + len(x) - 1
        return [sum(c[i] * x[j - i] for i in range(len(c)) if 0 <= j - i < len(x)) for j in range(n)]

    def kron(self, X, Y):
        """kron(X, Y) — Kronecker product."""
        if self._is_cvxpy():
            return self._import_cvxpy().kron(X, Y)
        if self._is_numpy(X) and self._is_numpy(Y):
            return np.kron(X, Y)
        return X * Y

    def outer(self, x, y):
        """outer(x, y) — outer product x * y^T."""
        if self._is_cvxpy():
            return self._import_cvxpy().outer(x, y)
        if self._is_numpy(x) and self._is_numpy(y):
            return np.outer(x, y)
        return [[xi * yi for yi in y] for xi in x]

    def upper_tri(self, X):
        """upper_tri(X) — upper triangle as vector."""
        if self._is_cvxpy():
            return self._import_cvxpy().upper_tri(X)
        if self._is_numpy(X):
            return X[np.triu_indices(X.shape[0], k=1)]
        n = len(X)
        return [X[i][j] for i in range(n) for j in range(i + 1, n)]
