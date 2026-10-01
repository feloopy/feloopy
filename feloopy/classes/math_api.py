# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np


class MathAPI:
    """
    Mixin class containing all solver-dispatched math functions.
    
    This class is designed to be used as a mixin with the model class.
    It does not have an __init__ method and expects self.features and 
    self.model to be available from the parent class.
    """

    # =========================================================================
    # Uncertainty / Robust Optimization Functions
    # =========================================================================

    def scen_probs(self):
        """
        Returns an array of scenario probabilities
        """
        if self.features['interface_name'] == 'rsome_dro':
            return self.model.p
        elif self.features['interface_name'] == 'rsome_ro':
            return None

    def exval(self,expr):

        """
        Expected Value
        --------------
        1) Returns the expected value of random variables if the uncertainty set of expectations is being determined.
        2) Returns the worst case expected values of an expression that has random variables inside.

        """
        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import E
            try:
                return E(expr)
            except AttributeError:
                raise AttributeError(
                    "exval() is not available with the installed version of rsome. "
                    "The 'E' function requires expressions with an '.E' attribute. "
                    "For RO models, use minmax/maxmin obj_operators instead. "
                    "For DRO models, use minsup/maxinf obj_operators instead."
                )

    def norm(self, expr_or_1D_array_of_variables, degree=2):
        """
        Returns the first, second, or infinity norm of a 1-D array.

        Parameters
        ----------
        expr_or_1D_array_of_variables : array-like
            Input 1-D array expression.
        degree : int, optional
            Norm order: 1, 2, or inf (default 2).

        Returns
        -------
        expression or float
            The norm value.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import norm
            return norm(expr_or_1D_array_of_variables, degree)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            if isinstance(expr_or_1D_array_of_variables, (list, tuple)):
                expr_or_1D_array_of_variables = cvxpy.hstack(expr_or_1D_array_of_variables)
            return cvxpy.norm(expr_or_1D_array_of_variables, degree)

        else:
            arr = np.asarray(expr_or_1D_array_of_variables)
            return np.linalg.norm(arr, ord=degree)

    def sumsqr(self, expr_or_1D_array_of_variables):
        """
        Sum of squares of elements: sum(x_i^2).

        Parameters
        ----------
        expr_or_1D_array_of_variables : array-like
            Input expression or variable array.

        Returns
        -------
        expression or float
            The sum of squares.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import sumsqr
            return sumsqr(expr_or_1D_array_of_variables)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.sum_squares(expr_or_1D_array_of_variables)

        else:
            arr = np.asarray(expr_or_1D_array_of_variables)
            return np.sum(arr ** 2)

    def maxof(self, *args):
        """
        Element-wise maximum of multiple expressions.

        Parameters
        ----------
        *args : expressions
            Two or more expressions to compare.

        Returns
        -------
        expression or float
            Element-wise maximum.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import maxof
            return maxof(*args)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.max(cvxpy.vstack(args))

        else:
            return np.maximum.reduce(args)

    def minof(self, *args):
        """
        Element-wise minimum of multiple expressions.

        Parameters
        ----------
        *args : expressions
            Two or more expressions to compare.

        Returns
        -------
        expression or float
            Element-wise minimum.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import minof
            return minof(*args)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.min(cvxpy.vstack(args))

        else:
            return np.minimum.reduce(args)

    def E(self, expr):
        """
        Expected value of a random expression.

        Parameters
        ----------
        expr : expression
            An expression involving random variables.

        Returns
        -------
        expression
            The expected value.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import E
            return E(expr)

    def forall(self, expr, *sets):
        """
        Universal quantification over uncertainty sets.

        Parameters
        ----------
        expr : expression
            Constraint expression to hold for all scenarios.
        *sets : uncertainty sets
            The uncertainty sets to quantify over.

        Returns
        -------
        constraint
            A robust constraint valid for all scenarios.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            return expr.forall(*sets)

    def minsup(self, expr, fset):

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            self.model.minsup(expr, fset)
            if self.features['solution_method'] == 'exact':
                self.features['directions'].append('min')
                self.features['objectives'].append(expr)
                self.features['objective_labels'].append(None)
                self.features['objective_counter'][0] += 1
                self.features['objective_counter'][1] += 1
                self.features['_auto_obj_operators'] = 'sup'
                self.features['_objective_already_set'] = True

    def maxinf(self, expr, fset):

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            self.model.maxinf(expr, fset)
            if self.features['solution_method'] == 'exact':
                self.features['directions'].append('max')
                self.features['objectives'].append(expr)
                self.features['objective_labels'].append(None)
                self.features['objective_counter'][0] += 1
                self.features['objective_counter'][1] += 1
                self.features['_auto_obj_operators'] = 'inf'
                self.features['_objective_already_set'] = True

    def state_function(self):
        
        """
        Creates and returns a state function.
        """

        if hasattr(self.model, 'state_function'):
            return self.model.state_function()
        else:
            raise AttributeError(
                "state_function() is not available in the installed version of rsome. "
                "This feature requires a newer version of rsome."
            )

    def ldr(self, shape=(), name=None):
        """
        Create a Linear Decision Rule (LDR) variable for adaptive RO.

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the LDR variable array (default () for scalar).
        name : str, optional
            Name of the variable.

        Returns
        -------
        DecRule
            A linear decision rule variable that can be adapted to
            random variables via .adapt() before use in constraints.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            return self.model.ldr(shape, name)

    def minmax(self, obj, *uset_constraints):
        """
        Minimize the worst-case objective over an uncertainty set.

        Parameters
        ----------
        obj : expression
            Objective involving random variables.
        *uset_constraints : constraints
            Constraints defining the uncertainty set.

        Notes
        -----
        Call this instead of obj() for min-max robust objectives.
        Constraints added via con() are applied at solve time.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            self.model.minmax(obj, *uset_constraints)
            if self.features['solution_method'] == 'exact':
                self.features['directions'].append('min')
                self.features['objectives'].append(obj)
                self.features['objective_labels'].append(None)
                self.features['objective_counter'][0] += 1
                self.features['objective_counter'][1] += 1
                self.features['_auto_obj_operators'] = 'max'
                self.features['_objective_already_set'] = True

    def maxmin(self, obj, *uset_constraints):
        """
        Maximize the best-case objective over an uncertainty set.

        Parameters
        ----------
        obj : expression
            Objective involving random variables.
        *uset_constraints : constraints
            Constraints defining the uncertainty set.

        Notes
        -----
        Call this instead of obj() for max-min robust objectives.
        Constraints added via con() are applied at solve time.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            self.model.maxmin(obj, *uset_constraints)
            if self.features['solution_method'] == 'exact':
                self.features['directions'].append('max')
                self.features['objectives'].append(obj)
                self.features['objective_labels'].append(None)
                self.features['objective_counter'][0] += 1
                self.features['objective_counter'][1] += 1
                self.features['_auto_obj_operators'] = 'min'
                self.features['_objective_already_set'] = True

    def set(self, index='', bound=None, step=1, to_list=False):
        """
        Define a set. If bound is provided, the set will be a range from bound[0] to bound[1] with a step of step.
        If index is provided, the set will be created using the label index.

        Parameters
        ----------
        index : str, optional
            Label index to create the set.
        bound : list of int, optional
            Start and end values of the range. If provided, the set will be a range from bound[0] to bound[1].
        step : int, default 1
            Step size of the range.
        to_list : bool, default False
            If True, return the set as a list.

        Returns
        -------
        set or list
            The created set.

        Raises
        ------
        ValueError
            If neither bound nor index is provided.
        """
        
        if self.features['interface_name']=='gams':
            from gamspy import Set
            return Set(self.model, name=index, records=[f"{index}{i}" for i in range(bound[0],bound[1],step)])

        # Check if index is an empty string
        if index == '':
            if not to_list:
                return set(range(bound[0], bound[1], step))
            else:
                return list(range(bound[0], bound[1], step))

        # Check if bound is provided
        if bound is not None:
            if not to_list:
                return set(f'{index}{i}' for i in range(bound[0], bound[1], step))
            else:
                return list(f'{index}{i}' for i in range(bound[0], bound[1], step))

        # Check if index is provided
        elif index:
            if not to_list:
                return set(f'{index}{i}' for i in range(0, len(index), step))
            else:
                return list(f'{index}{i}' for i in range(0, len(index), step))

        # Raise an error if neither bound nor index is provided
        else:
            raise ValueError('Either bound or index must be provided.')

    def ambiguity_set(self,*args,**kwds):
        """
        Ambiguity set defintion
        """
        return self.model.ambiguity(*args,**kwds)

    ambiguity = ambiguity_set

    # =========================================================================
    # Basic Arithmetic / Aggregation
    # =========================================================================

    def sum(self, input, domain_tuple=None):
        """
        Calculate the sum of all values in the input.

        :param input: List of values to be summed.
        :return: The sum of the input values.
        """
        if self.features['interface_name'] == 'cplex':
            return self.model.sum(input)
        elif self.features['interface_name'] == 'gurobi':
            from gurobipy import quicksum
            return quicksum(input)
        elif self.features['interface_name'] == 'mip':
            from mip import xsum
            return xsum(input)
        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            if isinstance(input, (list, tuple)):
                input = cvxpy.hstack(input)
            return cvxpy.sum(input)
        elif self.features['interface_name'] == 'jump':
            from ..generators.variable.jump_expression import jump_sum, _is_jump_obj
            # materialise first: any() over a generator consumes the first
            # element and jump_sum would then never see it
            if not isinstance(input, (list, tuple)):
                input = list(input)
            if any(_is_jump_obj(x) for x in input):
                return jump_sum(input)
            return sum(input)
        else:
            return sum(input)


    def card(self, set):
        """
        Card Definition
        ~~~~~~~~~~~~~~~~
        To measure size of the set, etc.
        """

        return len(set)
    
    def abs(self, input):
        """
        Absolute value.

        Parameters
        ----------
        input : expression
            Input expression.

        Returns
        -------
        expression
            The absolute value expression.
        """

        if self.features['interface_name'] in ['cplex_cp', 'gekko']:

            return self.model.abs(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.abs(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_abs, _is_jump_obj
            if _is_jump_obj(input):
                return jump_abs(input)
            return abs(input)

        else:

            return abs(input)


    def acos(self, input):
        """

        Inverse cosine

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.acos(input)


        else:

            return np.arccos(input)
        

    def acosh(self, input):
        """

        Inverse hyperbolic cosine

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.acosh(input)

        else:

            return np.arccosh(input)
        

    def asin(self, input):
        """

        Inverse sine

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.asin(input)


        else:

            return np.asin(input)
        
    def asinh(self, input):
        """

        Inverse hyperbolic sine

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.asinh(input)

        else:

            return np.arcsinh(input)
        
    def atan(self, input):
        """

        Inverse tangent

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.atan(input)

        else:

            return np.arctan(input)
        
    def atanh(self, input):
        """

        Inverse hyperbolic tangent

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.atanh(input)

        else:

            return np.arctanh(input)
        
    def cos(self, input):
        """

        Cosine

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.cos(input)

        elif self.features['interface_name'] == 'scip':

            import pyscipopt as scip
            return scip.cos(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import cos as pyomo_cos
            return pyomo_cos(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.cos()
            return np.cos(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.cos(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_cos, _is_jump_obj
            if _is_jump_obj(input):
                return jump_cos(input)
            return np.cos(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.cos(input)

        else:

            return np.cos(input)

    def erf(self, input):
        """

        Error function

        """

        if self.features['interface_name'] == 'gekko':

            return self.model.erf(input)
        
    def erfc(self, input):
        """

        complementary error function

        """
        if self.features['interface_name'] == 'gekko':

            return self.model.erfc(input)

    def plus(self, input1, input2):
        """
        Addition of two expressions: input1 + input2.

        Parameters
        ----------
        input1, input2 : expressions
            Expressions to add.

        Returns
        -------
        expression
            The sum expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.plus(input1, input2)
        else:
            return input1 + input2

    def minus(self, input1, input2):
        """
        Subtraction of two expressions: input1 - input2.

        Parameters
        ----------
        input1, input2 : expressions
            Expressions to subtract.

        Returns
        -------
        expression
            The difference expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.minus(input1, input2)
        else:
            return input1 - input2

    def times(self, input1, input2):
        """
        Multiplication of two expressions: input1 * input2.

        Parameters
        ----------
        input1, input2 : expressions
            Expressions to multiply.

        Returns
        -------
        expression
            The product expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.times(input1, input2)
        else:
            return input1 * input2

    def true(self):
        """
        Boolean true constant.

        Returns
        -------
        bool or expression
            True value.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.true()
        else:
            return True

    def false(self):
        """
        Boolean false constant.

        Returns
        -------
        bool or expression
            False value.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.false()
        else:
            return False

    def trunc(self, input):
        '''
        Builds the truncated integer parts of a float expression
        '''

        if self.features['interface_name'] == 'cplex_cp':

            return self.model.trunc(input)

        else:

            return "None"

    def int_div(self, input1, input2):
        """
        Integer (floor) division: input1 // input2.

        Parameters
        ----------
        input1, input2 : expressions or scalars
            Dividend and divisor.

        Returns
        -------
        expression
            The integer division result.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.int_div(input1, input2)
        else:
            return input1 // input2

    def float_div(self, input1, input2):
        """
        Floating-point division: input1 / input2.

        Parameters
        ----------
        input1, input2 : expressions or scalars
            Dividend and divisor.

        Returns
        -------
        expression
            The division result.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.float_div(input1, input2)
        else:
            return input1 / input2

    def mod(self, input1, input2):
        """
        Modulo (remainder): input1 % input2.

        Parameters
        ----------
        input1, input2 : expressions or scalars
            Dividend and divisor.

        Returns
        -------
        expression
            The remainder.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.mod(input1, input2)
        else:
            return input1 % input2

    def square(self, input):
        """
        Element-wise square: input^2.

        Parameters
        ----------
        input : expression
            Input expression.

        Returns
        -------
        expression
            The squared expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.square(input)

        elif self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import square
            return square(input)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.square(input)

        else:
            return input * input

    def quad(self, expr_or_1D_array_of_variables, positive_or_negative_semidefinite_matrix):
        """
        Quadratic form: x^T Q x.

        Parameters
        ----------
        expr_or_1D_array_of_variables : array-like
            Vector variable or expression.
        positive_or_negative_semidefinite_matrix : array-like
            A positive or negative semidefinite matrix.

        Returns
        -------
        expression
            The quadratic form expression.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import quad
            return quad(expr_or_1D_array_of_variables, positive_or_negative_semidefinite_matrix)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.quad_form(expr_or_1D_array_of_variables, positive_or_negative_semidefinite_matrix)

    def expcone(self,rhs,a,b):
        """
        Returns an exponential cone constraint in the form: b*exp(a/b) <= rhs.

        Args
        rhs : array of variables or affine expression
        a : Scalar.
        b : Scalar.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import expcone
            return expcone(rhs,a,b)

    def power(self, input1, input2):
        """
        Element-wise power: abs(input1)^(input2/input3).

        Parameters
        ----------
        input1 : expression
            Base expression.
        input2 : int or array-like
            Numerator of the exponent.
        input3 : int or array-like, optional
            Denominator of the exponent (default 1).

        Returns
        -------
        expression
            The power expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.power(input1, input2)
        elif self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import power
            return power(input1, input2)
        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.power(input1, input2)
        elif self.features['interface_name'] == 'jump':
            from ..generators.variable.jump_expression import _is_jump_obj
            if _is_jump_obj(input1):
                return input1 ** input2
            return input1 ** input2
        else:
            return input1 ** input2

    def kldive(self, input, emprical_rpob, ambiguity_constant):
        """
        KL divergence constraint: sum(p * log(p/q)) <= r.

        Parameters
        ----------
        input : array-like
            Probability vector (variable or expression).
        emprical_rpob : array-like
            Empirical probability vector.
        ambiguity_constant : scalar
            Ambiguity budget constant.

        Returns
        -------
        constraint or float
            The KL divergence value or constraint.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import kldiv
            return kldiv(input, emprical_rpob, ambiguity_constant)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.kl_div(input, emprical_rpob) <= ambiguity_constant

        else:
            p = np.asarray(input, dtype=float)
            q = np.asarray(emprical_rpob, dtype=float)
            return np.sum(p * np.log(p / q)) - np.sum(p) + np.sum(q)

    kldiv = kldive

    def entropy(self, input):
        """
        Entropy expression: sum(input * log(input)).

        Parameters
        ----------
        input : array-like
            Input expression (typically probabilities).

        Returns
        -------
        expression or float
            The entropy value.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import entropy
            return entropy(input)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.sum(cvxpy.entr(input))

        else:
            arr = np.asarray(input, dtype=float)
            return np.sum(arr * np.log(arr))

    def pnorm(self, expr, degree=2, method=None):
        """
        Generalized p-norm with real degree > 1.

        Parameters
        ----------
        expr : array-like
            Input 1-D array expression.
        degree : float, optional
            Order of the norm. Must be > 1 (default 2).
        method : str, optional
            Reformulation method: 'soc' or 'exc'.

        Returns
        -------
        expression or float
            The p-norm value.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import pnorm
            return pnorm(expr, degree, method)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.pnorm(expr, degree)

        else:
            arr = np.asarray(expr, dtype=float)
            return np.linalg.norm(arr, ord=degree)

    def gmean(self, expr, beta=None):
        """
        Weighted geometric mean: prod(expr ** beta) ** (1/sum(beta)).

        Parameters
        ----------
        expr : array-like
            Input 1-D array of nonneg expressions.
        beta : array-like of ints, optional
            Weights for each element (default: all ones).

        Returns
        -------
        expression or float
            The geometric mean.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import gmean
            return gmean(expr, beta)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            if beta is None:
                return cvxpy.geo_mean(expr)
            return cvxpy.geo_mean(expr, beta)

        else:
            arr = np.asarray(expr, dtype=float)
            if beta is None:
                beta = np.ones_like(arr)
            beta = np.asarray(beta, dtype=float)
            return np.prod(arr ** (beta / np.sum(beta)))

    def rsocone(self, x, y, z):
        """
        Rotated second-order cone constraint: sumsqr(x) <= y*z.

        Parameters
        ----------
        x : array-like
            Vector expression.
        y, z : scalar expressions
            Scalar expressions.

        Returns
        -------
        bool
            Whether the constraint is satisfied (numeric check).
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import rsocone
            return rsocone(x, y, z)

        else:
            return np.sum(np.asarray(x) ** 2) <= float(y) * float(z)

    def fnorm(self, *args):
        """
        Frobenius norm of all given arrays.

        Parameters
        ----------
        *args : array-like
            Input arrays of any shape.

        Returns
        -------
        expression or float
            The Frobenius norm.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import fnorm
            return fnorm(*args)

        else:
            total = 0.0
            for arg in args:
                arr = np.asarray(arg, dtype=float)
                total += np.sum(arr ** 2)
            return np.sqrt(total)

    def pexp(self, expr, scale):
        """
        Perspective of exponential: scale * exp(expr/scale).

        Parameters
        ----------
        expr : expression
            Input expression.
        scale : scalar
            Scale parameter.

        Returns
        -------
        expression or float
            The perspective exponential.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import pexp
            return pexp(expr, scale)

        else:
            return float(scale) * np.exp(np.asarray(expr, dtype=float) / float(scale))

    def plog(self, expr, scale):
        """
        Perspective of logarithm: scale * log(expr/scale).

        Parameters
        ----------
        expr : expression
            Input expression.
        scale : scalar
            Scale parameter.

        Returns
        -------
        expression or float
            The perspective logarithm.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import plog
            return plog(expr, scale)

        else:
            return float(scale) * np.log(np.asarray(expr, dtype=float) / float(scale))

    def softplus(self, expr):
        """
        Softplus function: log(1 + exp(x)).

        Parameters
        ----------
        expr : expression
            Input scalar expression.

        Returns
        -------
        expression or float
            The softplus value.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import softplus
            return softplus(expr)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.logistic(expr)

        else:
            return np.log(1 + np.exp(np.asarray(expr, dtype=float)))

    def trace(self, expr):
        """
        Trace of a 2-D square matrix.

        Parameters
        ----------
        expr : array-like
            A 2-D square matrix expression.

        Returns
        -------
        expression or float
            The trace (sum of diagonal elements).
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import trace
            return trace(expr)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.trace(expr)

        else:
            return np.trace(np.asarray(expr))

    def diag(self, expr, k=0, fill=False):
        """
        Diagonal elements of a 2-D array.

        Parameters
        ----------
        expr : array-like
            A 2-D matrix expression.
        k : int, optional
            Diagonal offset: 0 for main, >0 above, <0 below (default 0).
        fill : bool, optional
            If True, return 2-D matrix with zeros off-diagonal (default False).

        Returns
        -------
        expression
            1-D diagonal elements (fill=False) or 2-D matrix (fill=True).
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import diag
            return diag(expr, k, fill)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.diag(expr, k)

        else:
            return np.diag(np.asarray(expr), k)

    def tril(self, expr, k=0):
        """
        Lower triangular elements of a 2-D array (zeros above diagonal).

        Parameters
        ----------
        expr : array-like
            A 2-D matrix expression.
        k : int, optional
            Diagonal above which to zero (default 0 = main diagonal).

        Returns
        -------
        expression
            The lower triangular part.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import tril
            return tril(expr, k)

        else:
            return np.tril(np.asarray(expr), k)

    def triu(self, expr, k=0):
        """
        Upper triangular elements of a 2-D array (zeros below diagonal).

        Parameters
        ----------
        expr : array-like
            A 2-D matrix expression.
        k : int, optional
            Diagonal below which to zero (default 0 = main diagonal).

        Returns
        -------
        expression
            The upper triangular part.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import triu
            return triu(expr, k)

        else:
            return np.triu(np.asarray(expr), k)

    def logdet(self, expr):
        """
        Log-determinant of a positive semidefinite matrix: log(det(A)).

        Parameters
        ----------
        expr : array-like
            A 2-D positive semidefinite matrix expression.

        Returns
        -------
        expression or float
            The log-determinant.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import logdet
            return logdet(expr)

        elif self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.log_det(expr)

        else:
            sign, logdet = np.linalg.slogdet(np.asarray(expr))
            return logdet

    def rootdet(self, expr):
        """
        Root-determinant: det(A)^(1/L) for PSD matrix A of dimension L.

        Parameters
        ----------
        expr : array-like
            A 2-D positive semidefinite matrix expression.

        Returns
        -------
        expression or float
            The root-determinant.
        """

        if self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:
            from rsome import rootdet
            return rootdet(expr)

        else:
            A = np.asarray(expr)
            det_val = np.linalg.det(A)
            L = A.shape[0]
            return np.sign(det_val) * np.abs(det_val) ** (1.0 / L)

    def log(self, input):
        """

        Natural Logarithm

        """

        if self.features['interface_name'] in ['cplex_cp']:

            return self.model.log(input)

        elif self.features['interface_name'] in ['gekko']:

            return self.model.log(input)
        
        elif self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:

            from rsome import log
            return log(input)

        elif self.features['interface_name'] == 'scip':

            import pyscipopt as scip
            return scip.log(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import log as pyomo_log
            return pyomo_log(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.log()
            return np.log(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.log(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_log, _is_jump_obj
            if _is_jump_obj(input):
                return jump_log(input)
            return np.log(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.log(input)

        else:

            return np.log(input)

    def log10(self, input):
        """

        Logarithm Base 10

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.log10(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import log10 as pyomo_log10
            return pyomo_log10(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return UnoExpr('log10', input)
            return np.log10(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.log(input) / np.log(10)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.log10(input)

        else:

            return np.log10(input)
        
    def sin(self, input):
        """

        Sine

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.sin(input)

        elif self.features['interface_name'] == 'scip':

            import pyscipopt as scip
            return scip.sin(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import sin as pyomo_sin
            return pyomo_sin(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.sin()
            return np.sin(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.sin(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_sin, _is_jump_obj
            if _is_jump_obj(input):
                return jump_sin(input)
            return np.sin(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.sin(input)

        else:

            return np.sin(input)

    def whether(self, condition, do_if_true, do_if_false):
        """
        Element-wise conditional: where(condition, x, y).

        Parameters
        ----------
        condition : array-like
            Boolean condition array.
        do_if_true : expression or scalar
            Value where condition is True.
        do_if_false : expression or scalar
            Value where condition is False.

        Returns
        -------
        expression
            The conditional expression.
        """
        return np.where(condition, do_if_true, do_if_false)

    def sinh(self, input):
        """

        Hyperbolic sine

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.sinh(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import sinh as pyomo_sinh
            return pyomo_sinh(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.sinh()
            return np.sinh(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.sinh(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.sinh(input)

        else:

            return np.sinh(input)
      
    def sqrt(self, input):
        """

        Square root

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.sqrt(input)

        elif self.features['interface_name'] == 'scip':

            import pyscipopt as scip
            return scip.sqrt(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import sqrt as pyomo_sqrt
            return pyomo_sqrt(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.sqrt()
            return np.sqrt(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.sqrt(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_sqrt, _is_jump_obj
            if _is_jump_obj(input):
                return jump_sqrt(input)
            return np.sqrt(input)

        else:

            return np.sqrt(input)
        
    def tan(self, input):
        """

        Tangent

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.tan(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import tan as pyomo_tan
            return pyomo_tan(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.tan()
            return np.tan(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.tan(input)

        elif self.features['interface_name'] == 'jump':

            from ..generators.variable.jump_expression import jump_tan, _is_jump_obj
            if _is_jump_obj(input):
                return jump_tan(input)
            return np.tan(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.tan(input)

        else:

            return np.tan(input)
        
    def tanh(self, input):
        """

        Hyperbolic tangent

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.tanh(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import tanh as pyomo_tanh
            return pyomo_tanh(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.tanh()
            return np.tanh(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.tanh(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.tanh(input)

        else:

            return np.tanh(input)

    def cosh(self, input):
        """

        Hyperbolic cosine

        """

        if self.features['interface_name'] in ['gekko']:

            return self.model.cosh(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import cosh as pyomo_cosh
            return pyomo_cosh(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.cosh()
            return np.cosh(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.cosh(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.cosh(input)

        else:

            return np.cosh(input)

    def sigmoid(self, input):
        """

        Sigmoid function

        """

        if self.features['interface_name'] in ['gekko']:
            return self.model.sigmoid(input)
        else:
            return 1 / (1 + np.exp(-input))
        
    def exponent(self, input):
        """
        Natural exponential function: exp(x).

        Parameters
        ----------
        input : expression
            Input scalar expression.

        Returns
        -------
        expression
            The exponential expression.
        """

        if self.features['interface_name'] in ['cplex_cp', 'gekko']:

            return self.model.exp(input)

        elif self.features['interface_name'] in ['rsome_ro', 'rsome_dro']:

            from rsome import exp
            return exp(input)

        elif self.features['interface_name'] == 'cvxpy':

            import cvxpy
            return cvxpy.exp(input)

        elif self.features['interface_name'] == 'scip':

            import pyscipopt as scip
            return scip.exp(input)

        elif self.features['interface_name'] in ['bonmin', 'couenne']:

            from pyomo.environ import exp as pyomo_exp
            return pyomo_exp(input)

        elif self.features['interface_name'] in ['uno', 'unopy']:

            from ..generators.variable.uno_variable_generator import UnoVar, UnoExpr
            if isinstance(input, (UnoVar, UnoExpr)):
                return input.exp()
            return np.exp(input)

        elif self.features['interface_name'] == 'casadi':

            import casadi as cas
            return cas.exp(input)

        else:

            return np.exp(input)

    def exp(self, input):
        """
        Natural exponential function: exp(x).

        Alias of :meth:`exponent`.

        Parameters
        ----------
        input : expression
            Input scalar expression.

        Returns
        -------
        expression
            The exponential expression.
        """
        return self.exponent(input)

    def count(self, input, value):
        """
        Count occurrences of value in input.

        Parameters
        ----------
        input : list or array
            Input collection.
        value : scalar
            Value to count.

        Returns
        -------
        int
            Number of occurrences.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.count(input, value)
        else:
            return input.count(value)

    def scal_prod(self, input1, input2):
        """
        Scalar (dot) product of two vectors: input1 . input2.

        Parameters
        ----------
        input1, input2 : array-like
            Input vectors.

        Returns
        -------
        expression or scalar
            The dot product.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.scal_prod(input1, input2)
        else:
            return np.dot(input1, input2)

    def range(self, x, lb=None, ub=None):
        """
        Bound a variable between lower and upper limits.

        Parameters
        ----------
        x : variable
            Variable to bound.
        lb : scalar, optional
            Lower bound.
        ub : scalar, optional
            Upper bound.

        Returns
        -------
        list
            List of bound constraints.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.range(x, lb, ub)
        else:
            return [x >= lb] + [x <= ub]

    def floor(self, x):
        """
        Floor function: largest integer <= x.

        Parameters
        ----------
        x : expression or scalar
            Input expression.

        Returns
        -------
        expression
            The floor expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.floor(x)
        else:
            return np.floor(x)

    def ceil(self, x):
        """
        Ceiling function: smallest integer >= x.

        Parameters
        ----------
        x : expression or scalar
            Input expression.

        Returns
        -------
        expression
            The ceiling expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.ceil(x)
        else:
            return np.ceil(x)

    def round(self, x):
        """
        Round to nearest integer.

        Parameters
        ----------
        x : expression or scalar
            Input expression.

        Returns
        -------
        expression
            The rounded expression.
        """

        if self.features['interface_name'] == 'cplex_cp':
            return self.model.round(x)
        else:
            return np.round(x)

    def prod(self, expr, axis=None):
        """
        Product of elements along an axis.

        Parameters
        ----------
        expr : array-like
            Input expression.
        axis : int, optional
            Axis along which to compute (None for all elements).

        Returns
        -------
        expression or float
            The product.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.prod(expr, axis=axis)

        else:
            return np.prod(np.asarray(expr), axis=axis)

    def minimum(self, *args):
        """
        Element-wise minimum of multiple expressions.

        Parameters
        ----------
        *args : expressions
            Two or more expressions.

        Returns
        -------
        expression or float
            Element-wise minimum.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.minimum(*args)

        else:
            return np.minimum.reduce(args)

    def maximum(self, *args):
        """
        Element-wise maximum of multiple expressions.

        Parameters
        ----------
        *args : expressions
            Two or more expressions.

        Returns
        -------
        expression or float
            Element-wise maximum.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.maximum(*args)

        else:
            return np.maximum.reduce(args)

    def max(self, *args):
        """
        Maximum value across elements or arguments.

        Parameters
        ----------
        *args : expressions or list/tuple of expressions
            One or more expressions, or a single list/tuple.

        Returns
        -------
        expression
            The maximum expression.
        """
        interface = self.features.get('interface_name', '')

        if interface == 'cplex_cp':
            return self.model.max(*args)
        elif interface == 'cvxpy':
            import cvxpy
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return cvxpy.max(cvxpy.hstack(args[0]))
            return cvxpy.max(*args)
        elif interface == 'jump':
            from ..generators.variable.jump_expression import jump_max
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                items = list(args[0])
                result = items[0]
                for item in items[1:]:
                    result = jump_max(result, item)
                return result
            return jump_max(args[0], args[1]) if len(args) == 2 else np.max(*args)
        else:
            if len(args) == 2 and np.ndim(args[0]) == 0 and np.ndim(args[1]) == 0:
                return np.maximum(args[0], args[1])
            return np.max(*args)

    def min(self, *args):
        """
        Minimum value across elements or arguments.

        Parameters
        ----------
        *args : expressions or list/tuple of expressions
            One or more expressions, or a single list/tuple.

        Returns
        -------
        expression
            The minimum expression.
        """
        interface = self.features.get('interface_name', '')

        if interface == 'cplex_cp':
            return self.model.min(*args)
        elif interface == 'cvxpy':
            import cvxpy
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return cvxpy.min(cvxpy.hstack(args[0]))
            return cvxpy.min(*args)
        elif interface == 'jump':
            from ..generators.variable.jump_expression import jump_min
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                items = list(args[0])
                result = items[0]
                for item in items[1:]:
                    result = jump_min(result, item)
                return result
            return jump_min(args[0], args[1]) if len(args) == 2 else np.min(*args)
        else:
            if len(args) == 2 and np.ndim(args[0]) == 0 and np.ndim(args[1]) == 0:
                return np.minimum(args[0], args[1])
            return np.min(*args)

    def mean(self, expr, axis=None):
        """
        Mean of elements along an axis.

        Parameters
        ----------
        expr : array-like
            Input expression.
        axis : int, optional
            Axis along which to compute.

        Returns
        -------
        expression or float
            The mean.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.mean(expr, axis=axis)

        else:
            return np.mean(np.asarray(expr), axis=axis)

    def sum_squares(self, expr):
        """
        Sum of squares of all elements.

        Parameters
        ----------
        expr : array-like
            Input expression.

        Returns
        -------
        expression or float
            The sum of squares.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.sum_squares(expr)

        else:
            arr = np.asarray(expr)
            return np.sum(arr ** 2)

    def quad_form(self, x, P):
        """
        Quadratic form: x^T P x.

        Parameters
        ----------
        x : array-like
            Vector variable or expression.
        P : array-like
            Positive semidefinite matrix.

        Returns
        -------
        expression or float
            The quadratic form.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.quad_form(x, P)

        else:
            x_arr = np.asarray(x)
            P_arr = np.asarray(P)
            return float(x_arr @ P_arr @ x_arr)

    def quad_over_lin(self, x, y):
        """
        Quadratic over linear: (x^T x) / y.

        Parameters
        ----------
        x : array-like
            Vector expression.
        y : scalar expression
            Denominator (must be positive).

        Returns
        -------
        expression or float
            The quad-over-linear value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.quad_over_lin(x, y)

        else:
            return float(np.sum(np.asarray(x) ** 2)) / float(y)

    def matrix_frac(self, x, P):
        """
        Matrix fraction: x^T P^{-1} x.

        Parameters
        ----------
        x : array-like
            Vector expression.
        P : array-like
            Positive definite matrix.

        Returns
        -------
        expression or float
            The matrix fraction value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.matrix_frac(x, P)

        else:
            x_arr = np.asarray(x)
            P_arr = np.asarray(P)
            return float(x_arr @ np.linalg.inv(P_arr) @ x_arr)

    def mixed_norm(self, X, p, q):
        """
        Mixed norm: ||X||_{p,q} = (sum_i (||x_i||_p)^q)^(1/q).

        Parameters
        ----------
        X : array-like
            Input matrix expression.
        p : int
            Inner norm order.
        q : int
            Outer norm order.

        Returns
        -------
        expression or float
            The mixed norm.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.mixed_norm(X, p, q)

        else:
            arr = np.asarray(X)
            inner_norms = np.linalg.norm(arr, ord=p, axis=1)
            return float(np.linalg.norm(inner_norms, ord=q))

    def log_sum_exp(self, expr):
        """
        Log-sum-exp: log(sum(exp(x_i))).

        Parameters
        ----------
        expr : array-like
            Input vector expression.

        Returns
        -------
        expression or float
            The log-sum-exp value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.log_sum_exp(expr)

        else:
            arr = np.asarray(expr, dtype=float)
            return float(np.log(np.sum(np.exp(arr))))

    def sum_largest(self, expr, k):
        """
        Sum of k largest elements.

        Parameters
        ----------
        expr : array-like
            Input vector expression.
        k : int
            Number of largest elements to sum.

        Returns
        -------
        expression or float
            The sum of k largest elements.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.sum_largest(expr, k)

        else:
            arr = np.asarray(expr).flatten()
            return float(np.sum(np.sort(arr)[-k:]))

    def sum_smallest(self, expr, k):
        """
        Sum of k smallest elements.

        Parameters
        ----------
        expr : array-like
            Input vector expression.
        k : int
            Number of smallest elements to sum.

        Returns
        -------
        expression or float
            The sum of k smallest elements.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.sum_smallest(expr, k)

        else:
            arr = np.asarray(expr).flatten()
            return float(np.sum(np.sort(arr)[:k]))

    def harmonic_mean(self, expr):
        """
        Harmonic mean: n / sum(1/x_i).

        Parameters
        ----------
        expr : array-like
            Input vector expression (must be positive).

        Returns
        -------
        expression or float
            The harmonic mean.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.harmonic_mean(expr)

        else:
            arr = np.asarray(expr, dtype=float)
            return float(len(arr) / np.sum(1.0 / arr))

    def log_det(self, X):
        """
        Log-determinant of a positive definite matrix.

        Parameters
        ----------
        X : array-like
            Positive definite matrix expression.

        Returns
        -------
        expression or float
            The log-determinant.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.log_det(X)

        else:
            sign, logdet = np.linalg.slogdet(np.asarray(X))
            return logdet

    def tr_inv(self, X):
        """
        Trace of inverse: tr(X^{-1}).

        Parameters
        ----------
        X : array-like
            Positive definite matrix expression.

        Returns
        -------
        expression or float
            The trace of inverse.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.tr_inv(X)

        else:
            return float(np.trace(np.linalg.inv(np.asarray(X))))

    def lambda_max(self, X):
        """
        Maximum eigenvalue of a symmetric matrix.

        Parameters
        ----------
        X : array-like
            Symmetric matrix expression.

        Returns
        -------
        expression or float
            The maximum eigenvalue.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.lambda_max(X)

        else:
            return float(np.max(np.linalg.eigvalsh(np.asarray(X))))

    def lambda_min(self, X):
        """
        Minimum eigenvalue of a symmetric matrix.

        Parameters
        ----------
        X : array-like
            Symmetric matrix expression.

        Returns
        -------
        expression or float
            The minimum eigenvalue.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.lambda_min(X)

        else:
            return float(np.min(np.linalg.eigvalsh(np.asarray(X))))

    def von_neumann_entr(self, X):
        """
        Von Neumann entropy of a symmetric matrix.

        Parameters
        ----------
        X : array-like
            Symmetric matrix expression.

        Returns
        -------
        expression or float
            The von Neumann entropy.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.von_neumann_entr(X)

        else:
            eigvals = np.linalg.eigvalsh(np.asarray(X))
            eigvals = eigvals[eigvals > 0]
            return float(-np.sum(eigvals * np.log(eigvals)))

    def std(self, expr, axis=None, ddof=0):
        """
        Standard deviation along an axis.

        Parameters
        ----------
        expr : array-like
            Input expression.
        axis : int, optional
            Axis along which to compute.
        ddof : int, optional
            Delta degrees of freedom (default 0).

        Returns
        -------
        expression or float
            The standard deviation.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.std(expr, axis=axis, ddof=ddof)

        else:
            return float(np.std(np.asarray(expr), axis=axis, ddof=ddof))

    def var(self, expr, axis=None, ddof=0):
        """
        Variance along an axis.

        Parameters
        ----------
        expr : array-like
            Input expression.
        axis : int, optional
            Axis along which to compute.
        ddof : int, optional
            Delta degrees of freedom (default 0).

        Returns
        -------
        expression or float
            The variance.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.var(expr, axis=axis, ddof=ddof)

        else:
            return float(np.var(np.asarray(expr), axis=axis, ddof=ddof))

    def rel_entr(self, x, y):
        """
        Relative entropy: x * log(x/y).

        Parameters
        ----------
        x, y : expressions
            Nonneg scalar expressions.

        Returns
        -------
        expression or float
            The relative entropy.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.rel_entr(x, y)

        else:
            x_f, y_f = float(x), float(y)
            return x_f * np.log(x_f / y_f)

    def kl_div(self, x, y):
        """
        KL divergence: x * log(x/y) - x + y.

        Parameters
        ----------
        x, y : expressions
            Nonneg scalar expressions.

        Returns
        -------
        expression or float
            The KL divergence.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.kl_div(x, y)

        else:
            x_f, y_f = float(x), float(y)
            return x_f * np.log(x_f / y_f) - x_f + y_f

    def logistic(self, x):
        """
        Logistic function: log(1 + exp(x)).

        Parameters
        ----------
        x : expression
            Input scalar expression.

        Returns
        -------
        expression or float
            The logistic value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.logistic(x)

        else:
            return float(np.log(1 + np.exp(float(x))))

    def entr(self, x):
        """
        Negative entropy: -x * log(x).

        Parameters
        ----------
        x : expression
            Nonneg scalar expression.

        Returns
        -------
        expression or float
            The entropy value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.entr(x)

        else:
            x_f = float(x)
            return -x_f * np.log(x_f) if x_f > 0 else 0.0

    def pos(self, x):
        """
        Positive part: max(x, 0).

        Parameters
        ----------
        x : expression
            Input expression.

        Returns
        -------
        expression or float
            The positive part.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.pos(x)

        else:
            return float(np.maximum(float(x), 0))

    def neg(self, x):
        """
        Negative part: max(-x, 0).

        Parameters
        ----------
        x : expression
            Input expression.

        Returns
        -------
        expression or float
            The negative part.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.neg(x)

        else:
            return float(np.maximum(-float(x), 0))

    def inv_pos(self, x):
        """
        Inverse of positive part: 1/x for x > 0.

        Parameters
        ----------
        x : expression
            Nonneg expression.

        Returns
        -------
        expression or float
            The inverse position.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.inv_pos(x)

        else:
            return 1.0 / float(x)

    def xexp(self, x):
        """
        Weighted exponential: x * exp(x).

        Parameters
        ----------
        x : expression
            Input expression.

        Returns
        -------
        expression or float
            The xexp value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.xexp(x)

        else:
            return float(x) * np.exp(float(x))

    def huber(self, x, M=1):
        """
        Huber function: min(x^2, 2*M*|x| - M^2).

        Parameters
        ----------
        x : expression
            Input expression.
        M : float, optional
            Transition point (default 1).

        Returns
        -------
        expression or float
            The Huber value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.huber(x, M)

        else:
            x_abs = abs(float(x))
            return x_abs ** 2 if x_abs <= M else 2 * M * x_abs - M ** 2

    def tv(self, x):
        """
        Total variation of a vector.

        Parameters
        ----------
        x : array-like
            Input vector expression.

        Returns
        -------
        expression or float
            The total variation.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.tv(x)

        else:
            arr = np.asarray(x).flatten()
            return float(np.sum(np.abs(np.diff(arr))))

    def scalene(self, x, alpha, beta):
        """
        Scalene function: max(alpha*x, beta*x).

        Parameters
        ----------
        x : expression
            Input expression.
        alpha, beta : scalars
            Multipliers.

        Returns
        -------
        expression or float
            The scalene value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.scalene(x, alpha, beta)

        else:
            return max(float(alpha) * float(x), float(beta) * float(x))

    def log_normcdf(self, x):
        """
        Log of normal CDF.

        Parameters
        ----------
        x : expression
            Input scalar expression.

        Returns
        -------
        expression or float
            The log-normal-CDF value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.log_normcdf(x)

        else:
            from scipy.stats import norm
            return float(norm.logcdf(float(x)))

    def dotsort(self, x, w):
        """
        Dot product of x with sorted w.

        Parameters
        ----------
        x : array-like
            Input vector expression.
        w : array-like
            Weight vector.

        Returns
        -------
        expression or float
            The dotsort value.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.dotsort(x, w)

        else:
            return float(np.dot(np.asarray(x).flatten(), np.sort(np.asarray(w).flatten())))

    def vstack(self, *args):
        """
        Vertical stacking of arrays.

        Parameters
        ----------
        *args : array-like
            Arrays to stack vertically.

        Returns
        -------
        expression or ndarray
            The vertically stacked array.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.vstack(args)

        else:
            return np.vstack(args)

    def hstack(self, *args):
        """
        Horizontal stacking of arrays.

        Parameters
        ----------
        *args : array-like
            Arrays to stack horizontally.

        Returns
        -------
        expression or ndarray
            The horizontally stacked array.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.hstack(args)

        else:
            return np.hstack(args)

    def vec(self, X):
        """
        Vectorize a matrix (column-major flattening).

        Parameters
        ----------
        X : array-like
            Input matrix expression.

        Returns
        -------
        expression or ndarray
            The vectorized array.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.vec(X)

        else:
            return np.asarray(X).flatten(order='F')

    def reshape(self, X, shape):
        """
        Reshape an expression.

        Parameters
        ----------
        X : array-like
            Input expression.
        shape : tuple
            Target shape.

        Returns
        -------
        expression or ndarray
            The reshaped array.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.reshape(X, shape)

        else:
            return np.asarray(X).reshape(shape)

    def convolve(self, x, y):
        """
        Convolution of two 1-D vectors.

        Parameters
        ----------
        x, y : array-like
            Input vectors.

        Returns
        -------
        expression or ndarray
            The convolution result.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.convolve(x, y)

        else:
            return np.convolve(np.asarray(x).flatten(), np.asarray(y).flatten())

    def kron(self, X, Y):
        """
        Kronecker product of two matrices.

        Parameters
        ----------
        X, Y : array-like
            Input matrices.

        Returns
        -------
        expression or ndarray
            The Kronecker product.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.kron(X, Y)

        else:
            return np.kron(np.asarray(X), np.asarray(Y))

    def outer(self, x, y):
        """
        Outer product of two vectors.

        Parameters
        ----------
        x, y : array-like
            Input vectors.

        Returns
        -------
        expression or ndarray
            The outer product matrix.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.outer(x, y)

    def upper_tri(self, X):
        """
        Upper triangular elements of a matrix (as a vector).

        Parameters
        ----------
        X : array-like
            Input square matrix.

        Returns
        -------
        expression
            The upper triangular elements.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.upper_tri(X)

        else:
            arr = np.asarray(X)
            n = arr.shape[0]
            return arr[np.triu_indices(n, k=1)]

    def cumsum(self, expr, axis=None):
        """
        Cumulative sum along an axis.

        Parameters
        ----------
        expr : array-like
            Input expression.
        axis : int, optional
            Axis along which to compute.

        Returns
        -------
        expression or ndarray
            The cumulative sum.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.cumsum(expr, axis=axis)

        else:
            return np.cumsum(np.asarray(expr), axis=axis)

    def diff(self, x, k=1, axis=-1):
        """
        k-th order discrete difference along an axis.

        Parameters
        ----------
        x : array-like
            Input expression.
        k : int, optional
            Order of difference (default 1).
        axis : int, optional
            Axis along which to compute (default -1).

        Returns
        -------
        expression or ndarray
            The differenced array.
        """

        if self.features['interface_name'] == 'cvxpy':
            import cvxpy
            return cvxpy.diff(x, k=k, axis=axis)

        else:
            return np.diff(np.asarray(x), n=k, axis=axis)
