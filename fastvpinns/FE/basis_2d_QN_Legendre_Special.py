"""
file: basis_2d_QN_Legendre_Special.py
description: This file contains the class Basis2DQNLegendreSpecial which is used 
             to define the basis functions for a 2D Quad element using a Legendre polynomial.
authors: Thivin Anandh D
changelog: 30/Aug/2023 - Initial version
known_issues: None
"""

import numpy as np

# import the legendre polynomials
from scipy.special import legendre
import matplotlib.pyplot as plt

from .basis_function_2d import BasisFunction2D


class Basis2DQNLegendreSpecial(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q1 element.

    Attributes:
        num_shape_functions (int): The number of shape functions.

    Methods:
        __init__(self, num_shape_functions: int): Initializes the Basis2DQNLegendreSpecial object.
        test_fcn(self, n_test, x): Calculates the test function values for a given number of tests and input values.
        test_grad_fcn(self, n_test, x): Calculates the gradients of the test functions for a given number of tests and input values.
        test_grad_grad_fcn(self, n_test, x): Calculates the second derivatives of the test functions for a given number of tests and input values.
        value(self, xi, eta): Returns the values of the basis functions at the given (xi, eta) coordinates.
        gradx(self, xi, eta): Returns the x-derivatives of the basis functions at the given (xi, eta) coordinates.
        grady(self, xi, eta): Returns the y-derivatives of the basis functions at the given (xi, eta) coordinates.
        gradxx(self, xi, eta): Returns the xx-derivatives of the basis functions at the given (xi, eta) coordinates.
        gradxy(self, xi, eta): Returns the xy-derivatives of the basis functions at the given (xi, eta) coordinates.
        gradyy(self, xi, eta): Returns the yy-derivatives of the basis functions at the given (xi, eta) coordinates.
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def test_fcn(self, n_test, x):
        """
        Calculate the test function values for a given number of tests and input values.

        :param n_test: The number of tests to perform.
        :type n_test: int
        :param x: The input value(s) for the test function.
        :type x: float or array-like
        :return: An array containing the test function values for each test.
        :rtype: numpy.ndarray
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1)
            obj2 = legendre(n - 1)
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def test_grad_fcn(self, n_test, x):
        """
        Calculate the gradient of the test function at a given point.

        :param n_test: The number of test functions to calculate.
        :type n_test: int
        :param x: The point at which to evaluate the gradient.
        :type x: float
        :return: An array containing the gradients of the test functions at the given point.
        :rtype: np.ndarray
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1).deriv()
            obj2 = legendre(n - 1).deriv()
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def test_grad_grad_fcn(self, n_test, x):
        """
        Calculate the gradient of the second derivative of a function using Legendre polynomials.

        :param n_test: The number of test cases to evaluate.
        :type n_test: int
        :param x: The input value at which to evaluate the function.
        :type x: float
        :return: An array containing the results of the test cases.
        :rtype: ndarray
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1).deriv(2)
            obj2 = legendre(n - 1).deriv(2)
            test = obj1(x) - obj2(x)

            test_total.append(test)
        return np.asarray(test_total)

    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinate.
        :type xi: numpy.ndarray

        :param eta: The eta coordinate.
        :type eta: numpy.ndarray

        :return: The values of the basis functions at the given coordinates.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_y
            )

        return values

    def gradx(self, xi, eta):
        """
        This method returns the x-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinate.
        :type xi: numpy.ndarray

        :param eta: The eta coordinate.
        :type eta: numpy.ndarray

        :return: The x-derivatives of the basis functions.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_x = self.test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_x[i, :] * test_function_y
            )

        return values

    def grady(self, xi, eta):
        """
        This method returns the y-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinates.
        :type xi: numpy.ndarray

        :param eta: The eta coordinates.
        :type eta: numpy.ndarray

        :return: The y-derivatives of the basis functions.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.test_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_grad_y
            )

        return values

    def gradxx(self, xi, eta):
        """
        This method returns the xx-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinate.
        :type xi: numpy.ndarray

        :param eta: The eta coordinate.
        :type eta: numpy.ndarray

        :return: The xx-derivatives of the basis functions.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_grad_x = self.test_grad_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_grad_x[i, :] * test_function_y
            )

        return values

    def gradxy(self, xi, eta):
        """
        This method returns the xy-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinate.
        :type xi: numpy.ndarray

        :param eta: The eta coordinate.
        :type eta: numpy.ndarray

        :return: The xy-derivatives of the basis functions.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_x = self.test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.test_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_x[i, :] * test_function_grad_y
            )

        return values

    def gradyy(self, xi, eta):
        """
        This method returns the yy-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: The xi coordinates.
        :type xi: numpy.ndarray

        :param eta: The eta coordinates.
        :type eta: numpy.ndarray

        :return: The yy-derivatives of the basis functions.
        :rtype: numpy.ndarray
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_grad_y = self.test_grad_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_grad_grad_y
            )

        return values
