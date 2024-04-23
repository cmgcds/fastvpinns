# Purpose: Defines the basis functions for a 2D Quad element using a Lengendre polynomial.
# Author: Thivin Anandh D
# Date: 30/Aug/2023

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

        Parameters:
            n_test (int): The number of tests to perform.
            x (float or array-like): The input value(s) for the test function.

        Returns:
            numpy.ndarray: An array containing the test function values for each test.

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

        Parameters:
        - n_test (int): The number of test functions to calculate.
        - x (float): The point at which to evaluate the gradient.

        Returns:
        - np.ndarray: An array containing the gradients of the test functions at the given point.
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

        Parameters:
        - n_test (int): The number of test cases to evaluate.
        - x (float): The input value at which to evaluate the function.

        Returns:
        - test_total (ndarray): An array containing the results of the test cases.
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
