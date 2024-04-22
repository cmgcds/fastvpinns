# Purpose: Defines the basis functions for a 2D Quad element using a Lengendre polynomial.
# Author: Thivin Anandh D
# Date: 30/Aug/2023

import numpy as np
from .basis_function_2d import BasisFunction2D

# import the legendre polynomials
from scipy.special import eval_legendre, legendre

import matplotlib.pyplot as plt


class Basis2DQN(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q1 element.
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def Test_fcn(self, N_test, x):
        test_total = []
        for n in range(1, N_test + 1):
            obj1 = legendre(n + 1)
            obj2 = legendre(n - 1)
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def Test_grad_fcn(self, N_test, x):
        test_total = []
        for n in range(1, N_test + 1):
            obj1 = legendre(n + 1).deriv()
            obj2 = legendre(n - 1).deriv()
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def Test_grad_grad_fcn(self, N_test, x):
        test_total = []
        for n in range(1, N_test + 1):
            obj1 = legendre(n + 1).deriv().deriv()
            obj2 = legendre(n - 1).deriv().deriv()
            test = obj1(x) - obj2(x)

            test_total.append(test)
        return np.asarray(test_total)

    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.Test_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.Test_fcn(num_shape_func_in_1d, eta)

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

        test_function_grad_x = self.Test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.Test_fcn(num_shape_func_in_1d, eta)

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

        test_function_x = self.Test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.Test_grad_fcn(num_shape_func_in_1d, eta)

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

        test_function_grad_grad_x = self.Test_grad_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.Test_fcn(num_shape_func_in_1d, eta)

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

        test_function_grad_x = self.Test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.Test_grad_fcn(num_shape_func_in_1d, eta)

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

        test_function_x = self.Test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_grad_y = self.Test_grad_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_grad_grad_y
            )

        return values
