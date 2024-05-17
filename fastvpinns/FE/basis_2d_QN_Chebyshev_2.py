"""
file: basis_2d_QN_Chebyshev_2.py
description: This file contains the class Basis2DQNChebyshev2 which defines the basis functions for a 
              2D Q1 element using Chebyshev polynomials.
              Test functions and derivatives are inferred from the work by Ehsan Kharazmi et.al
             (hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition)
             available at https://github.com/ehsankharazmi/hp-VPINNs/
authors: Thivin Anandh D
changelog: 30/Aug/2023 - Initial version
known_issues: None
"""

# import the legendre polynomials
from scipy.special import jacobi

import numpy as np
from .basis_function_2d import BasisFunction2D


class Basis2DQNChebyshev2(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q1 element.
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def jacobi_wrapper(self, n, a, b, x):
        """
        Evaluate the Jacobi polynomial of degree n with parameters a and b at the given points x.

        :param n: Degree of the Jacobi polynomial.
        :type n: int
        :param a: First parameter of the Jacobi polynomial.
        :type a: float
        :param b: Second parameter of the Jacobi polynomial.
        :type b: float
        :param x: Points at which to evaluate the Jacobi polynomial.
        :type x: array_like

        :return: Values of the Jacobi polynomial at the given points x.
        :rtype: array_like
        """
        x = np.array(x, dtype=np.float64)
        return jacobi(n, a, b)(x)

    ## Helper Function
    def test_fcnx(self, n_test, x):
        """
        Compute the x-component of the test functions for a given number of test functions and x-coordinates.

        :param n_test: Number of test functions.
        :type n_test: int
        :param x: x-coordinates at which to evaluate the test functions.
        :type x: array_like

        :return: Values of the x-component of the test functions.
        :rtype: array_like
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, x) / self.jacobi_wrapper(
                n + 1, -1 / 2, -1 / 2, 1
            ) - self.jacobi_wrapper(n - 1, -1 / 2, -1 / 2, x) / self.jacobi_wrapper(
                n - 1, -1 / 2, -1 / 2, 1
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def test_fcny(self, n_test, y):
        """
        Compute the y-component of the test functions for a given number of test functions and y-coordinates.

        :param n_test: Number of test functions.
        :type n_test: int
        :param y: y-coordinates at which to evaluate the test functions.
        :type y: array_like
        :return: Values of the y-component of the test functions.
        :rtype: array_like
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, y) / self.jacobi_wrapper(
                n + 1, -1 / 2, -1 / 2, 1
            ) - self.jacobi_wrapper(n - 1, -1 / 2, -1 / 2, y) / self.jacobi_wrapper(
                n - 1, -1 / 2, -1 / 2, 1
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def dtest_fcn(self, n_test, x):
        """
        Compute the first and second derivatives of the test function.

        :param n_test: The number of test functions.
        :type n_test: int
        :param x: The input value.
        :type x: float
        :return: Array of first derivatives of the test function, Array of second derivatives of the test function.
        :rtype: tuple(ndarray, ndarray)
        """
        d1test_total = []
        d2test_total = []
        for n in range(1, n_test + 1):
            if n == 1:
                d1test = (
                    ((n + 1) / 2)
                    * self.jacobi_wrapper(n, 1 / 2, 1 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d2test = (
                    ((n + 2) * (n + 1) / (2 * 2))
                    * self.jacobi_wrapper(n - 1, 3 / 2, 3 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n == 2:
                d1test = ((n + 1) / 2) * self.jacobi_wrapper(
                    n, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n - 1) / 2
                ) * self.jacobi_wrapper(
                    n - 2, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d2test = (
                    ((n + 2) * (n + 1) / (2 * 2))
                    * self.jacobi_wrapper(n - 1, 3 / 2, 3 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            else:
                d1test = ((n + 1) / 2) * self.jacobi_wrapper(
                    n, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n - 1) / 2
                ) * self.jacobi_wrapper(
                    n - 2, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d2test = ((n + 2) * (n + 1) / (2 * 2)) * self.jacobi_wrapper(
                    n - 1, 3 / 2, 3 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n) * (n - 1) / (2 * 2)
                ) * self.jacobi_wrapper(
                    n - 3, 3 / 2, 3 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
        return np.asarray(d1test_total), np.asarray(d2test_total)

    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.
        :type eta: array_like
        :return: Values of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * test_y
            )

        return values

    def gradx(self, xi, eta):
        """
        This method returns the x-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.
        :type eta: array_like
        :return: Values of the x-derivatives of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_test_x = self.dtest_fcn(num_shape_func_in_1d, xi)[0]
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_test_x[i, :] * test_y
            )

        return values

    def grady(self, xi, eta):
        """
        This method returns the y-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.

        :return: Values of the y-derivatives of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        grad_test_y = self.dtest_fcn(num_shape_func_in_1d, eta)[0]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * grad_test_y
            )

        return values

    def gradxx(self, xi, eta):
        """
        This method returns the xx-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.
        :type eta: array_like

        :return: Values of the xx-derivatives of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_grad_x = self.dtest_fcn(num_shape_func_in_1d, xi)[1]
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_grad_x[i, :] * test_y
            )

        return values

    def gradxy(self, xi, eta):
        """
        This method returns the xy-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.
        :type eta: array_like
        :return: Values of the xy-derivatives of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_test_x = self.dtest_fcn(num_shape_func_in_1d, xi)[0]
        grad_test_y = self.dtest_fcn(num_shape_func_in_1d, eta)[0]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_test_x[i, :] * grad_test_y
            )

        return values

    def gradyy(self, xi, eta):
        """
        This method returns the yy-derivatives of the basis functions at the given (xi, eta) coordinates.

        :param xi: x-coordinates at which to evaluate the basis functions.
        :type xi: array_like
        :param eta: y-coordinates at which to evaluate the basis functions.
        :type eta: array_like

        :return: Values of the yy-derivatives of the basis functions.
        :rtype: array_like
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        grad_grad_y = self.dtest_fcn(num_shape_func_in_1d, eta)[1]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * grad_grad_y
            )

        return values
