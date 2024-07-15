"""
The file `basis_function_2d.py` contains a wrapper class for all the finite element basis functions 
used in the FE2D code. The 2D basis functions have methods to return the value 
of the basis function and its derivatives at the reference point (xi, eta).

Author: Thivin Anandh D

Changelog: 30/Aug/2023 - First version

Known issues: None

Dependencies: None specified
"""

from abc import abstractmethod


class BasisFunction2D:
    """
    This class defines the basis functions for a 2D element.

    :num_shape_functions (int): The number of shape functions.
    :value(xi, eta): Evaluates the basis function at the given xi and eta coordinates.
    """

    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions

    @abstractmethod
    def value(self, xi, eta):
        """
        Evaluates the basis function at the given xi and eta coordinates.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The value of the basis function at the given coordinates.
        :rtype: float
        """
        pass

    @abstractmethod
    def gradx(self, xi, eta):
        """
        Computes the partial derivative of the basis function with respect to xi.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The partial derivative of the basis function with respect to xi.
        :rtype: float
        """
        pass

    @abstractmethod
    def grady(self, xi, eta):
        """
        Computes the partial derivative of the basis function with respect to eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The partial derivative of the basis function with respect to eta.
        :rtype: float
        """
        pass

    @abstractmethod
    def gradxx(self, xi, eta):
        """
        Computes the second partial derivative of the basis function with respect to xi.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The second partial derivative of the basis function with respect to xi.
        :rtype: float
        """
        pass

    @abstractmethod
    def gradxy(self, xi, eta):
        """
        Computes the mixed partial derivative of the basis function with respect to xi and eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The mixed partial derivative of the basis function with respect to xi and eta.
        :rtype: float
        """
        pass

    @abstractmethod
    def gradyy(self, xi, eta):
        """
        Computes the second partial derivative of the basis function with respect to eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The second partial derivative of the basis function with respect to eta.
        :rtype: float
        """
        pass


# ---------------- Legendre -------------------------- #
from .basis_2d_QN_Legendre import *  # Normal Legendre from Jacobi -> J(n) = J(n-1) - J(n+1)
from .basis_2d_QN_Legendre_Special import *  # L(n) = L(n-1) - L(n+1)

# ---------------- Jacobi -------------------------- #
from .basis_2d_QN_Jacobi import *  # Normal Jacobi

# ---------------- Chebyshev -------------------------- #
from .basis_2d_QN_Chebyshev_2 import *  # Normal Chebyshev
