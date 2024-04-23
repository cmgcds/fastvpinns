# This class is a wrapper class to all the finite element basis functions that
# are used in the FE2D code. The 2D basis functions, will have the following
# methods:
#   1. value(xi, eta) - This will return the value of the basis function at the reference point (xi, eta)
#   2. gradx(xi, eta) - This will return the value of the derivative of the basis function with respect to xi
#   3. grady(xi, eta) - This will return the value of the derivative of the basis function with respect to eta
#   4. gradxx(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi
#   5. gradxy(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi and eta


# Author: Thivin Anandh D
# Date:  30/Aug/2023
# History: First version - 30/Aug/2023 - Thivin Anandh D

from abc import ABC, abstractmethod


class BasisFunction2D:
    """
    Represents a basis function in 2D.

    Args:
        num_shape_functions (int): The number of shape functions.

    Methods:
        value(xi, eta): Evaluates the basis function at the given xi and eta coordinates.
        gradx(xi, eta): Computes the partial derivative of the basis function with respect to xi.
        grady(xi, eta): Computes the partial derivative of the basis function with respect to eta.
        gradxx(xi, eta): Computes the second partial derivative of the basis function with respect to xi.
        gradxy(xi, eta): Computes the mixed partial derivative of the basis function with respect to xi and eta.
        gradyy(xi, eta): Computes the second partial derivative of the basis function with respect to eta.
    """

    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions

    @abstractmethod
    def value(self, xi, eta):
        """
        Evaluates the basis function at the given xi and eta coordinates.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The value of the basis function at the given coordinates.
        """
        pass

    @abstractmethod
    def gradx(self, xi, eta):
        """
        Computes the partial derivative of the basis function with respect to xi.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The partial derivative of the basis function with respect to xi.
        """
        pass

    @abstractmethod
    def grady(self, xi, eta):
        """
        Computes the partial derivative of the basis function with respect to eta.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The partial derivative of the basis function with respect to eta.
        """
        pass

    @abstractmethod
    def gradxx(self, xi, eta):
        """
        Computes the second partial derivative of the basis function with respect to xi.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The second partial derivative of the basis function with respect to xi.
        """
        pass

    @abstractmethod
    def gradxy(self, xi, eta):
        """
        Computes the mixed partial derivative of the basis function with respect to xi and eta.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The mixed partial derivative of the basis function with respect to xi and eta.
        """
        pass

    @abstractmethod
    def gradyy(self, xi, eta):
        """
        Computes the second partial derivative of the basis function with respect to eta.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The second partial derivative of the basis function with respect to eta.
        """
        pass


# ---------------- Legendre -------------------------- #
from .basis_2d_QN_Legendre import *  # Normal Legendre from Jacobi -> J(n) = J(n-1) - J(n+1)
from .basis_2d_QN_Legendre_Special import *  # L(n) = L(n-1) - L(n+1)

# ---------------- Jacobi -------------------------- #
from .basis_2d_QN_Jacobi import *  # Normal Jacobi

# ---------------- Chebyshev -------------------------- #
from .basis_2d_QN_Chebyshev_2 import *  # Normal Chebyshev
