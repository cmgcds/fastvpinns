"""
file: basis_function_3d.py
description: Abstract class for all basis functions 3D
authors: Thivin Anandh D
changelog: 03/May/2024 - First version
known_issues: None
dependencies: None specified.
"""

from abc import abstractmethod


class BasisFunction3D:  # pragma: no cover
    """
    Represents a basis function in 3D.

    Args:
        num_shape_functions (int): The number of shape functions.

    Methods:
        value(xi, eta, zeta): Evaluates the basis function at the given xi and eta coordinates.
        gradx(xi, eta, zeta): Computes the partial derivative of the basis function with respect to xi.
        grady(xi, eta, zeta): Computes the partial derivative of the basis function with respect to eta.
        gradxx(xi, eta, zeta): Computes the second partial derivative of the basis function with respect to xi.
        gradxy(xi, eta, zeta): Computes the mixed partial derivative of the basis function with respect to xi and eta.
        gradyy(xi, eta, zeta): Computes the second partial derivative of the basis function with respect to eta.
    """

    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions

    @abstractmethod
    def value(self, xi, eta, zeta):
        """
        Evaluates the basis function at the given xi and eta coordinates.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The value of the basis function at the given coordinates.
        :rtype: float
        """

    @abstractmethod
    def gradx(self, xi, eta, zeta):
        """
        Computes the partial derivative of the basis function with respect to xi.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The partial derivative of the basis function with respect to xi.
        :rtype: float
        """

    @abstractmethod
    def grady(self, xi, eta, zeta):
        """
        Computes the partial derivative of the basis function with respect to eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The partial derivative of the basis function with respect to eta.
        :rtype: float
        """

    @abstractmethod
    def gradxx(self, xi, eta, zeta):
        """
        Computes the second partial derivative of the basis function with respect to xi.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The second partial derivative of the basis function with respect to xi.
        :rtype: float
        """

    @abstractmethod
    def gradxy(self, xi, eta, zeta):
        """
        Computes the mixed partial derivative of the basis function with respect to xi and eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The mixed partial derivative of the basis function with respect to xi and eta.
        :rtype: float
        """

    @abstractmethod
    def gradyy(self, xi, eta, zeta):
        """
        Computes the second partial derivative of the basis function with respect to eta.

        :param float xi: The xi coordinate.
        :param float eta: The eta coordinate.
        :return: The second partial derivative of the basis function with respect to eta.
        :rtype: float
        """
