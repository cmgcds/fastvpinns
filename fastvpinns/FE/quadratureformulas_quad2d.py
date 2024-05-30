"""
The file `quadratureformulas_quad2d.py` defines the Quadrature Formulas for the 2D Quadrilateral elements.
It supports both Gauss-Legendre and Gauss-Jacobi quadrature types.
The quadrature points and weights are calculated based on the specified quadrature order and type.

Author: Thivin Anandh D

Changelog: Not specified

Known issues: None

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.special import roots_legendre, roots_jacobi, jacobi, gamma
from scipy.special import legendre
from scipy.special import eval_legendre, legendre

from .quadratureformulas import Quadratureformulas


class Quadratureformulas_Quad2D(Quadratureformulas):
    """
    Defines the Quadrature Formulas for the 2D Quadrilateral elements.

    :param quad_order: The order of the quadrature.
    :type quad_order: int
    :param quad_type: The type of the quadrature.
    :type quad_type: str
    """

    def __init__(self, quad_order: int, quad_type: str):
        """
        Constructor for the Quadratureformulas_Quad2D class.

        :param quad_order: The order of the quadrature.
        :type quad_order: int
        :param quad_type: The type of the quadrature.
        :type quad_type: str
        """
        # initialize the super class
        super().__init__(
            quad_order=quad_order, quad_type=quad_type, num_quad_points=quad_order * quad_order
        )

        # Calculate the Gauss-Legendre quadrature points and weights for 1D
        # nodes_1d, weights_1d = roots_jacobi(self.quad_order, 1, 1)

        quad_type = self.quad_type

        if quad_type == "gauss-legendre":
            # Commented out by THIVIN -  to Just use legendre quadrature points as it is
            # if quad_order == 2:
            #     nodes_1d = np.array([-1, 1])
            #     weights_1d = np.array([1, 1])
            # else:
            nodes_1d, weights_1d = np.polynomial.legendre.leggauss(quad_order)  # Interior points
            # nodes_1d = np.concatenate(([-1, 1], nodes_1d))
            # weights_1d = np.concatenate(([1, 1], weights_1d))

            # Generate the tensor outer product of the nodes
            xi_quad, eta_quad = np.meshgrid(nodes_1d, nodes_1d)
            xi_quad = xi_quad.flatten()
            eta_quad = eta_quad.flatten()

            # Multiply the weights accordingly for 2D
            quad_weights = (weights_1d[:, np.newaxis] * weights_1d).flatten()

            # Assign the values
            self.xi_quad = xi_quad
            self.eta_quad = eta_quad
            self.quad_weights = quad_weights

        elif quad_type == "gauss-jacobi":

            def GaussJacobiWeights(Q: int, a, b):
                [X, W] = roots_jacobi(Q, a, b)
                return [X, W]

            def jacobi_wrapper(n, a, b, x):

                x = np.array(x, dtype=np.float64)

                return jacobi(n, a, b)(x)

            # Weight coefficients
            def GaussLobattoJacobiWeights(Q: int, a, b):
                W = []
                X = roots_jacobi(Q - 2, a + 1, b + 1)[0]
                if a == 0 and b == 0:
                    W = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, X) ** 2))
                    Wl = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, -1) ** 2))
                    Wr = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, 1) ** 2))
                else:
                    W = (
                        2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, X) ** 2)
                        )
                    )
                    Wl = (
                        (b + 1)
                        * 2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, -1) ** 2)
                        )
                    )
                    Wr = (
                        (a + 1)
                        * 2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, 1) ** 2)
                        )
                    )
                W = np.append(W, Wr)
                W = np.append(Wl, W)
                X = np.append(X, 1)
                X = np.append(-1, X)
                return [X, W]

            # get quadrature points and weights in 1D
            x, w = GaussLobattoJacobiWeights(self.quad_order, 0, 0)

            # Generate the tensor outer product of the nodes
            xi_quad, eta_quad = np.meshgrid(x, x)
            xi_quad = xi_quad.flatten()
            eta_quad = eta_quad.flatten()

            # Multiply the weights accordingly for 2D
            quad_weights = (w[:, np.newaxis] * w).flatten()

            # Assign the values
            self.xi_quad = xi_quad
            self.eta_quad = eta_quad
            self.quad_weights = quad_weights

        else:
            print("Supported quadrature types are: gauss-legendre, gauss-jacobi")
            print(
                f"Invalid quadrature type {quad_type} in {self.__class__.__name__} from {__name__}."
            )
            raise ValueError("Quadrature type not supported.")

    def get_quad_values(self):
        """
        Returns the quadrature weights, xi and eta values.

        :return: A tuple containing the quadrature weights, xi values, and eta values.
        :rtype: tuple
        """
        return self.quad_weights, self.xi_quad, self.eta_quad

    def get_num_quad_points(self):
        """
        Returns the number of quadrature points.

        :return: The number of quadrature points.
        :rtype: int
        """
        return self.num_quad_points
