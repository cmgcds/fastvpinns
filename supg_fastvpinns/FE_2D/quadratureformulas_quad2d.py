import numpy as np
from scipy.special import roots_legendre, roots_jacobi, jacobi, gamma
from scipy.special import legendre
from scipy.special import eval_legendre, legendre


class Quadratureformulas_Quad2D:
    """
    Defines the Quadrature Formulas for the 2D Quadrilateral elements.

    Attributes:
    - quad_order (int): The order of the quadrature.
    - quad_type (str): The type of the quadrature.
    - num_quad_points (int): The number of quadrature points.

    Methods:
    - get_quad_values(): Returns the quadrature weights, xi and eta values.
    - get_num_quad_points(): Returns the number of quadrature points.

    """
    
    def __init__(self, quad_order: int, quad_type: str):
        """
        The constructor of the Quadformulas_2D class.

        Parameters:
        - quad_order (int): The order of the quadrature.
        - quad_type (str): The type of the quadrature.

        Returns:
        None
        """
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.num_quad_points = quad_order * quad_order
        
        if self.quad_order < 2:
            raise Exception("Quadrature order should be greater than 1.")
        
        # Calculate the Gauss-Legendre quadrature points and weights for 1D
        # nodes_1d, weights_1d = roots_jacobi(self.quad_order, 1, 1)


        quad_type = self.quad_type

        if quad_type == "gauss-legendre":
            """
            This method returns the Gauss-Legendre quadrature points and weights.
            """
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
        
        

        elif quad_type == "gauss-lobatto":
            """
            This method returns the Gauss-Lobatto quadrature points and weights.
            """
            def lgP(n, xi):
                """
                Evaluates P_{n}(xi) using an iterative algorithm
                """
                obj1  = legendre(n+1)
                obj2 = legendre(n-1)
                test = obj1(xi) - obj2(xi)

                return test

            def dLgP(n, xi):
                """
                Evaluates the first derivative of P_{n}(xi)
                """
                obj1 = legendre(n+1).deriv()
                obj2 = legendre(n-1).deriv()
                test  = obj1(xi) - obj2(xi)
                return test

            def d2LgP(n, xi):
                """
                Evaluates the second derivative of P_{n}(xi)
                """
                obj1 = legendre(n+1).deriv().deriv()
                obj2 = legendre(n-1).deriv().deriv()
                test  = obj1(xi) - obj2(xi)
                return test

            def d3LgP(n, xi):
                """
                Evaluates the third derivative of P_{n}(xi)
                """
                obj1 = legendre(n+1).deriv().deriv().deriv()
                obj2 = legendre(n-1).deriv().deriv().deriv()
                test  = obj1(xi) - obj2(xi)
                return test

            def gLLNodesAndWeights(n, epsilon=1e-15):
                """
                Computes the GLL nodes and weights
                """
                if n < 2:
                    print('Error: n must be larger than 1')
                else:
                    x = np.empty(n)
                    w = np.empty(n)

                    x[0] = -1
                    x[n - 1] = 1
                    w[0] = w[0] = 2.0 / ((n * (n - 1)))
                    w[n - 1] = w[0]

                    n_2 = n // 2

                    for i in range(1, n_2):
                        xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) * np.cos((4 * i + 1) * np.pi / (4 * (n - 1) + 1))

                        error = 1.0

                        while error > epsilon:
                            y = dLgP(n - 1, xi)
                            y1 = d2LgP(n - 1, xi)
                            y2 = d3LgP(n - 1, xi)

                            dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

                            xi -= dx
                            error = abs(dx)

                        x[i] = -xi
                        x[n - i - 1] = xi

                        w[i] = 2 / (n * (n - 1) * lgP(n - 1, x[i]) ** 2)
                        w[n - i - 1] = w[i]

                    if n % 2 != 0:
                        x[n_2] = 0
                        w[n_2] = 2.0 / ((n * (n - 1)) * lgP(n - 1, np.array(x[n_2])) ** 2)

                    return x, w

            x, w = gLLNodesAndWeights(self.quad_order)

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
        
        elif quad_type == "gauss-jacobi":
            """
            This method returns the Gauss-Jacobi quadrature points and weights.
            """
            def GaussJacobiWeights(Q: int,a,b):
                [X , W] = roots_jacobi(Q,a,b)
                return [X, W]

            def Jacobi(n,a,b,x):

                x=np.array(x, dtype=np.float64)

                return (jacobi(n,a,b)(x))

            # Weight coefficients
            def GaussLobattoJacobiWeights(Q: int,a,b):
                W = []
                X = roots_jacobi(Q-2,a+1,b+1)[0]
                if a == 0 and b==0:
                    W = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,X)**2) )
                    Wl = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,-1)**2) )
                    Wr = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,1)**2) )
                else:
                    W = 2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,X)**2) )
                    Wl = (b+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,-1)**2) )
                    Wr = (a+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,1)**2) )
                W = np.append(W , Wr)
                W = np.append(Wl , W)
                X = np.append(X , 1)
                X = np.append(-1 , X)    
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
            print("Supported quadrature types are: gauss-legendre, gauss-lobatto and gauss-jacobi, gauss-lobatto-new")
            print(f"Invalid quadrature type {quad_type} in {self.__class__.__name__} from {__name__}.")
            raise Exception("Quadrature type not supported.")



    def get_quad_values(self):
        """
        Returns the quadrature weights, xi and eta values.
        """
        return self.quad_weights, self.xi_quad, self.eta_quad

    def get_num_quad_points(self):
        """
        Returns the number of quadrature points.
        """
        return self.num_quad_points