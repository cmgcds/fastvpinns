# This program will be used to setup the FE2D and quadrature rule for a given cell based on the
# given mesh and the degree of the basis functions

# Author: Thivin Anandh D
# Date:  30/Aug/2023
# Implementation History :
# the grad_x_orig and grad_y_orig will actually store the magnitute with which we need to multiply this grad_x_ref and grad_y_ref
# to obtain the actual values of the gradient in the original cell
# this is done to improve efficiency


# Importing the required libraries
# from .basis_function_2d import *

# import Quadrature rules
from .quadratureformulas_quad2d import *
from .fe2d_setup_main import *


class FE2D_Cell:
    """
    This class is used to Store the FE Values, Such as Coordinates, Basis Functions, Quadrature Rules, etc. for a given cell.
    """

    def __init__(
        self,
        cell_coordinates: np.ndarray,
        cell_type: str,
        fe_order: int,
        fe_type: str,
        quad_order: int,
        quad_type: str,
        fe_transformation_type: str,
        forcing_function,
    ):
        self.cell_coordinates = cell_coordinates
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.fe_transformation = fe_transformation_type
        self.forcing_function = forcing_function

        # Basis function Class
        self.basis_function = None

        # Quadrature Values
        self.quad_xi = None
        self.quad_eta = None
        self.quad_weight = None
        self.jacobian = None
        self.mult = None

        # FE Values
        self.basis_at_quad = None
        self.basis_gradx_at_quad = None
        self.basis_grady_at_quad = None
        self.basis_gradxy_at_quad = None
        self.basis_gradxx_at_quad = None
        self.basis_gradyy_at_quad = None

        # Quadrature Coordinates
        self.quad_actual_coordinates = None

        # Forcing function values at the quadrature points
        self.forcing_at_quad = None

        # FE Transformation Class
        self.fetransformation = None

        # get instance of the FE_setup class
        self.fe_setup = FE2DSetupMain(
            cell_type=self.cell_type,
            fe_order=self.fe_order,
            fe_type=self.fe_type,
            quad_order=self.quad_order,
            quad_type=self.quad_type,
        )

        # Call the function to assign the basis function
        self.assign_basis_function()

        # Assign the quadrature points and weights
        self.assign_quadrature()

        # Assign the FE Transformation
        self.assign_fe_transformation()

        # calculate mult -> quadrature weights * Jacobian
        self.assign_quad_weights_and_jacobian()

        # Calculate the basis function values at the quadrature points
        self.assign_basis_values_at_quadrature_points()

        # calculate the actual coordinates of the quadrature points
        self.assign_quadrature_coordinates()

        # Calculate the forcing function values at the actual quadrature points
        # NOTE : The function is just for printing the shape of the force matrix, the
        # actual calculation is performed on the fespace class
        self.assign_forcing_term(self.forcing_function)

        # # print the values
        # print("============================================================================")
        # print("Cell Co-ord : ", self.cell_coordinates)
        # print("Basis function values at the quadrature points: \n", self.basis_at_quad / self.mult)
        # print("Basis function gradx at the quadrature points: \n", self.basis_gradx_at_quad)
        # print("Basis function grady at the quadrature points: \n", self.basis_grady_at_quad)
        # print("Forcing function values at the quadrature points: \n", self.forcing_at_quad)

        # grad_x = np.array([5,6,7,8])
        # grad_y = np.array([1,2,3,4])

        # pde = np.matmul(self.basis_gradx_at_quad, grad_x.reshape(-1,1)) + np.matmul(self.basis_grady_at_quad, grad_y.reshape(-1,1))
        # print("PDE values at the quadrature points: \n", pde)

    def assign_basis_function(self) -> BasisFunction2D:
        """
        The function will assign the basis function class based on the cell type and the FE order.
        """
        self.basis_function = self.fe_setup.assign_basis_function()

    def assign_quadrature(self) -> None:
        """
        The function will assign the quadrature points and weights based on the cell type and the quadrature order.
        """
        self.quad_weight, self.quad_xi, self.quad_eta = self.fe_setup.assign_quadrature_rules()

    def assign_fe_transformation(self) -> None:
        """
        The function will assign the FE Transformation class based on the cell type and the FE order.
        """
        self.fetransformation = self.fe_setup.assign_fe_transformation(
            self.fe_transformation, self.cell_coordinates
        )
        # Sets cell co-ordinates for the FE Transformation
        self.fetransformation.set_cell()

        # obtains the Jacobian of the transformation
        self.jacobian = self.fetransformation.get_jacobian(self.quad_xi, self.quad_eta).reshape(
            -1, 1
        )

    def assign_basis_values_at_quadrature_points(self) -> None:
        """
        The function will assign the basis function values at the quadrature points.
        There are N_quad_points, So the basis function values will be of size N_basis_functions x N_quad_points
        """
        self.basis_at_quad = []
        self.basis_gradx_at_quad = []
        self.basis_grady_at_quad = []
        self.basis_gradxy_at_quad = []
        self.basis_gradxx_at_quad = []
        self.basis_gradyy_at_quad = []

        # print("quad_x\n", self.quad_xi)
        # print("quad_y\n", self.quad_eta)
        # print("quad_weight\n", self.quad_weight)
        # exit(1)
        self.basis_at_quad = self.basis_function.value(self.quad_xi, self.quad_eta)

        # For Gradients we need to perform a transformation to the original cell
        grad_x_ref = self.basis_function.gradx(self.quad_xi, self.quad_eta)
        grad_y_ref = self.basis_function.grady(self.quad_xi, self.quad_eta)

        grad_x_orig, grad_y_orig = self.fetransformation.get_orig_from_ref_derivative(
            grad_x_ref, grad_y_ref, self.quad_xi, self.quad_eta
        )

        self.basis_gradx_at_quad = grad_x_orig
        self.basis_grady_at_quad = grad_y_orig

        self.basis_gradx_at_quad_ref = grad_x_ref
        self.basis_grady_at_quad_ref = grad_y_ref

        # get the double derivatives of the basis functions ( ref co-ordinates )
        grad_xx_ref = self.basis_function.gradxx(self.quad_xi, self.quad_eta)
        grad_xy_ref = self.basis_function.gradxy(self.quad_xi, self.quad_eta)
        grad_yy_ref = self.basis_function.gradyy(self.quad_xi, self.quad_eta)

        # get the double derivatives of the basis functions ( orig co-ordinates )
        grad_xx_orig, grad_xy_orig, grad_yy_orig = (
            self.fetransformation.get_orig_from_ref_second_derivative(
                grad_xx_ref, grad_xy_ref, grad_yy_ref, self.quad_xi, self.quad_eta
            )
        )

        # = the value
        self.basis_gradxy_at_quad = grad_xy_orig
        self.basis_gradxx_at_quad = grad_xx_orig
        self.basis_gradyy_at_quad = grad_yy_orig

        # # print the values
        # print("=============================================================================")
        # print("cell co-ordinates: \n", self.cell_coordinates)
        # print("Shape functions : \n", self.basis_at_quad)
        # print("Shape functions gradx : \n", self.basis_gradx_at_quad)
        # print("Shape functions grady : \n", self.basis_grady_at_quad)

        # Multiply each row with the quadrature weights
        # Basis at Quad - n_test * N_quad
        self.basis_at_quad = self.basis_at_quad * self.mult
        self.basis_gradx_at_quad = self.basis_gradx_at_quad * self.mult
        self.basis_grady_at_quad = self.basis_grady_at_quad * self.mult
        self.basis_gradxy_at_quad = self.basis_gradxy_at_quad * self.mult
        self.basis_gradxx_at_quad = self.basis_gradxx_at_quad * self.mult
        self.basis_gradyy_at_quad = self.basis_gradyy_at_quad * self.mult

        # print("mat\n",self.basis_at_quad)
        # print("mat_grad_x\n",self.basis_gradx_at_quad)
        # print("mat_grad_y\n",self.basis_grady_at_quad)
        # exit(1)

        # print the values
        # print("=============================================================================")
        # print("cell co-ordinates: \n", self.cell_coordinates)
        # print("Basis function values at the quadrature points: \n", self.basis_at_quad)
        # print("Basis function gradx at the quadrature points: \n", self.basis_gradx_at_quad)
        # print("Basis function grady at the quadrature points: \n", self.basis_grady_at_quad)

    def assign_quad_weights_and_jacobian(self) -> None:
        """
        The function will assign the quadrature weights and the Jacobian of the transformation.
        """
        self.mult = self.quad_weight * self.jacobian.flatten()

    def assign_quadrature_coordinates(self) -> None:
        """
        The function will assign the actual coordinates of the quadrature points.
        """
        # Input -> Xi (NQuad,), Eta (NQuad,) --> output -> (2 * NQuad) --> Transpose it to get (NQuad, 2)
        actual_co_ord = []
        for xi, eta in zip(self.quad_xi, self.quad_eta):
            actual_co_ord.append(self.fetransformation.get_original_from_ref(xi, eta))

        self.quad_actual_coordinates = np.array(actual_co_ord)

    def assign_forcing_term(self, forcing_function) -> None:
        """
        The function will assign the forcing function values at the quadrature points.
        The final shape will be N_shape_functions x 1
        """

        # get number of shape functions
        n_shape_functions = self.basis_function.num_shape_functions

        # Loop over all the basis functions and compute the integral
        f_integral = np.zeros((n_shape_functions, 1), dtype=np.float64)

        # The above code is for backward compatability with old code. this function will just assign the values as zeros
        # the actual calculation is performed in the fespace class

        # for i in range(n_shape_functions):
        #     val = 0
        #     for q in range(self.basis_at_quad.shape[1]):
        #         x = self.quad_actual_coordinates[q, 0]
        #         y = self.quad_actual_coordinates[q, 1]
        #         # print("f_values[q] = ",f_values[q])

        #         # the JAcobian and the quadrature weights are pre multiplied to the basis functions
        #         val +=  ( self.basis_at_quad[i, q] ) * self.forcing_function(x, y)
        #         # print("val = ", val)

        #     f_integral[i] = val

        self.forcing_at_quad = f_integral
