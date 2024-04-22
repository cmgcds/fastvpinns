from src.FE_2D.basis_function_2d import *
from src.Geometry.geometry_2d import *
from src.FE_2D.fespace2d import *
import numpy as np
import pandas as pd
import pytest

# import the data handler class
from src.data.datahandler2d import *


## Aux functions ###
def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    return 32.0 * (x * (1 - x) + y * (1 - y))


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {
        1000: bottom_boundary,
        1001: right_boundary,
        1002: top_boundary,
        1003: left_boundary,
    }


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}


# lets get the cell information from the points
domain = Geometry_2D("quadrilateral")

# Read mesh from a .mesh file
cells, boundary_points = domain.read_mesh(
    "unitsquare_quad_4.mesh",
    boundary_point_refinement_level=5,
    sampling_method="uniform",
)

# get fespace2d
fespace = Fespace2D(
    mesh=domain.mesh,
    cells=cells,
    boundary_points=boundary_points,
    cell_type="quadrilateral",
    fe_order=1,
    quad_order=2,
    fe_transformation_type="affine",
    bound_function_dict=get_boundary_function_dict(),
    bound_condition_dict=get_bound_cond_dict(),
    forcing_function=rhs,
    output_path=".",
)


# perform the assembly operation on each cell
datahandler = DataHandler2D(fespace, batch=False, batch_size=32)


value_at_cell = []
grad_x_at_cell = []
grad_y_at_cell = []
area_at_cell = []
# perform assembly
for cell_id in range(len(cells)):
    # loop through quadrature points
    fe_cell = fespace.fe_cell[cell_id]

    # get the quadrature points
    num_quad_points = fe_cell.quad_weight.shape[0]
    num_basis = fe_cell.basis_at_quad.shape[0]

    # basis_matrix
    basis_function = datahandler.shape_function_val[cell_id].numpy()
    basis_gradx = datahandler.shape_function_grad_x[cell_id].numpy()
    basis_grady = datahandler.shape_function_grad_y[cell_id].numpy()
    forcing_function = datahandler.forcing_matrix[cell_id].numpy()
    jacobian = fe_cell.jacobian
    # print("Jacobian: ", jacobian)
    print("Basis function gradx: \n", basis_gradx)
    print("Basis function grady: \n", basis_grady)
    # print("Shape of basis function: ", basis_function.shape)
    # repeat the jacobian for all the quadrature points
    jacobian = np.repeat(jacobian, num_quad_points, axis=0)

    print("Jacobian : \n", jacobian)

    # generate random values for the predictor variables
    prediction = np.random.rand(num_quad_points)
    print("Prediction : \n", prediction)
    # get quad weights
    quad_weights = fe_cell.quad_weight

    print("Quad weights : \n", quad_weights)

    area = 0
    val = 0
    grad_val_x = 0
    grad_val_y = 0

    for q in range(num_quad_points):
        area += jacobian[q] * quad_weights[q]
        for i in range(num_basis):
            val += basis_function[i, q] * jacobian[q] * quad_weights[q] * prediction[q]
            grad_val_x += (
                basis_gradx[i, q] * jacobian[q] * quad_weights[q] * prediction[q]
            )
            grad_val_y += (
                basis_grady[i, q] * jacobian[q] * quad_weights[q] * prediction[q]
            )
    area_at_cell.append(area)
    value_at_cell.append(val)
    grad_x_at_cell.append(grad_val_x)
    grad_y_at_cell.append(grad_val_y)


print("Area : ", area_at_cell)
print("Area Total: ", np.sum(area_at_cell))
print("Basis : ", value_at_cell)
print("Total : ", np.sum(value_at_cell))
print("Grad-x value : ", grad_x_at_cell)
print("Grad-x Total: ", np.sum(grad_x_at_cell))
print("Grad-y value : ", grad_y_at_cell)
print("Grad-y Total: ", np.sum(grad_y_at_cell))
