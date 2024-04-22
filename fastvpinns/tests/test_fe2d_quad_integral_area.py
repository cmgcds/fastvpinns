from src.FE_2D.basis_function_2d import *
from src.Geometry.geometry_2d import *
from src.FE_2D.fespace2d import *
import numpy as np
import pandas as pd
import pytest


import pytest


## Auxilaary functions, which are required for the construction of FE Space within the element ##
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


###------ End of Auxilary functions -----###


# @pytest.mark.parametrize("cell_type", ["quadrilateral"])
# @pytest.mark.parametrize("quad_order", [2,3,4,5,6,7,8,9])
# @pytest.mark.parametrize("fe_transformation", ["affine", "bilinear"])
# @pytest.mark.parametrize("fe_order", [0,1,2,3,4,5,6,7,8,9])
# def test_square_area_without_shape_function(cell_type, fe_order, quad_order, fe_transformation):
#     """
#     This function will test the area of a square without using the shape functions.
#     """


#     domain = Geometry_2D("quadrilateral")

#     # Read mesh from a .mesh file
#     cells, boundary_points = domain.read_mesh("unitsquare_quad.mesh", boundary_point_refinement_level=5, sampling_method="uniform")

#     # get fespace2d
#     fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, \
#                         cell_type=cell_type,fe_order=fe_order, quad_order=quad_order, fe_transformation_type=fe_transformation,
# bound_function_dict = get_boundary_function_dict(), bound_condition_dict = get_bound_cond_dict(), \
#             forcing_function=rhs, output_path=".")

#     # lets compute the area of the domain
#     area = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum( (fespace.fe_cell[i].quad_weight * 1.0 * fespace.fe_cell[i].jacobian))
#         # print(f"Cell {i} : {val}")
#         area += val

#     area2 = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum(fespace.fe_cell[i].basis_at_quad)
#         # print(f"Cell {i} : {val}")
#         area2 += val

#     # assert if area is 1.0
#     assert (area == pytest.approx(1.0, abs=1e-6)) and (area2 == pytest.approx(1.0, abs=1e-6))


# @pytest.mark.parametrize("cell_type", ["quadrilateral"])
# @pytest.mark.parametrize("quad_order", [2,3,4,5,6,7,8,9])
# @pytest.mark.parametrize("fe_transformation", ["affine", "bilinear"])
# @pytest.mark.parametrize("fe_order", [0,1,2,3,4,5,6,7,8,9])
# def test_circle_area_without_shape_function(cell_type, fe_order, quad_order, fe_transformation):
#     """
#     This function will test the area of a square without using the shape functions.
#     """


#     domain = Geometry_2D("quadrilateral")

#     # Read mesh from a .mesh file
#     cells, boundary_points = domain.read_mesh("tests/circle_quad.mesh", boundary_point_refinement_level=5, sampling_method="uniform")

#     # get fespace2d
#     fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, \
#                         cell_type=cell_type,fe_order=fe_order, quad_order=quad_order, fe_transformation_type=fe_transformation,
# bound_function_dict = get_boundary_function_dict(), bound_condition_dict = get_bound_cond_dict(), \
#             forcing_function=rhs, output_path=".")

#     # lets compute the area of the domain
#     area = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum( (fespace.fe_cell[i].quad_weight * 1.0 * fespace.fe_cell[i].jacobian))
#         # print(f"Cell {i} : {val}")
#         area += val

#     area2 = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum(fespace.fe_cell[i].basis_at_quad)
#         # print(f"Cell {i} : {val}")
#         area2 += val

#     # assert if area is 1.0
#     assert (area == pytest.approx(np.pi, abs=1e-1)) and (area2 == pytest.approx(np.pi, abs=1e-1))   # the Quads cannot approximate the circle well enough


# @pytest.mark.parametrize("cell_type", ["triangle"])
# @pytest.mark.parametrize("quad_order", [1,2,3,7,8,9,11,19,100])
# @pytest.mark.parametrize("fe_transformation", ["affine"])
# @pytest.mark.parametrize("fe_order", [0,1,2,3,4,5,6])
# def test_square_area_without_shape_function(cell_type, fe_order, quad_order, fe_transformation):
#     """
#     This function will test the area of a square without using the shape functions.
#     """

#     print(f"-- Testing with: cell_type={cell_type}, fe_order={fe_order}, quad_order={quad_order}, fe_transformation={fe_transformation}")
#     domain = Geometry_2D("triangle")

#     # Read mesh from a .mesh file
#     cells, boundary_points = domain.read_mesh("tests/unitsquare_tri.mesh", boundary_point_refinement_level=5, sampling_method="uniform")

#     # get fespace2d
#     fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, \
#                         cell_type=cell_type,fe_order=fe_order, quad_order=quad_order, fe_transformation_type=fe_transformation,
# bound_function_dict = get_boundary_function_dict(), bound_condition_dict = get_bound_cond_dict(), \
#             forcing_function=rhs, output_path=".")

#     # lets compute the area of the domain
#     area = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum( (fespace.fe_cell[i].quad_weight * 1.0 * fespace.fe_cell[i].jacobian))
#         # print(f"Cell {i} : {val}")
#         area += val

#     area2 = 0
#     for i in range(fespace.n_cells):
#         val =  np.sum(fespace.fe_cell[i].basis_at_quad)
#         # print(f"Cell {i} : {val}")
#         area2 += val
#     print(f"Area without Shape : {area} and with Shape : {area2}")
#     # assert if area is 1.0
#     assert (area == pytest.approx(1.0, abs=1e-5)) and (area2 == pytest.approx(1.0, abs=1e-5))


@pytest.mark.parametrize("cell_type", ["triangle"])
@pytest.mark.parametrize("quad_order", [1, 2, 3, 7, 8, 9, 11, 19, 100])
@pytest.mark.parametrize("fe_transformation", ["affine"])
@pytest.mark.parametrize("fe_order", [0, 1, 2, 3, 4, 5, 6])
def test_circle_area_without_shape_function(
    cell_type, fe_order, quad_order, fe_transformation
):
    """
    This function will test the area of a square without using the shape functions.
    """
    print(
        f"Testing with: cell_type={cell_type}, fe_order={fe_order}, quad_order={quad_order}, fe_transformation={fe_transformation}"
    )
    domain = Geometry_2D("triangle")

    # Read mesh from a .mesh file
    cells, boundary_points = domain.read_mesh(
        "tests/circle_tri.mesh",
        boundary_point_refinement_level=5,
        sampling_method="uniform",
    )

    # get fespace2d
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=cell_type,
        fe_order=fe_order,
        quad_order=quad_order,
        fe_transformation_type=fe_transformation,
        bound_function_dict=get_boundary_function_dict(),
        bound_condition_dict=get_bound_cond_dict(),
        forcing_function=rhs,
        output_path=".",
    )

    # lets compute the area of the domain
    area = 0
    for i in range(fespace.n_cells):
        val = np.sum(
            (fespace.fe_cell[i].quad_weight * 1.0 * fespace.fe_cell[i].jacobian)
        )
        # print(f"Cell {i} : {val}")
        area += val

    area2 = 0
    for i in range(fespace.n_cells):
        val = np.sum(fespace.fe_cell[i].basis_at_quad)
        # print(f"Cell {i} : {val}")
        area2 += val

    # assert if area is 1.0
    assert (area == pytest.approx(np.pi, abs=1e-1)) and (
        area2 == pytest.approx(np.pi, abs=1e-1)
    )  # the Quads cannot approximate the circle well enough


if __name__ == "__main__":
    """
    Run All the test cases for the basis functions
    """

    test_square_area_without_shape_function()
    test__area_without_shape_function()
