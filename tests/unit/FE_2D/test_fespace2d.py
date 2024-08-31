# Author : Thivin Anandh. D
# Added test cases for validating Quadrature routines by computing the areas.
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
import pytest
import shutil


@pytest.fixture()
def test_quadrature_uniform(request):
    """Tests Quadrature routines for different quadrature types and transformations by calculating Area"""

    coord = request.param
    quad_type = "gauss"
    transformation = "affine"

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=4, n_cells_y=4, num_boundary_points=100
    )

    bound_function_dict = {
        1000: lambda x, y: np.ones_like(x),
        1001: lambda x, y: np.ones_like(x),
        1002: lambda x, y: np.ones_like(x),
        1003: lambda x, y: np.ones_like(x),
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: np.ones_like(x)

    return (
        domain,
        cells,
        boundary_points,
        bound_function_dict,
        bound_condition_dict,
        rhs,
        quad_type,
        transformation,
    )


@pytest.mark.parametrize(
    "fe_type", ["legendre", "jacobi", "legendre_special", "chebyshev_2", "jacobi_plain"]
)
@pytest.mark.parametrize("transformation", ["affine", "bilinear"])
def test_shape_functions(fe_type, transformation):
    """Tests Shape function values for its shape and size"""

    # using pathlib create a temporary directory tests/dump
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=2, n_cells_y=2, num_boundary_points=10
    )

    bound_function_dict = {
        1000: lambda x, y: np.ones_like(x),
        1001: lambda x, y: np.ones_like(x),
        1002: lambda x, y: np.ones_like(x),
        1003: lambda x, y: np.ones_like(x),
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: np.ones_like(x)

    # Create fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        # make fe_order a random number between 2 and 5
        fe_order=np.random.randint(2, 8),
        fe_type=fe_type,
        # make quad a random number between 3 and 10
        quad_order=np.random.randint(3, 10),
        quad_type="gauss-legendre",
        fe_transformation_type=transformation,
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=False,
    )

    # get the first cell and check all its parameters
    fe_cell = fespace.fe_cell[0]

    # get all the matrices
    basis_at_quad = fespace.fe_cell[0].basis_at_quad
    basis_gradx_at_quad = fespace.fe_cell[0].basis_gradx_at_quad
    basis_grady_at_quad = fespace.fe_cell[0].basis_grady_at_quad
    basis_gradxy_at_quad = fespace.fe_cell[0].basis_gradxy_at_quad
    basis_gradxx_at_quad = fespace.fe_cell[0].basis_gradxx_at_quad
    basis_gradyy_at_quad = fespace.fe_cell[0].basis_gradyy_at_quad

    # Assert their shapes
    assert basis_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)
    assert basis_gradx_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)
    assert basis_grady_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)
    assert basis_gradxy_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)
    assert basis_gradxx_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)
    assert basis_gradyy_at_quad.shape == (fespace.fe_order**2, fespace.quad_order**2)

    # Clean up objects
    del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

    # remove the temporary directory even if it has files in it
    shutil.rmtree("tests/dump")


# Check the generate plot function
def test_generate_plot():
    """Tests the generate plot function"""

    ## generate a temporary directory
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=2, n_cells_y=2, num_boundary_points=10
    )

    bound_function_dict = {
        1000: lambda x, y: np.ones_like(x),
        1001: lambda x, y: np.ones_like(x),
        1002: lambda x, y: np.ones_like(x),
        1003: lambda x, y: np.ones_like(x),
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: np.ones_like(x)

    # Create fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        # make fe_order a random number between 2 and 5
        fe_order=np.random.randint(2, 8),
        fe_type="legendre",
        # make quad a random number between 3 and 10
        quad_order=np.random.randint(3, 10),
        quad_type="gauss-legendre",
        fe_transformation_type="affine",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=True,
    )

    # assert the existence of the plot
    assert Path("tests/dump/mesh.png").exists()

    # Clean up objects
    del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

    # remove the plot
    shutil.rmtree("tests/dump")


# Test dirichlet boundary data vector
# Commented out by thivin on 31-Aug-2023
# reason: The NSE branch have a different way of handling the vector valued data functions.
# The test cases will be added in the NSE branch.
# def test_dirichlet_boundary_data_vector():
#     """Tests the dirichlet boundary data vector"""

#     # use pathlib to create a temporary directory
#     Path("tests/dump").mkdir(parents=True, exist_ok=True)

#     # Define the geometry
#     domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
#     cells, boundary_points = domain.generate_quad_mesh_internal(
#         x_limits=[0, 1], y_limits=[0, 1], n_cells_x=2, n_cells_y=2, num_boundary_points=10
#     )

#     bval_1 = np.random.rand()
#     bval_2 = np.random.rand()
#     # Vector valued boundary function
#     bound_function_dict = {
#         1000: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
#         1001: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
#         1002: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
#         1003: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
#     }
#     bound_condition_dict = {
#         1000: "dirichlet",
#         1001: "dirichlet",
#         1002: "dirichlet",
#         1003: "dirichlet",
#     }
#     rhs = lambda x, y: np.ones_like(x)

#     # Create fespace
#     fespace = Fespace2D(
#         mesh=domain.mesh,
#         cells=cells,
#         boundary_points=boundary_points,
#         cell_type=domain.mesh_type,
#         # make fe_order a random number between 2 and 5
#         fe_order=np.random.randint(2, 8),
#         fe_type="legendre",
#         # make quad a random number between 3 and 10
#         quad_order=np.random.randint(3, 10),
#         quad_type="gauss-legendre",
#         fe_transformation_type="affine",
#         bound_function_dict=bound_function_dict,
#         bound_condition_dict=bound_condition_dict,
#         forcing_function=rhs,
#         output_path="tests/dump",
#         generate_mesh_plot=False,
#     )

#     # generate the dirichlet boundary data for first component
#     dirichlet_boundary_data = fespace.generate_dirichlet_boundary_data_vector(0)

#     # assert shape[0] of x with shape[0] of y in dirichlet_boundary_data
#     assert len(dirichlet_boundary_data[0]) == len(dirichlet_boundary_data[1])

#     # check the mean of the first component of the dirichlet boundary data
#     assert np.isclose(np.mean(dirichlet_boundary_data[1]), bval_1, atol=1e-6)

#     # Clean up objects
#     del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

#     # remove the temporary directory
#     shutil.rmtree("tests/dump")


# check the cell number condition on get shape function and gradient routines


def test_valid_cell_number():
    """Tests the invalid cell number condition"""

    # generate a temporary directory
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    n_cell_x = np.random.randint(2, 10)
    n_cell_y = np.random.randint(2, 10)
    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1],
        y_limits=[0, 1],
        n_cells_x=n_cell_x,
        n_cells_y=n_cell_y,
        num_boundary_points=10,
    )

    bound_function_dict = {
        1000: lambda x, y: np.ones_like(x),
        1001: lambda x, y: np.ones_like(x),
        1002: lambda x, y: np.ones_like(x),
        1003: lambda x, y: np.ones_like(x),
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: np.ones_like(x)

    # Create fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        # make fe_order a random number between 2 and 5
        fe_order=np.random.randint(2, 8),
        fe_type="legendre",
        # make quad a random number between 3 and 10
        quad_order=np.random.randint(3, 10),
        quad_type="gauss-legendre",
        fe_transformation_type="affine",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=False,
    )

    assert fespace.n_cells == n_cell_x * n_cell_y

    # Generate a random number between 0 and (n_cell_x * n_cell_y)
    for _ in range(5):
        cell_number = np.random.randint(0, n_cell_x * n_cell_y)
        # get the shape function and gradient routines
        assert fespace.get_shape_function_val(cell_number) is not None
        assert fespace.get_shape_function_grad_x(cell_number) is not None
        assert fespace.get_shape_function_grad_y(cell_number) is not None
        assert fespace.get_shape_function_grad_x_ref(cell_number) is not None
        assert fespace.get_shape_function_grad_y_ref(cell_number) is not None

    # Check the negative cell number condition
    with pytest.raises(ValueError):
        fespace.get_shape_function_val(-1)
        fespace.get_shape_function_grad_x(-1)
        fespace.get_shape_function_grad_y(-1)
        fespace.get_shape_function_grad_x_ref(-1)
        fespace.get_shape_function_grad_y_ref(-1)

    # check the cell number greater than n_cells condition
    with pytest.raises(ValueError):
        fespace.get_shape_function_val(n_cell_x * n_cell_y)
        fespace.get_shape_function_grad_x(n_cell_x * n_cell_y)
        fespace.get_shape_function_grad_y(n_cell_x * n_cell_y)
        fespace.get_shape_function_grad_x_ref(n_cell_x * n_cell_y)
        fespace.get_shape_function_grad_y_ref(n_cell_x * n_cell_y)

    # Clean up objects
    del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

    # remove the temporary directory
    shutil.rmtree("tests/dump")


def test_rhs_vector():
    """Tests the dirichlet boundary data vector"""

    # use pathlib to create a temporary directory
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=2, n_cells_y=2, num_boundary_points=10
    )

    bval_1 = np.random.rand()
    bval_2 = np.random.rand()
    fval_1 = np.random.rand()
    fval_2 = np.random.rand()
    # Vector valued boundary function
    bound_function_dict = {
        1000: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
        1001: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
        1002: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
        1003: lambda x, y: [np.ones_like(x) * bval_1, np.ones_like(x) * bval_2],
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: [np.ones_like(x) * fval_1, np.ones_like(x) * fval_2]

    # Create fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        # make fe_order a random number between 2 and 5
        fe_order=np.random.randint(2, 8),
        fe_type="legendre",
        # make quad a random number between 3 and 10
        quad_order=np.random.randint(3, 10),
        quad_type="gauss-legendre",
        fe_transformation_type="affine",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=False,
    )

    # generate the forcing term for first component
    force_1 = fespace.get_forcing_function_values_vector(0, 0)

    # assert shape
    assert force_1.shape == (fespace.fe_order**2, 1)

    # generate the forcing term for second component
    force_2 = fespace.get_forcing_function_values_vector(0, 1)

    # assert shape
    assert force_2.shape == (fespace.fe_order**2, 1)

    # remove the temporary directory
    shutil.rmtree("tests/dump")
