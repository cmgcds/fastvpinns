# Author : Thivin Anandh. D
# Added test cases for validating Quadrature routines by computing the areas.
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf
import shutil
from pathlib import Path
from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
import pytest


@pytest.mark.parametrize("coord", [[0, 1, 0, 1], [-1, 1, -1, 1], [-3, 4, -2, 5], [2, 5, -2, 3]])
@pytest.mark.parametrize("quad_type", ["gauss-legendre", "gauss-jacobi"])
@pytest.mark.parametrize("transformation", ["affine", "bilinear"])
def test_quadrature_uniform(coord, quad_type, transformation):
    """Tests Quadrature routines for different quadrature types and transformations by calculating Area"""

    # generate a temp directory called tests/dump using pathlib
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/dump")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[coord[0], coord[1]],
        y_limits=[coord[2], coord[3]],
        n_cells_x=4,
        n_cells_y=4,
        num_boundary_points=100,
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
        fe_type="jacobi",
        # make quad a random number between 3 and 10
        quad_order=np.random.randint(3, 10),
        quad_type=quad_type,
        fe_transformation_type=transformation,
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=False,
    )

    # Loop over cells, calculate integral, and assert
    int_sum = 0
    for fe_cell in fespace.fe_cell:
        int_sum += np.sum(fe_cell.mult)

    actual_area = (coord[1] - coord[0]) * (coord[3] - coord[2])

    assert np.isclose(
        int_sum, actual_area, rtol=1e-4
    ), f"Failed for quad_type: {quad_type}, transformation: {transformation}"
    print(f"Test passed for quad_type: {quad_type}, transformation: {transformation}")

    # Clean up objects
    del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

    # remove the temp directory
    shutil.rmtree("tests/dump")


@pytest.mark.parametrize("quad_type", ["gauss-legendre", "gauss-jacobi"])
@pytest.mark.parametrize("transformation", ["affine", "bilinear"])
def test_quadrature_complex(quad_type, transformation):
    """Tests Quadrature routines for different quadrature types and transformations by calculating Area"""

    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "external", 10, 10, "tests/dump")

    cells, boundary_points = domain.read_mesh(
        mesh_file="tests/support_files/circle_quad.mesh",
        boundary_point_refinement_level=2,
        bd_sampling_method="uniform",
        refinement_level=0,
    )

    bound_function_dict = {1000: lambda x, y: np.ones_like(x)}
    bound_condition_dict = {1000: "dirichlet"}

    rhs = lambda x, y: np.ones_like(x)

    # Create fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=2,
        fe_type="jacobi",
        quad_order=3,
        quad_type=quad_type,
        fe_transformation_type=transformation,
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path="tests/dump",
        generate_mesh_plot=False,
    )

    # Loop over cells, calculate integral, and assert
    int_sum = 0
    for fe_cell in fespace.fe_cell:
        int_sum += np.sum(fe_cell.mult)

    actual_area = np.pi

    assert np.isclose(
        int_sum, actual_area, rtol=1e-2
    ), f"Failed for quad_type: {quad_type}, transformation: {transformation}"
    print(f"Test passed for quad_type: {quad_type}, transformation: {transformation}")

    # Clean up objects
    del domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, fespace

    # remove the temp directory
    shutil.rmtree("tests/dump")
