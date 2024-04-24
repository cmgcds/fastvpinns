# Author: Thivin Anandh. D
# Routines to check the mesh reading (both internal and external) and the boundary points.

import pytest
import numpy as np
from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D


@pytest.fixture
def geometry_2d():
    """
    Fixture that returns an instance of Geometry_2D for testing.
    """
    return Geometry_2D("quadrilateral", "internal", 10, 10, ".")


def test_read_mesh(geometry_2d):
    """
    Test case for the read_mesh method of the Geometry_2D class.
    """
    # Define test inputs
    domain = Geometry_2D("quadrilateral", "external", 10, 10, ".")
    cells, boundary_points = domain.read_mesh(
        mesh_file="tests/support_files/circle_quad.mesh",
        boundary_point_refinement_level=2,
        bd_sampling_method="uniform",
        refinement_level=0,
    )

    # Perform assertions
    assert isinstance(cells, np.ndarray)
    assert isinstance(boundary_points, dict)

    # Additional assertions can be added based on the expected behavior of the function
    # Example assertion: Check if the number of cells is correct
    assert cells.shape[0] == 1024


def test_read_mesh_invalid_file_extension(geometry_2d):
    """
    Test case for the read_mesh method of the Geometry_2D class with an invalid file extension.
    """
    # Define test inputs
    domain = Geometry_2D("quadrilateral", "external", 10, 10, ".")

    # Expect a ValueError when the file extension is not .mesh
    with pytest.raises(ValueError, match="Mesh file should be in .mesh format."):
        cells, boundary_points = domain.read_mesh(
            mesh_file="tests/support_files/circle_quad.txt",
            boundary_point_refinement_level=2,
            bd_sampling_method="uniform",
            refinement_level=0,
        )


def test_read_mesh_invalid_mesh_type(geometry_2d):
    """
    Test case for the read_mesh method of the Geometry_2D class with an invalid mesh type.
    """
    # Define test inputs
    domain = Geometry_2D("invalid_type", "external", 10, 10, ".")

    # Expect a ValueError when the mesh type is not quadrilateral or triangle
    with pytest.raises(ValueError, match="Mesh type should be quadrilateral only"):
        cells, boundary_points = domain.read_mesh(
            mesh_file="tests/support_files/circle_quad.mesh",
            boundary_point_refinement_level=2,
            bd_sampling_method="uniform",
            refinement_level=0,
        )


def test_read_mesh_invalid_sampling_method(geometry_2d):
    """
    Test case for the read_mesh method of the Geometry_2D class with an invalid sampling method.
    """
    # Define test inputs
    domain = Geometry_2D("quadrilateral", "external", 10, 10, ".")

    # Expect a ValueError when the sampling method is not uniform or lhs
    with pytest.raises(ValueError, match="Sampling method should be either uniform or lhs."):
        cells, boundary_points = domain.read_mesh(
            mesh_file="tests/support_files/circle_quad.mesh",
            boundary_point_refinement_level=2,
            bd_sampling_method="invalid_method",
            refinement_level=0,
        )


def test_generate_quad_mesh_internal():
    """
    Test case for the generate_quad_mesh_internal method of the Geometry_2D class.
    """
    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, ".")

    # Define test inputs
    x_limits = [0, 1]
    y_limits = [0, 1]
    n_cells_x = 4
    n_cells_y = 4
    num_boundary_points = 100

    # Call the method with the test inputs
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=x_limits,
        y_limits=y_limits,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        num_boundary_points=num_boundary_points,
    )

    # Perform assertions

    # Check if the output types are correct
    assert isinstance(cells, np.ndarray)
    assert isinstance(boundary_points, dict)

    # Check if the number of cells is correct
    assert cells.shape[0] == n_cells_x * n_cells_y
