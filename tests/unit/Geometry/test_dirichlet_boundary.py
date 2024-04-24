# Author : Thivin Anandh. D
# Added test cases for validating Dirichlet boundary routines
# Routines, provide a value to the boundary points and check if the value is set correctly.

import pytest
import numpy as np
from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D


def test_dirichlet_boundary_internal():
    """
    Test case for validating Dirichlet boundary routines.
    Routines provide a value to the boundary points and check if the value is set correctly.
    """

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, ".")
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=4, n_cells_y=4, num_boundary_points=100
    )

    values = [np.random.rand() for _ in range(4)]

    bound_function_dict = {
        1000: lambda x, y: np.ones_like(x) * values[0],
        1001: lambda x, y: np.ones_like(x) * values[1],
        1002: lambda x, y: np.ones_like(x) * values[2],
        1003: lambda x, y: np.ones_like(x) * values[3],
    }
    bound_condition_dict = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }
    rhs = lambda x, y: np.ones_like(x)

    # obtain the boundary dict for each of the component and compute their means
    for bound_id in bound_condition_dict.keys():
        mean_val = np.mean(
            bound_function_dict[bound_id](
                boundary_points[bound_id][:, 0], boundary_points[bound_id][:, 1]
            )
        )
        assert np.isclose(mean_val, values[bound_id - 1000])


def test_dirichlet_boundary_external():
    """
    Test case for validating Dirichlet boundary routines.
    Routines provide a value to the boundary points and check if the value is set correctly.
    """

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "external", 10, 10, ".")
    cells, boundary_points = domain.read_mesh(
        mesh_file="tests/support_files/circle_quad.mesh",
        boundary_point_refinement_level=2,
        bd_sampling_method="uniform",
        refinement_level=0,
    )

    val = np.random.rand()
    bound_function_dict = {1000: lambda x, y: np.ones_like(x) * val}

    bound_condition_dict = {1000: "dirichlet"}
    rhs = lambda x, y: np.ones_like(x)

    # obtain the boundary dict for each of the component and compute their means
    for bound_id in bound_condition_dict.keys():
        mean_val = np.mean(
            bound_function_dict[bound_id](
                boundary_points[bound_id][:, 0], boundary_points[bound_id][:, 1]
            )
        )
        assert np.isclose(mean_val, val)
