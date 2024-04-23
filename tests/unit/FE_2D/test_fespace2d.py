# Author : Thivin Anandh. D
# Added test cases for validating Quadrature routines by computing the areas. 
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
import pytest

@pytest.fixture()
def test_quadrature_uniform(request):
    """Tests Quadrature routines for different quadrature types and transformations by calculating Area"""

    coord = request.param
    quad_type = "gauss"
    transformation = "affine"

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, ".")
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

    return domain, cells, boundary_points, bound_function_dict, bound_condition_dict, rhs, quad_type, transformation
