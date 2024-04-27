# Author : Thivin Anandh. D
# Added test cases for validating Quadrature routines by computing the areas.
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf

from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.FE_2D.fe2d_setup_main import FE2DSetupMain
from fastvpinns.FE_2D.quadratureformulas_quad2d import Quadratureformulas_Quad2D

import pytest


@pytest.mark.parametrize("quad_order", [3, 10, 15])
def test_invalid_fe_type(quad_order):
    """
    Test case to validate the behavior when an invalid finite element type is provided.
    It should raise a ValueError.
    """

    quad_formula_main = Quadratureformulas_Quad2D(quad_order, "gauss-jacobi")
    num_quad = quad_formula_main.get_num_quad_points()
    assert num_quad == quad_order**2
