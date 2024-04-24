# Author : Thivin Anandh. D
# Added test cases for validating Datahandler routines
# The test cases are parametrized for different quadrature types and transformations.

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model import DenseModel
from fastvpinns.physics.cd2d import pde_loss_cd2d
from fastvpinns.utils.compute_utils import compute_errors_combined


@pytest.fixture
def cd2d_test_data_internal():
    """
    Generate test data for the cd2d equation.
    """
    omega = 4.0 * np.pi
    left_boundary = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)
    right_boundary = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)
    bottom_boundary = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)
    top_boundary = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)
    boundary_functions = {
        1000: bottom_boundary,
        1001: right_boundary,
        1002: top_boundary,
        1003: left_boundary,
    }

    boundary_conditions = {
        1000: "dirichlet",
        1001: "dirichlet",
        1002: "dirichlet",
        1003: "dirichlet",
    }

    bilinear_params = lambda: {"eps": 1.0, "b_x": 0.2, "b_y": -0.1, "c": 0.0}

    def rhs(x, y):
        result = 0.2 * np.sin(2 * np.pi * y) * np.sinh(2 * np.pi * x)
        result += 4.0 * np.pi * np.cos(2 * np.pi * y) * np.sinh(2 * np.pi * x)
        result += 4.0 * np.pi * np.cos(2 * np.pi * y) * np.tanh(np.pi * x)
        result += 0.4 * np.cos(2 * np.pi * y)
        result = (np.pi * result) / (np.cosh(2 * np.pi * x) + 1)
        return result

    forcing_function = rhs

    exact_solution = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)

    return (
        boundary_functions,
        boundary_conditions,
        bilinear_params,
        forcing_function,
        exact_solution,
    )


def cd2d_learning_rate_static_data():
    """
    Generate the learning rate dictionary for the cd2d equation.
    """
    initial_learning_rate = 0.001
    use_lr_scheduler = False
    decay_steps = 1000
    decay_rate = 0.99
    staircase = False

    learning_rate_dict = {}
    learning_rate_dict["initial_learning_rate"] = initial_learning_rate
    learning_rate_dict["use_lr_scheduler"] = use_lr_scheduler
    learning_rate_dict["decay_steps"] = decay_steps
    learning_rate_dict["decay_rate"] = decay_rate
    learning_rate_dict["staircase"] = staircase

    return learning_rate_dict


# This is a module-scoped fixture that stores the results of the setup
@pytest.fixture(scope="module")
def setup_results():
    return {}


@pytest.mark.parametrize("quad_order", [3, 5, 7])
@pytest.mark.parametrize("fe_order", [2, 4, 6])
@pytest.mark.parametrize("cell_dimensions", [[4, 4], [2, 4], [6, 5]])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_setup(
    cd2d_test_data_internal, quad_order, fe_order, cell_dimensions, precision, setup_results
):
    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        cd2d_test_data_internal
    )
    n_cells_x = cell_dimensions[0]
    n_cells_y = cell_dimensions[1]
    n_cells = n_cells_x * n_cells_y
    output_folder = "tests/test_dump"

    if precision == "float32":
        precision = tf.float32
    elif precision == "float64":
        precision = tf.float64

    # use pathlib to create the directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # generate a internal mesh
    domain = Geometry_2D("quadrilateral", "internal", 89, 89, output_folder)
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1],
        y_limits=[0, 1],
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        num_boundary_points=500,
    )

    # create the fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=fe_order,
        fe_type="jacobi",
        quad_order=quad_order,
        quad_type="gauss-jacobi",
        fe_transformation_type="bilinear",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )
    # create the data handler
    datahandler = DataHandler2D(fespace, domain, dtype=precision)

    # obtain all variables of Datahandler
    x_pde_list = datahandler.x_pde_list
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()
    shape_val_mat_list = datahandler.shape_val_mat_list
    grad_x_mat_list = datahandler.grad_x_mat_list
    grad_y_mat_list = datahandler.grad_y_mat_list
    forcing_function_list = datahandler.forcing_function_list

    test_points = datahandler.get_test_points()

    setup_results[(quad_order, fe_order, tuple(cell_dimensions), precision)] = (
        x_pde_list,
        train_dirichlet_input,
        train_dirichlet_output,
        shape_val_mat_list,
        grad_x_mat_list,
        grad_y_mat_list,
        forcing_function_list,
        test_points,
    )


def test_x_pde_list(setup_results):
    """
    Test function for checking the properties of x_pde_list.

    :param setup_results: A dictionary containing setup results.
    """
    for key, value in setup_results.items():
        x_pde_list = value[0]

        quad_order = key[0]
        fe_order = key[1]
        cell_dimensions = key[2]
        n_cell = cell_dimensions[0] * cell_dimensions[1]
        precision = key[3]

        # check if the x_pde_list is a tensor
        assert isinstance(x_pde_list, tf.Tensor)
        # check precision
        assert x_pde_list.dtype == precision
        # check shape
        assert x_pde_list.shape == (n_cell * quad_order**2, 2)


def test_dirichlet_inputs(setup_results):
    """
    Test function for checking the properties of dirichlet inputs.

    :param setup_results: A dictionary containing setup results.
    """
    for key, value in setup_results.items():
        train_dirichlet_input = value[1]
        train_dirichlet_output = value[2]

        quad_order = key[0]
        fe_order = key[1]
        cell_dimensions = key[2]
        n_cell = cell_dimensions[0] * cell_dimensions[1]
        precision = key[3]

        # check if the x_pde_list is a tensor
        assert isinstance(train_dirichlet_input, tf.Tensor)
        assert isinstance(train_dirichlet_output, tf.Tensor)
        # check precision
        assert train_dirichlet_input.dtype == precision
        assert train_dirichlet_output.dtype == precision

        # check first dimensions of input and output
        assert train_dirichlet_input.shape[0] == train_dirichlet_output.shape[0]


def test_shape_tensors(setup_results):
    """
    Test function for checking the properties of Shape function and gradient matrices.

    :param setup_results: A dictionary containing setup results.
    """
    for key, value in setup_results.items():
        shape_val_mat_list = value[3]
        grad_x_mat_list = value[4]
        grad_y_mat_list = value[5]

        quad_order = key[0]
        fe_order = key[1]
        cell_dimensions = key[2]
        n_cell = cell_dimensions[0] * cell_dimensions[1]
        precision = key[3]

        # check if the x_pde_list is a tensor
        assert isinstance(shape_val_mat_list, tf.Tensor)
        assert isinstance(grad_x_mat_list, tf.Tensor)
        assert isinstance(grad_y_mat_list, tf.Tensor)
        # check precision
        assert shape_val_mat_list.dtype == precision
        assert grad_x_mat_list.dtype == precision
        assert grad_y_mat_list.dtype == precision
        # check shape
        assert shape_val_mat_list.shape == (n_cell, fe_order**2, quad_order**2)
        assert grad_x_mat_list.shape == (n_cell, fe_order**2, quad_order**2)
        assert grad_y_mat_list.shape == (n_cell, fe_order**2, quad_order**2)


def test_forcing_function_list(setup_results):
    """
    Test function for checking the properties of forcing function list.

    :param setup_results: A dictionary containing setup results.
    """
    for key, value in setup_results.items():
        forcing_function_list = value[6]

        quad_order = key[0]
        fe_order = key[1]
        cell_dimensions = key[2]
        n_cell = cell_dimensions[0] * cell_dimensions[1]
        precision = key[3]

        # check if the x_pde_list is a tensor
        assert isinstance(forcing_function_list, tf.Tensor)
        # check precision
        assert forcing_function_list.dtype == precision
        # check shape
        assert forcing_function_list.shape == (fe_order**2, n_cell)


def test_num_test_points(setup_results):
    """
    Test function for checking the number of test points.

    :param setup_results: A dictionary containing setup results.
    """
    for key, value in setup_results.items():
        test_points = value[7]
        quad_order = key[0]
        fe_order = key[1]
        cell_dimensions = key[2]
        n_cell = cell_dimensions[0] * cell_dimensions[1]
        precision = key[3]

        # check if the x_pde_list is a tensor
        assert isinstance(test_points, tf.Tensor)
        # check precision
        assert test_points.dtype == precision
        # check shape
        assert test_points.shape == (89 * 89, 2)
