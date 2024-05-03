# Author: Thivin Anandh D
# Purpose : Check the accuracy of the cd2d solution for Internal Meshes
# Date : 23/Apr/2024

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
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


@pytest.fixture
def cd2d_test_data_circle():
    """
    Fixture function that provides test data for the cd2d equation with circular boundary.
    Returns:
        Tuple: boundary_functions, boundary_conditions, bilinear_params, forcing_function, exact_solution
    """
    omega = 4.0 * np.pi

    circle_boundary = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)

    boundary_functions = {1000: circle_boundary}

    boundary_conditions = {1000: "dirichlet"}

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


def test_cd2d_accuracy_internal(cd2d_test_data_internal):
    """
    Test the accuracy of the cd2d solution for on internal generated grid
    Args:
        cd2d_test_data_internal (tuple): Test data for the cd2d equation
    """

    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        cd2d_test_data_internal
    )

    output_folder = "tests/test_dump"

    # use pathlib to create the directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # generate a internal mesh
    domain = Geometry_2D("quadrilateral", "internal", 100, 100, output_folder)
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=8, n_cells_y=8, num_boundary_points=500
    )

    # create the fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=4,
        fe_type="jacobi",
        quad_order=5,
        quad_type="gauss-jacobi",
        fe_transformation_type="bilinear",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )
    # create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=tf.float32)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # get the learning rate dictionary
    lr_dict = cd2d_learning_rate_static_data()

    # generate a model
    model = DenseModel(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_cd2d,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(6000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    print(f"l2_error = {l2_error}, l1_error = {l1_error}")
    assert l2_error < 6e-2 and l1_error < 6e-2


def test_cd2d_accuracy_external(cd2d_test_data_circle):
    """
    Test function for accuracy of the cd2d equation solver.
    Args:
        cd2d_test_data_circle (tuple): Test data for the cd2d equation
    """

    # Obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        cd2d_test_data_circle
    )

    output_folder = "tests/test_dump"

    # Use pathlib to create the directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Generate an internal mesh
    domain = Geometry_2D("quadrilateral", "external", 100, 100, output_folder)
    cells, boundary_points = domain.read_mesh(
        "tests/support_files/circle_quad.mesh", 2, "uniform", refinement_level=1
    )

    # Create the fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=3,
        fe_type="jacobi",
        quad_order=4,
        quad_type="gauss-jacobi",
        fe_transformation_type="bilinear",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )

    # Create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=tf.float32)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # Get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # Get the learning rate dictionary
    lr_dict = cd2d_learning_rate_static_data()

    # Generate a model
    model = DenseModel(
        layer_dims=[2, 50, 50, 50, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_cd2d,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # Train the model
    for epoch in range(6000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # Check the L2 error, L1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    assert l2_error < 6e-2 and l1_error < 6e-2
