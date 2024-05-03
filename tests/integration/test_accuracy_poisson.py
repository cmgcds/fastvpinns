# Author: Thivin Anandh D
# Purpose : Check the accuracy of the poisson solution for
#          - Different transformations and precision
#         - Different fe_types and quad_types
#        - Different activation functions and learning rate types
#         For poisson, all hyperparameter combinations should give an L2 error < 6e-2
# Date : 23/Apr/2024

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model import DenseModel
from fastvpinns.physics.poisson2d import pde_loss_poisson
from fastvpinns.utils.compute_utils import compute_errors_combined


@pytest.fixture
def poisson_test_data():
    """
    Generate test data for the Poisson equation.
    """
    omega = 4.0 * np.pi
    left_boundary = lambda x, y: np.zeros_like(x)
    right_boundary = lambda x, y: np.zeros_like(x)
    bottom_boundary = lambda x, y: np.zeros_like(x)
    top_boundary = lambda x, y: np.zeros_like(x)
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

    bilinear_params = lambda: {"eps": 1.0}

    forcing_function = lambda x, y: -2.0 * (omega**2) * np.sin(omega * x) * np.sin(omega * y)

    exact_solution = lambda x, y: -1.0 * np.sin(omega * x) * np.sin(omega * y)

    return (
        boundary_functions,
        boundary_conditions,
        bilinear_params,
        forcing_function,
        exact_solution,
    )


@pytest.fixture
def poisson_test_data_circle():
    """
    Fixture function that provides test data for the Poisson equation with circular boundary.
    Returns:
        Tuple: boundary_functions, boundary_conditions, bilinear_params, forcing_function, exact_solution
    """
    omega = 4.0 * np.pi

    circle_boundary = lambda x, y: -1.0 * np.sin(omega * x) * np.sin(omega * y)

    boundary_functions = {1000: circle_boundary}

    boundary_conditions = {1000: "dirichlet"}

    bilinear_params = lambda: {"eps": 1.0}

    forcing_function = lambda x, y: -2.0 * (omega**2) * np.sin(omega * x) * np.sin(omega * y)

    exact_solution = lambda x, y: -1.0 * np.sin(omega * x) * np.sin(omega * y)

    return (
        boundary_functions,
        boundary_conditions,
        bilinear_params,
        forcing_function,
        exact_solution,
    )


def poisson_learning_rate_static_data():
    """
    Generate the learning rate dictionary for the Poisson equation.
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


@pytest.mark.parametrize("transformation", ["affine", "bilinear"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_poisson_accuracy_transformation_precision(poisson_test_data, transformation, precision):
    """
    Test the accuracy of the Poisson solution for different transformations and precision.
    """
    if precision == "float32":
        precision = tf.float32
    else:
        precision = tf.float64

    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson_test_data
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
        fe_transformation_type=transformation,
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )

    # create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=precision)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # get the learning rate dictionary
    lr_dict = poisson_learning_rate_static_data()

    # generate a model
    model = DenseModel(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=precision,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(5000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    assert l2_error < 6e-2 and l1_error < 6e-2


@pytest.mark.parametrize("fe_type", ["jacobi", "legendre_special", "chebyshev_2"])
@pytest.mark.parametrize("quad_type", ["gauss-jacobi", "gauss-legendre"])
def test_poisson_accuracy_fetype_quadtype(poisson_test_data, fe_type, quad_type):
    """
    Test the accuracy of the Poisson solution for different fe_types and quad_types.

    Parameters:
    - poisson_test_data: The test data for the Poisson equation.
    - fe_type: The type of finite element basis functions.
    - quad_type: The type of quadrature rule.
    """
    precision = tf.float32

    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson_test_data
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
        fe_type=fe_type,
        quad_order=5,
        quad_type=quad_type,
        fe_transformation_type="bilinear",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )

    # create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=precision)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # get the learning rate dictionary
    lr_dict = poisson_learning_rate_static_data()

    # generate a model
    model = DenseModel(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=precision,
        use_attention=False,
        activation="tanh",
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(5000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    assert l2_error < 6e-2 and l1_error < 6e-2


@pytest.mark.parametrize("activation", ["tanh", "swish", "gelu"])
@pytest.mark.parametrize("lr_type", ["adaptive"])
def test_poisson_accuracy_activation_lr(poisson_test_data, activation, lr_type):

    precision = tf.float32

    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson_test_data
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
    datahandler = DataHandler2D(fespace, output_folder, dtype=precision)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # get the learning rate dictionary
    lr_dict = poisson_learning_rate_static_data()

    if lr_type == "adaptive":
        lr_dict["use_lr_scheduler"] = True

    # generate a model
    model = DenseModel(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=precision,
        use_attention=False,
        activation=activation,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(6500):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    assert l2_error < 8.2e-2 and l1_error < 8.2e-2

    if lr_type == "adaptive":
        current_learning_rate = tf.keras.backend.get_value(model.optimizer.lr)
        assert current_learning_rate < 0.001


def poisson_learning_rate_static_data():
    """
    Function that returns the learning rate dictionary for the Poisson equation.
    Returns:
        dict: Learning rate dictionary
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


@pytest.mark.parametrize("transformation", ["bilinear"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_poisson_accuracy_complex(poisson_test_data_circle, transformation, precision):
    """
    Test function for accuracy of the Poisson equation solver.
    Args:
        poisson_test_data_circle (tuple): Test data for the Poisson equation
        transformation (str): Transformation type
        precision (str): Precision type
    """
    if precision == "float32":
        precision = tf.float32
    else:
        precision = tf.float64

    # Obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson_test_data_circle
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
        fe_transformation_type=transformation,
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=output_folder,
        generate_mesh_plot=False,
    )

    # Create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=precision)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # Get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # Get the learning rate dictionary
    lr_dict = poisson_learning_rate_static_data()

    # Generate a model
    model = DenseModel(
        layer_dims=[2, 50, 50, 50, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=precision,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # Train the model
    for epoch in range(5000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # Check the L2 error, L1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    assert l2_error < 6e-2 and l1_error < 6e-2
