# Author: Thivin Anandh D
# Purpose : Check the accuracy of the cd2d solution for Internal Meshes
# Date : 23/Apr/2024

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf
import pandas as pd

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model_inverse_domain import DenseModel_Inverse_Domain
from fastvpinns.physics.cd2d_inverse_domain import pde_loss_cd2d_inverse_domain
from fastvpinns.utils.compute_utils import compute_errors_combined


@pytest.fixture
def cd2d_learning_rate_static_data():
    """
    Generate the learning rate dictionary for the cd2d equation.
    """
    initial_learning_rate = 0.003
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

    circle_boundary = lambda x, y: np.ones_like(x) * 0.0

    boundary_functions = {1000: circle_boundary}

    boundary_conditions = {1000: "dirichlet"}

    bilinear_params = lambda: {"eps": 0.1, "b_x": 1, "b_y": 0.0, "c": 0.0}

    def rhs(x, y):
        return 10.0 * np.ones_like(x)

    forcing_function = rhs

    exact_solution = lambda x, y: np.tanh(np.pi * x) * np.cos(2 * np.pi * y)

    def get_inverse_params_actual_dict(x, y):
        """
        This function will return a dictionary of inverse parameters
        """
        # Initial Guess
        eps = 0.5 * (np.sin(x) + np.cos(y))
        return {"eps": eps}

    get_inverse_params_actual_dict_fn = get_inverse_params_actual_dict
    return (
        boundary_functions,
        boundary_conditions,
        bilinear_params,
        forcing_function,
        exact_solution,
        get_inverse_params_actual_dict_fn,
    )


def test_cd2d_accuracy_external(cd2d_test_data_circle, cd2d_learning_rate_static_data):
    """
    Test function for accuracy of the cd2d equation solver.
    Args:
        cd2d_test_data_circle (tuple): Test data for the cd2d equation
    """

    # Obtain the test data
    (
        bound_function_dict,
        bound_condition_dict,
        bilinear_params,
        rhs,
        exact_solution,
        get_inverse_params_actual_dict_fn,
    ) = cd2d_test_data_circle

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
        quad_order=5,
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
    lr_dict = cd2d_learning_rate_static_data

    points, sensor_values = datahandler.get_sensor_data(
        exact_solution,
        num_sensor_points=500,
        mesh_type="external",
        file_name="tests/support_files/fem_output_circle2.csv",
    )

    # Generate a model
    model = DenseModel_Inverse_Domain(
        layer_dims=[2, 30, 30, 30, 2],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_cd2d_inverse_domain,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        sensor_list=[points, sensor_values],
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()

    # read the exact solution from the csv file
    y_exact = pd.read_csv("tests/support_files/fem_output_circle2.csv", header=None)
    y_exact = y_exact.iloc[:, 2].values.reshape(-1)

    points, sensor_values = datahandler.get_sensor_data(
        exact_solution,
        num_sensor_points=500,
        mesh_type="external",
        file_name="tests/support_files/fem_output_circle2.csv",
    )

    target_inverse_params_dict = get_inverse_params_actual_dict_fn(
        test_points[:, 0], test_points[:, 1]
    )
    # get actual Epsilon
    actual_epsilon = np.array(target_inverse_params_dict["eps"]).reshape(-1)

    # Train the model
    for epoch in range(4000):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # Check the L2 error, L1 error of the model
    y_pred_actual = model(test_points).numpy()
    y_pred = y_pred_actual[:, 0].reshape(-1)  # First column is the solution'

    inverse_pred = y_pred_actual[:, 1].reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )
    # get Errors for inverse parameters
    (
        l2_error_inverse,
        linf_error_inverse,
        l2_error_relative_inverse,
        linf_error_relative_inverse,
        l1_error_inverse,
        l1_error_relative_inverse,
    ) = compute_errors_combined(actual_epsilon, inverse_pred)

    assert l1_error < 8.3e-2 and l1_error_inverse < 8.3e-2
