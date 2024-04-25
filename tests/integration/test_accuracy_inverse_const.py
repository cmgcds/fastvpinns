# Author: Thivin Anandh D
# Purpose : Check the accuracy of the cd2d solution for Internal Meshes
# Date : 23/Apr/2024

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE_2D.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model_inverse import DenseModel_Inverse
from fastvpinns.physics.poisson2d_inverse import pde_loss_poisson_inverse
from fastvpinns.utils.compute_utils import compute_errors_combined


@pytest.fixture
def poisson2d_inverse_const_test_data():
    """
    Generate test data for the cd2d equation.
    """
    EPS = 0.3
    left_boundary = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    right_boundary = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    bottom_boundary = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    top_boundary = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
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

    bilinear_params = lambda: {"eps": 2}

    def rhs(x, y):
        """
        This function will return the value of the rhs at a given point
        """
        EPS = 0.3  # actual epsilon
        X = x
        Y = y
        eps = EPS

        return (
            -EPS
            * (
                40.0 * X * eps * (np.tanh(X) ** 2 - 1) * np.sin(X)
                - 40.0 * X * eps * np.cos(X) * np.tanh(X)
                + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X)
                + 20 * (np.tanh(X) ** 2 - 1) * np.sin(X) * np.tanh(X)
                - 20 * (np.tanh(X) ** 2 - 1) * np.cos(X)
                - 10 * np.sin(X) * np.tanh(X)
            )
            * np.exp(-1.0 * X**2 * eps)
        )

    forcing_function = rhs

    exact_solution = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10

    return (
        boundary_functions,
        boundary_conditions,
        bilinear_params,
        forcing_function,
        exact_solution,
    )


@pytest.fixture
def poisson2d_inverse_const_test_data_circle():
    """
    Generate test data for the cd2d equation.
    """
    EPS = 0.3
    circle_boundary = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    boundary_functions = {1000: circle_boundary}

    boundary_conditions = {1000: "dirichlet"}

    bilinear_params = lambda: {"eps": 2}

    def rhs(x, y):
        """
        This function will return the value of the rhs at a given point
        """
        EPS = 0.3  # actual epsilon
        X = x
        Y = y
        eps = EPS

        return (
            -EPS
            * (
                40.0 * X * eps * (np.tanh(X) ** 2 - 1) * np.sin(X)
                - 40.0 * X * eps * np.cos(X) * np.tanh(X)
                + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X)
                + 20 * (np.tanh(X) ** 2 - 1) * np.sin(X) * np.tanh(X)
                - 20 * (np.tanh(X) ** 2 - 1) * np.cos(X)
                - 10 * np.sin(X) * np.tanh(X)
            )
            * np.exp(-1.0 * X**2 * eps)
        )

    forcing_function = rhs

    exact_solution = lambda x, y: np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10

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


def test_inverse_constant_poisson2d_internal(poisson2d_inverse_const_test_data):
    """
    Test the accuracy of the cd2d solution for on internal generated grid
    Args:
        poisson2d_inverse_const_test_data (tuple): Test data for the cd2d equation
    """

    def get_inverse_params_dict():
        """
        Get the inverse parameters dictionary
        """
        return {"eps": 2}  # initial guess

    # obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson2d_inverse_const_test_data
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

    points, sensor_values = datahandler.get_sensor_data(
        exact_solution, num_sensor_points=50, mesh_type="internal"
    )

    inverse_params_dict = datahandler.get_inverse_params(get_inverse_params_dict)

    # generate a model
    model = DenseModel_Inverse(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson_inverse,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        sensor_list=[points, sensor_values],
        inverse_params_dict=inverse_params_dict,
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(4000):
        loss = model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )
    # obtain the inverse parameters
    inverse_params = loss['inverse_params']['eps'].numpy()

    # print the inverse parameters
    print(f"Inverse Parameters: {inverse_params}")

    print(f"l2_error = {l2_error}, l1_error = {l1_error}")
    assert l2_error < 6e-2 and l1_error < 6e-2
    assert abs(inverse_params - 0.3) < 5e-2


def test_inverse_constant_poisson2d_external(poisson2d_inverse_const_test_data_circle):
    """
    Test function for accuracy of the cd2d equation solver.
    Args:
        cd2d_test_data_circle (tuple): Test data for the cd2d equation
    """

    def get_inverse_params_dict():
        """
        Get the inverse parameters dictionary
        """
        return {"eps": 3}  # initial guess

    # Obtain the test data
    bound_function_dict, bound_condition_dict, bilinear_params, rhs, exact_solution = (
        poisson2d_inverse_const_test_data_circle
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

    # create the data handler
    datahandler = DataHandler2D(fespace, output_folder, dtype=tf.float32)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(bilinear_params)

    # get the learning rate dictionary
    lr_dict = cd2d_learning_rate_static_data()

    points, sensor_values = datahandler.get_sensor_data(
        exact_solution,
        num_sensor_points=100,
        mesh_type="external",
        file_name="tests/support_files/const_inverse_poisson_solution.txt",
    )

    inverse_params_dict = datahandler.get_inverse_params(get_inverse_params_dict)

    # generate a model
    model = DenseModel_Inverse(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=lr_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson_inverse,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        sensor_list=[points, sensor_values],
        inverse_params_dict=inverse_params_dict,
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
    )

    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # train the model
    for epoch in range(4000):
        loss = model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # check the l2 error l1 error of the model
    y_pred = model(test_points).numpy()
    y_pred = y_pred.reshape(-1)

    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )
    # obtain the inverse parameters
    inverse_params = loss['inverse_params']['eps'].numpy()

    # print the inverse parameters
    print(f"Inverse Parameters: {inverse_params}")

    print(f"l2_error = {l2_error}, l1_error = {l1_error}")
    assert l2_error < 6e-2 and l1_error < 6e-2
    assert abs(inverse_params - 0.3) < 5e-2
