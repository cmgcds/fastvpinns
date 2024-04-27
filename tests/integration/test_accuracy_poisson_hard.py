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
from fastvpinns.model.model_hard import DenseModel_Hard
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
def poisson_2d_static_learning_rate():
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


def test_poisson_2d_hard_accuracy_internal(poisson_test_data, poisson_2d_static_learning_rate):
    """
    Test the accuracy of the cd2d solution for on internal generated grid
    Args:
        poisson_test_data (tuple): Test data for the cd2d equation
    """

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
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=4, n_cells_y=4, num_boundary_points=500
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
    lr_dict = poisson_2d_static_learning_rate

    @tf.function
    def apply_hard_boundary_constraints(inputs, x):
        """This method applies hard boundary constraints to the model.
        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :param x: Output tensor from the model
        :type x: tf.Tensor
        :return: Output tensor with hard boundary constraints
        :rtype: tf.Tensor
        """
        ansatz = (
            tf.tanh(4.0 * np.pi * inputs[:, 0:1])
            * tf.tanh(4.0 * np.pi * inputs[:, 1:2])
            * tf.tanh(4.0 * np.pi * (inputs[:, 0:1] - 1.0))
            * tf.tanh(4.0 * np.pi * (inputs[:, 1:2] - 1.0))
        )
        ansatz = tf.cast(ansatz, tf.float32)
        return ansatz * x

    # generate a model
    model = DenseModel_Hard(
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
        tensor_dtype=tf.float32,
        use_attention=False,
        hessian=False,
        hard_constraint_function=apply_hard_boundary_constraints,
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

    print(f"l2_error = {l2_error}, l1_error = {l1_error}")
    assert l2_error < 6e-2 and l1_error < 6e-2
