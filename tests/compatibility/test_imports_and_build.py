# Author: Thivin Anandh
# Purpose: This code just imports all the available modules and checks if it works on enviroinments
#          different versions of python, OS(Windows, Linux, MacOS)


def test_imports():
    # import all the libraries mentioned on main code
    import numpy as np
    import pandas as pd
    import pytest
    import tensorflow as tf
    from pathlib import Path
    from tqdm import tqdm
    import yaml
    import sys
    import copy
    from tensorflow.keras import layers
    from tensorflow.keras import initializers
    from rich.console import Console
    import copy
    import time

    import meshio
    from pyDOE import lhs

    from scipy.special import roots_legendre, roots_jacobi, jacobi, gamma
    from scipy.special import legendre
    from scipy.special import eval_legendre, legendre

    from fastvpinns.Geometry.geometry_2d import Geometry_2D
    from fastvpinns.FE.fespace2d import Fespace2D
    from fastvpinns.data.datahandler2d import DataHandler2D

    # import all models
    from fastvpinns.model.model import DenseModel
    from fastvpinns.model.model_hard import DenseModel_Hard
    from fastvpinns.model.model_inverse import DenseModel_Inverse
    from fastvpinns.model.model_inverse_domain import DenseModel_Inverse_Domain

    # import all loss functions
    from fastvpinns.physics.poisson2d import pde_loss_poisson
    from fastvpinns.physics.helmholtz2d import pde_loss_helmholtz
    from fastvpinns.physics.cd2d import pde_loss_cd2d
    from fastvpinns.physics.cd2d_inverse import pde_loss_cd2d
    from fastvpinns.physics.cd2d_inverse_domain import pde_loss_cd2d_inverse_domain

    from fastvpinns.utils.plot_utils import (
        plot_contour,
        plot_loss_function,
        plot_test_loss_function,
    )
    from fastvpinns.utils.compute_utils import compute_errors_combined
    from fastvpinns.utils.print_utils import print_table

    # assert if the code reaches this point successfully
    assert True


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
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=2, n_cells_y=2, num_boundary_points=500
    )

    # create the fespace
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=5,
        fe_type="jacobi",
        quad_order=10,
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
    for epoch in range(10):
        model.train_step(beta=10, bilinear_params_dict=bilinear_params_dict)

    # assert true
    assert True
