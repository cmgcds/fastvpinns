"""
This file contains the objective function for hyperparameter tuning of the FastVPINN model.

The objective function defines the search space for hyperparameters and evaluates the model's
performance using the suggested hyperparameter values. It sets up the geometry, finite element
space, data handler, and model based on the trial's suggestions. The model is then trained for
a fixed number of epochs, and its performance is evaluated using the relative L2 error.

Author: Divij Ghose

Changelog: 9/9/24 - Initial implementation of the objective function for hyperparameter tuning


Known issues: None

Dependencies: optuna, tensorflow, fastvpinns
"""

# objective.py
import optuna
import tensorflow as tf
import os

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model import DenseModel
from fastvpinns.physics.poisson2d import pde_loss_poisson
from fastvpinns.utils.compute_utils import compute_errors_combined
from sin_cos import *  # Import your example-specific functions


def objective(trial):
    # Suggest values for hyperparameters
    config = {
        "geometry": {
            "internal_mesh_params": {
                "n_cells_x": trial.suggest_int("n_cells_x", 2, 10),
                "n_cells_y": trial.suggest_int("n_cells_y", 2, 10),
                "n_boundary_points": trial.suggest_int("n_boundary_points", 100, 1000),
            }
        },
        "fe": {
            "fe_order": trial.suggest_int("fe_order", 2, 8),
            "fe_type": trial.suggest_categorical("fe_type", ["legendre", "jacobi"]),
            "quad_order": trial.suggest_int("quad_order", 3, 15),
            "quad_type": trial.suggest_categorical("quad_type", ["gauss-legendre", "gauss-jacobi"]),
        },
        "model": {
            "model_architecture": [2]
            + [
                trial.suggest_int(f"layer_{i}", 10, 100)
                for i in range(trial.suggest_int("n_layers", 1, 5))
            ]
            + [1],
            "activation": "tanh",
            "use_attention": False,
            "learning_rate": {
                "initial_learning_rate": trial.suggest_loguniform(
                    "initial_learning_rate", 1e-5, 1e-2
                ),
                "use_lr_scheduler": True,
                "decay_steps": trial.suggest_int("decay_steps", 1000, 10000),
                "decay_rate": trial.suggest_uniform("decay_rate", 0.9, 0.99),
            },
        },
        "pde": {"beta": 10},
    }

    # Set up your model and training process using the suggested hyperparameters

    output_temp_dir = "output_temp"
    if not os.path.exists(output_temp_dir):
        os.makedirs(output_temp_dir)

    domain = Geometry_2D("quadrilateral", "internal", 100, 100, output_temp_dir)
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1],
        y_limits=[0, 1],
        n_cells_x=config["geometry"]["internal_mesh_params"]["n_cells_x"],
        n_cells_y=config["geometry"]["internal_mesh_params"]["n_cells_y"],
        num_boundary_points=config["geometry"]["internal_mesh_params"]["n_boundary_points"],
    )

    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=config["fe"]["fe_order"],
        fe_type=config["fe"]["fe_type"],
        quad_order=config["fe"]["quad_order"],
        quad_type=config["fe"]["quad_type"],
        fe_transformation_type="bilinear",
        bound_function_dict=get_boundary_function_dict(),
        bound_condition_dict=get_bound_cond_dict(),
        forcing_function=rhs,
        output_path="output_temp",
        generate_mesh_plot=False,
    )

    datahandler = DataHandler2D(fespace, domain, dtype=tf.float32)

    params_dict = {"n_cells": fespace.n_cells}
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    model = DenseModel(
        layer_dims=config["model"]["model_architecture"],
        learning_rate_dict=config["model"]["learning_rate"],
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
        use_attention=config["model"]["use_attention"],
        activation=config["model"]["activation"],
        hessian=False,
    )

    # Train the model for a fixed number of epochs
    num_epochs = 50000  # You may want to adjust this based on your computational budget
    beta = tf.constant(config["pde"]["beta"], dtype=tf.float32)

    for epoch in range(num_epochs):
        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)

    # Evaluate the model
    test_points = domain.get_test_points()
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])
    y_pred = model(test_points).numpy().reshape(-1)

    _, _, l2_error_relative, _, _, _ = compute_errors_combined(y_exact, y_pred)

    return l2_error_relative  # Return the relative L2 error as the objective to minimize
