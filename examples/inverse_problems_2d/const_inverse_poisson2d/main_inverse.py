"""
file: main_inverse.py
description: This file is the main file for solving the inverse problem for the 2D Poisson equation with uniform parameters.
author: Thivin Anandh D, Divij Ghose, Sashikumaar Ganesan
date: 29/01/2024
changelog: 29/01/2024 - file created
known issues: None
"""

import copy
import sys
import time
from pathlib import Path

# import Libraries
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

# Import the example file
from inverse_uniform import *
from rich.console import Console
from tensorflow.keras import initializers, layers
from tqdm import tqdm

# Import all files from utility
from utility import *

from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.model.model_inverse import DenseModel_Inverse
from fastvpinns.physics.poisson2d_inverse import *
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.print_utils import print_table

if __name__ == "__main__":

    console = Console()
    # check input arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    # Read the YAML file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Extract the values from the YAML file
    i_output_path = config['experimentation']['output_path']

    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']
    i_n_boundary_points = config['geometry']['internal_mesh_params']['n_boundary_points']
    i_n_test_points_x = config['geometry']['internal_mesh_params']['n_test_points_x']
    i_n_test_points_y = config['geometry']['internal_mesh_params']['n_test_points_y']

    i_mesh_file_name = config['geometry']['external_mesh_params']['mesh_file_name']
    i_boundary_refinement_level = config['geometry']['external_mesh_params'][
        'boundary_refinement_level'
    ]
    i_boundary_sampling_method = config['geometry']['external_mesh_params'][
        'boundary_sampling_method'
    ]

    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']

    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']
    i_epochs = config['model']['epochs']
    i_dtype = config['model']['dtype']
    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")

    i_set_memory_growth = config['model']['set_memory_growth']
    i_learning_rate_dict = config['model']['learning_rate']

    i_num_sensor_points = config['inverse']['num_sensor_points']

    i_beta = config['pde']['beta']

    i_update_console_output = config['logging']['update_console_output']

    # ---------------------------------------------------------------#

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # Initiate a Geometry_2D object
    domain = Geometry_2D(
        i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path
    )

    # Read mesh from a .mesh file
    if i_mesh_generation_method == "external":
        cells, boundary_points = domain.read_mesh(
            mesh_file=i_mesh_file_name,
            boundary_point_refinement_level=i_boundary_refinement_level,
            bd_sampling_method=i_boundary_sampling_method,
            refinement_level=0,
        )

    elif i_mesh_generation_method == "internal":
        cells, boundary_points = domain.generate_quad_mesh_internal(
            x_limits=[i_x_min, i_x_max],
            y_limits=[i_y_min, i_y_max],
            n_cells_x=i_n_cells_x,
            n_cells_y=i_n_cells_y,
            num_boundary_points=i_n_boundary_points,
        )

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # get fespace2d
    fespace = Fespace2D(
        mesh=domain.mesh,
        cells=cells,
        boundary_points=boundary_points,
        cell_type=domain.mesh_type,
        fe_order=i_fe_order,
        fe_type=i_fe_type,
        quad_order=i_quad_order,
        quad_type=i_quad_type,
        fe_transformation_type="bilinear",
        bound_function_dict=bound_function_dict,
        bound_condition_dict=bound_condition_dict,
        forcing_function=rhs,
        output_path=i_output_path,
    )

    # Instantiate the DataHandler2D class
    datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    # Obtain sensor data
    points, sensor_values = datahandler.get_sensor_data(
        exact_solution, num_sensor_points=i_num_sensor_points, mesh_type=i_mesh_generation_method
    )

    # Obtain the inverse parameters
    inverse_params_dict = datahandler.get_inverse_params(get_inverse_params_dict)

    # obtain the target inverse parameters
    target_inverse_params_dict = get_inverse_params_actual_dict()

    # get actual Epsilon
    actual_epsilon = target_inverse_params_dict["eps"]

    model = DenseModel_Inverse(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
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
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    ## ------------- Need to do the below to print the summary of the custom model -------- ##
    # Compile the model, for summary
    # deep copy model into a new model
    # model2 = tf.keras.models.clone_model(model)
    # model2.compile(optimizer=tf.keras.optimizers.Adam())
    # model2.build(input_shape=(None, 2))
    # model2.summary()

    # ---------------------------------------------------------------#
    # --------------    Get Testing points   ----------------------- #
    # ---------------------------------------------------------------#

    # test_points = np.c_[xx.ravel(), yy.ravel()]
    # code obtains the test points based on internal or external mesh
    test_points = domain.get_test_points()
    console.print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # save points for plotting
    if i_mesh_generation_method == "internal":
        X = test_points[:, 0].reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:, 1].reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)
        plot_contour(
            x=X,
            y=Y,
            z=Y_Exact_Matrix,
            output_path=i_output_path,
            filename=f"exact_solution",
            title="Exact Solution",
        )

    # ---------------------------------------------------------------#
    # ------------- PRE TRAINING INITIALISATIONS ------------------  #
    # ---------------------------------------------------------------#
    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(
        total=num_epochs,
        desc='Training',
        unit='epoch',
        bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
        colour="green",
        ncols=100,
    )
    loss_array = []  # total loss
    test_loss_array = []  # test loss
    time_array = []  # time per epoc
    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    inverse_params_array = []

    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()

        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)

        elapsed = time.time() - batch_start_time
        progress_bar.update(1)
        # print(elapsed)
        time_array.append(elapsed)

        loss_array.append(loss['loss'])
        inverse_params_array.append(loss['inverse_params']['eps'].numpy())

        # ------ Progress bar update ------ #
        # if (epoch+1) % i_update_progress_bar == 0 or epoch == num_epochs-1:
        #     progress_bar.update(i_update_progress_bar)

        # ------ Intermediate results update ------ #
        if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:

            # Mean time per epoch
            mean_time = np.mean(time_array[-i_update_console_output:])

            # total time
            total_time_per_intermediate = np.sum(time_array[-i_update_console_output:])

            # epochs per second
            epochs_per_sec = i_update_console_output / np.sum(time_array[-i_update_console_output:])

            y_pred = model(test_points).numpy()
            y_pred = y_pred.reshape(-1)

            # get errors
            (
                l2_error,
                linf_error,
                l2_error_relative,
                linf_error_relative,
                l1_error,
                l1_error_relative,
            ) = compute_errors_combined(y_exact, y_pred)

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]", beta.numpy(), end=" ")
            console.print(f"[bold]Time/epoch : [/bold] {mean_time:.5f} s", end=" ")
            console.print("[bold]Epochs/second : [/bold]", int(epochs_per_sec), end=" ")
            console.print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
            console.print(
                f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
            )
            console.print(f"Test Losses        || L1 Error : {l1_error:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error:.3e}", end="\n")
            # add inverse parameters and senor loss
            console.print(
                f"Inverse Parameters || eps : {loss['inverse_params']['eps'].numpy():.3e}", end=" "
            )
            console.print(f"Sensor Loss : { float(loss['sensor_loss'].numpy()):.3e}", end="\n")

            # append test loss
            test_loss_array.append(l1_error)

            plot_loss_function(loss_array, i_output_path)  # plots NN loss
            plot_test_loss_function(test_loss_array, i_output_path)  # plots test loss
            # plot_inverse_param_function(inverse_params_array, r"$\epsilon$", actual_epsilon, i_output_path, "inverse_eps_prediction")

            # create a new array and perform cum_sum on time_array
            time_array_cum_sum = np.cumsum(time_array)

            #  Convert the three vectors into a single 2D matrix, where each vector is a column in the matrix
            if i_mesh_generation_method == "internal":
                # reshape y_pred into a 2D array
                y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)

                # Error
                error = np.abs(Y_Exact_Matrix - y_pred)

                # plot the prediction
                plot_contour(
                    x=X,
                    y=Y,
                    z=y_pred,
                    output_path=i_output_path,
                    filename=f"prediction_{epoch+1}",
                    title="Prediction",
                )
                # plot the error
                plot_contour(
                    x=X,
                    y=Y,
                    z=error,
                    output_path=i_output_path,
                    filename=f"error_{epoch+1}",
                    title="Error",
                )

            elif i_mesh_generation_method == "external":
                solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
                domain.write_vtk(
                    solution_array,
                    output_path=i_output_path,
                    filename=f"prediction_{epoch+1}.vtk",
                    data_names=["Sol", "Exact", "Error"],
                )

    # close the progress bar
    progress_bar.close()

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    print("[INFO] Model Saved Successfully")

    # predict the values
    y_pred = model(test_points).numpy().reshape(-1)

    # plot the loss function, prediction, exact solution and error
    plot_loss_function(loss_array, i_output_path)

    # get errors
    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]

    if i_mesh_generation_method == "internal":
        # reshape y_pred into a 2D array
        y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)

        # Error
        error = np.abs(Y_Exact_Matrix - y_pred)

        # plot the prediction
        plot_contour(
            x=X,
            y=Y,
            z=y_pred,
            output_path=i_output_path,
            filename=f"final_prediction",
            title="Prediction",
        )
        # plot the error
        plot_contour(
            x=X, y=Y, z=error, output_path=i_output_path, filename=f"final_error", title="Error"
        )

        # plot the inverse parameter prediction
        # close all existing plots
        plt.close("all")
        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.plot(inverse_params_array, label="Predicted Epsilon")
        plt.axhline(y=actual_epsilon, color="r", linestyle="--", label="Actual Epsilon")
        plt.xlabel("Epochs")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.title("Inverse Parameter Prediction")
        plt.tight_layout()
        plt.savefig(str(Path(i_output_path) / "inverse_eps_prediction.png"))
        plt.close("all")

    elif i_mesh_generation_method == "external":
        solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
        error = np.abs(y_exact - y_pred)
        domain.write_vtk(
            solution_array,
            output_path=i_output_path,
            filename=f"final_prediction.vtk",
            data_names=["Sol", "Exact", "Error"],
        )

    # domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )

    # print the Error values in table
    print_table(
        "Error Values",
        ["Error Type", "Value"],
        [
            "L2 Error",
            "Linf Error",
            "Relative L2 Error",
            "Relative Linf Error",
            "L1 Error",
            "Relative L1 Error",
        ],
        [l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative],
    )

    # print the time values in table
    print_table(
        "Time Values",
        ["Time Type", "Value"],
        [
            "Time per Epoch(s) - Median",
            "Time per Epoch(s) IQR-25% ",
            "Time per Epoch(s) IQR-75% ",
            "Mean without first(s)",
            "Mean with first(s)",
            "Epochs per second",
            "Total Train Time",
        ],
        [
            np.median(time_array[1:]),
            np.percentile(time_array[1:], 25),
            np.percentile(time_array[1:], 75),
            np.mean(time_array[1:]),
            np.mean(time_array),
            int(i_epochs / np.sum(time_array)),
            np.sum(time_array[1:]),
        ],
    )

    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
    np.savetxt(str(Path(i_output_path) / "inverse_params.txt"), np.array(inverse_params_array))
