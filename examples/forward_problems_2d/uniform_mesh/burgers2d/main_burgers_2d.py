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
from datetime import datetime
import matplotlib.pyplot as plt
import os

from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d_vector import DataHandler2D_Vector
from fastvpinns.model.model_burgers2d import DenseModel_Burgers
from fastvpinns.physics.burgers2d import pde_loss_burgers2d
from fastvpinns.utils.plot_utils import (
    plot_contour,
    plot_loss_function,
    plot_test_loss_function,
    plot_multiple_loss_function,
)
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table

# import the example file
from burgers_example import *

if __name__ == "__main__":

    console = Console()

    # check input arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <input file>")
        sys.exit(1)

    # Read the YAML file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Extract the values from the YAML file
    i_output_path = config['experimentation']['output_path']

    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_generate_mesh_plot = config['geometry']['generate_mesh_plot']
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

    i_beta = config['pde']['beta']

    i_update_console_output = config['logging']['update_console_output']

    i_use_wandb = config['wandb']['use_wandb']
    i_wandb_project_name = config['wandb']['project_name']
    i_wandb_run_prefix = config['wandb']['wandb_run_prefix']
    i_wandb_entity = config['wandb']['entity']

    # Initialise wandb
    if i_use_wandb:
        import wandb

        now = datetime.now()
        dateprefix = now.strftime("%d_%b_%Y_%H_%M")
        run_name = i_wandb_run_prefix + "_" + dateprefix
        wandb.init(
            project=i_wandb_project_name, entity=i_wandb_entity, name=run_name, config=config
        )

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # Initiate a Geometry_2D object
    domain = Geometry_2D(
        i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path
    )

    # load the mesh
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[i_x_min, i_x_max],
        y_limits=[i_y_min, i_y_max],
        n_cells_x=i_n_cells_x,
        n_cells_y=i_n_cells_y,
        num_boundary_points=i_n_boundary_points,
    )

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

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
        generate_mesh_plot=i_generate_mesh_plot,
    )

    datahandler_vector = DataHandler2D_Vector(
        fespace_list=[fespace, fespace], fespace_name_list=["u", "v"], domain=domain, dtype=i_dtype
    )

    rhs_list = datahandler_vector.get_rhs_list(
        component_list=[0, 1], fespaces_list=[fespace, fespace]
    )

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input_list, train_dirichlet_output_list = (
        datahandler_vector.get_dirichlet_input(
            component_list=[0, 1], fespaces_list=[fespace, fespace]
        )
    )
    train_neumann_input, train_neumann_output, train_neumann_normal_x, train_neumann_normal_y = (
        datahandler_vector.get_neumann_input(
            component_list=[0, 1],
            fespaces_list=[fespace, fespace],
            bd_cell_info_dict=domain.bd_cell_info_dict,
            bd_joint_info_dict=domain.bd_joint_info_dict,
        )
    )

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler_vector.get_bilinear_params_dict_as_tensors(
        get_bilinear_params_dict
    )

    model = DenseModel_Burgers(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_burgers2d,
        input_tensors_list=datahandler_vector.datahandler_variables_dict["u"]["x_pde_list"],
        orig_factor_matrices=datahandler_vector.datahandler_variables_dict,
        force_function_list=rhs_list,
        dirichlet_list=[
            train_dirichlet_input_list,
            train_dirichlet_output_list,
            train_neumann_input,
            train_neumann_output,
            train_neumann_normal_x,
            train_neumann_normal_y,
        ],
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    test_points = domain.get_test_points()
    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    # read the exact solution from the csv file
    # read the exact solution from the csv file
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])
    y_exact_u = y_exact[0]
    y_exact_v = y_exact[1]

    X = test_points[:, 0].reshape(i_n_test_points_x, i_n_test_points_y)
    Y = test_points[:, 1].reshape(i_n_test_points_x, i_n_test_points_y)
    Y_Exact_Matrix_u = y_exact_u.reshape(i_n_test_points_x, i_n_test_points_y)
    Y_Exact_Matrix_v = y_exact_v.reshape(i_n_test_points_x, i_n_test_points_y)

    # plot the exact solution
    plot_contour(
        x=X,
        y=Y,
        z=Y_Exact_Matrix_u,
        output_path=i_output_path,
        filename="exact_solution_u",
        title="Exact Solution",
    )
    plot_contour(
        x=X,
        y=Y,
        z=Y_Exact_Matrix_v,
        output_path=i_output_path,
        filename="exact_solution_v",
        title="Exact Solution",
    )

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
    test_loss_array_u = []  # test loss
    test_loss_array_v = []  # test loss
    dirichlet_loss_array = []  # dirichlet loss
    pde_loss_array = []  # pde loss

    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()
        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        elapsed = time.time() - batch_start_time

        # print(elapsed)
        time_array.append(elapsed)

        loss_array.append(loss['loss'])
        pde_loss_array.append(loss['loss_pde'])
        dirichlet_loss_array.append(loss['loss_dirichlet'])

        # ------ Intermediate results update ------ #
        if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:
            y_pred = model(test_points).numpy()
            y_pred_u = y_pred[:, 0].reshape(-1)
            y_pred_v = y_pred[:, 1].reshape(-1)

            # get errors
            (
                l2_error_u,
                linf_error_u,
                l2_error_relative_u,
                linf_error_relative_u,
                l1_error_u,
                l1_error_relative_u,
            ) = compute_errors_combined(y_exact_u, y_pred_u)

            # get errors
            (
                l2_error_v,
                linf_error_v,
                l2_error_relative_v,
                linf_error_relative_v,
                l1_error_v,
                l1_error_relative_v,
            ) = compute_errors_combined(y_exact_v, y_pred_v)

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            # append test loss
            test_loss_array_u.append(l1_error_u)
            test_loss_array_v.append(l1_error_v)

            plot_loss_function(loss_array, i_output_path)  # plots NN loss
            plot_test_loss_function(test_loss_array_u, i_output_path, "u")  # plots test loss
            plot_test_loss_function(test_loss_array_v, i_output_path, "v")  # plots test loss

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]", beta.numpy(), end=" ")
            console.print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
            console.print(
                f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
            )
            console.print(f"Test Losses(U)     || L1 Error : {l1_error_u:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error_u:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error_u:.3e}", end="\n")
            console.print(f"Test Losses(V)     || L1 Error : {l1_error_v:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error_v:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error_v:.3e}", end="\n")

            plot_multiple_loss_function(
                loss_function_list=[pde_loss_array, loss_array, dirichlet_loss_array],
                legend_labels=["PDE Loss", "Total Loss", "Dirichlet Loss"],
                output_path=i_output_path,
                filename="loss_function_components",
                y_label="Loss",
                title="Loss Function Components",
                x_label="Epochs",
            )

            if i_mesh_generation_method == "internal":
                # reshape y_pred into a 2D array
                y_pred_u = y_pred_u.reshape(i_n_test_points_x, i_n_test_points_y)
                y_pred_v = y_pred_v.reshape(i_n_test_points_x, i_n_test_points_y)

                # Error
                error_u = np.abs(Y_Exact_Matrix_u - y_pred_u)
                error_v = np.abs(Y_Exact_Matrix_v - y_pred_v)

                # plot the prediction
                plot_contour(
                    x=X,
                    y=Y,
                    z=y_pred_u,
                    output_path=i_output_path,
                    filename=f"prediction_u_{epoch+1}",
                    title="Prediction",
                )
                plot_contour(
                    x=X,
                    y=Y,
                    z=y_pred_v,
                    output_path=i_output_path,
                    filename=f"prediction_v_{epoch+1}",
                    title="Prediction",
                )

                # plot the error
                plot_contour(
                    x=X,
                    y=Y,
                    z=error_u,
                    output_path=i_output_path,
                    filename=f"error_u_{epoch+1}",
                    title="Error",
                )
                plot_contour(
                    x=X,
                    y=Y,
                    z=error_v,
                    output_path=i_output_path,
                    filename=f"error_v_{epoch+1}",
                    title="Error",
                )

            elif i_mesh_generation_method == "external":
                error_u = np.abs(y_exact_u - y_pred_u)
                error_v = np.abs(y_exact_v - y_pred_v)
                solution_array = np.c_[y_pred_u, y_pred_v, y_exact_u, y_exact_v, error_u, error_v]
                domain.write_vtk(
                    solution_array,
                    output_path=i_output_path,
                    filename=f"prediction_{epoch+1}.vtk",
                    data_names=["u_pred", "v_pred", "u_exact", "v_exact", "error_u", "error_v"],
                )

        progress_bar.update(1)

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    # print the Error values in table
    print_table(
        "Error Values- u",
        ["Error Type", "Value"],
        [
            "L2 Error",
            "Linf Error",
            "Relative L2 Error",
            "Relative Linf Error",
            "L1 Error",
            "Relative L1 Error",
        ],
        [
            l2_error_u,
            linf_error_u,
            l2_error_relative_u,
            linf_error_relative_u,
            l1_error_u,
            l1_error_relative_u,
        ],
    )

    # print the Error values in table
    print_table(
        "Error Values- v",
        ["Error Type", "Value"],
        [
            "L2 Error",
            "Linf Error",
            "Relative L2 Error",
            "Relative Linf Error",
            "L1 Error",
            "Relative L1 Error",
        ],
        [
            l2_error_v,
            linf_error_v,
            l2_error_relative_v,
            linf_error_relative_v,
            l1_error_v,
            l1_error_relative_v,
        ],
    )

    # print the time values in table
    print_table(
        "Time Values",
        ["Time Type", "Value"],
        [
            "Time per Epoch(s) - Median",
            "Time per Epoch(s) IQR-25% ",
            "Time per Epoch(s) IQR-75% ",
            "Mean (s)",
            "Epochs per second",
            "Total Train Time",
        ],
        [
            np.median(time_array),
            np.percentile(time_array, 25),
            np.percentile(time_array, 75),
            np.mean(time_array),
            int(i_epochs / np.sum(time_array)),
            np.sum(time_array),
        ],
    )

    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction_u.txt"), y_pred_u)
    np.savetxt(str(Path(i_output_path) / "prediction_v.txt"), y_pred_v)
    np.savetxt(str(Path(i_output_path) / "exact_u.txt"), y_exact_u)
    np.savetxt(str(Path(i_output_path) / "exact_v.txt"), y_exact_v)
    np.savetxt(str(Path(i_output_path) / "error_u.txt"), error_u)
    np.savetxt(str(Path(i_output_path) / "error_v.txt"), error_v)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))

    # copy the input file to the output folder
    os.system(f"cp {sys.argv[1]} {i_output_path}")

    # copy the main file to the output folder
    os.system(f"cp {__file__} {i_output_path}")

    # copy the example file to the output folder
    os.system(f"cp burgers_example.py {i_output_path}")

    # copy the model file to the output folder
    os.system(f"cp ../../../../fastvpinns/model/model_burgers2d.py {i_output_path}")

    # copy the ns2d file to the output folder
    os.system(f"cp ../../../../fastvpinns/physics/burgers2d.py {i_output_path}")

    if i_use_wandb:
        wandb.log({"loss_function": wandb.Image(i_output_path + "/loss_function.png")})
        wandb.save(sys.argv[1])

        # save the numpy arrays
        wandb.save(str(Path(i_output_path) / "loss_function.txt"))
        wandb.save(str(Path(i_output_path) / "prediction_u.txt"))
        wandb.save(str(Path(i_output_path) / "prediction_v.txt"))
        wandb.save(str(Path(i_output_path) / "exact_u.txt"))
        wandb.save(str(Path(i_output_path) / "exact_v.txt"))
        wandb.save(str(Path(i_output_path) / "error_u.txt"))
        wandb.save(str(Path(i_output_path) / "error_v.txt"))
        wandb.save(str(Path(i_output_path) / "time_per_epoch.txt"))

        # save the sub files
        wandb.save(str(Path(i_output_path) / __file__))
        wandb.save(str(Path(i_output_path) / "burgers_example.py"))
        wandb.save(str(Path(i_output_path) / "model_burgers2d.py"))
        wandb.save(str(Path(i_output_path) / "burgers2d.py"))
