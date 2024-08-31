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
from fastvpinns.model.model_nse2d_scaling import DenseModel_NSE2D_Scaling

# @@@@@ IMPORTANT @@@@@
# from fastvpinns.physics.nse2d_fs import pde_loss_nse2d_fs
from fastvpinns.physics.nse2d import pde_loss_nse2d

from fastvpinns.utils.plot_utils import (
    plot_contour_channel,
    plot_loss_function,
    plot_test_loss_function,
    plot_multiple_loss_function,
)
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table

# import the example file
from backward_step import *


def get_dirichlet_and_test_data_external(filename, dtype):
    """
    Function is a custom function to get boundary and test data from the external file
    """
    d = pd.read_csv(filename, delimiter=',', dtype=np.float64)
    bc_step = 2
    # Column values
    # ['x', 'y', 'u1', 'u2', 'p', ' u1_grad_x', ' u1_grad_y', ' u2_grad_x',
    # ' u2_grad_y', ' vorticity']
    x = d['x'].values.reshape(1000, 80)
    y = d['y'].values.reshape(1000, 80)
    u = d['u1'].values.reshape(1000, 80)
    v = d['u2'].values.reshape(1000, 80)
    p = d['p'].values.reshape(1000, 80)
    u_x = d[' u1_grad_x'].values.reshape(1000, 80)
    u_y = d[' u1_grad_y'].values.reshape(1000, 80)
    v_x = d[' u2_grad_x'].values.reshape(1000, 80)
    v_y = d[' u2_grad_y'].values.reshape(1000, 80)
    vorticity = d[' vorticity'].values.reshape(1000, 80)

    ref = np.vstack((u, v, p))
    # print(f"Shape of ref = {ref.shape}")

    test_points = np.vstack((x.flatten(), y.flatten())).T
    exact_solution = ref.reshape((3, -1)).T

    # convert to tensor
    test_points = tf.constant(test_points, dtype=dtype)
    exact_solution = tf.constant(exact_solution, dtype=dtype)

    # print(f"Shape of x = {x.shape}")
    ind_bc = np.zeros(x.shape, dtype=bool)
    ind_bc[[0, -1],] = True
    ind_bc[::bc_step, [0, -1]] = True

    x_bc = x[ind_bc].flatten()
    y_bc = y[ind_bc].flatten()

    print(
        f"Shape of x_bc = {x_bc.shape} :-> x = 0 {x_bc[x_bc == 0].shape} :-> x = 20 {x_bc[x_bc == 20].shape} :-> x = 40 {x_bc[x_bc == 40].shape} :-> x = 60 {x_bc[x_bc == 60].shape} :-> x = 80 {x_bc[x_bc == 80].shape} "
    )
    print(
        f"Shape of y_bc = {y_bc.shape} :-> y = -0.5 {y_bc[y_bc == -0.5].shape} :-> y = 0.5 {y_bc[y_bc == 0.5].shape} "
    )

    # input bc
    input_dirichlet = np.hstack((x_bc[:, None], y_bc[:, None]))

    u_bc = u[ind_bc].flatten()
    v_bc = v[ind_bc].flatten()

    # output dirichlet
    output_dirichlet_u = u_bc[:, None]
    output_dirichlet_v = v_bc[:, None]

    # print(f"Shape of input_dirichlet = {input_dirichlet.shape}")
    # print(f"Shape of output_dirichlet_u = {output_dirichlet_u.shape}")
    # print(f"Shape of output_dirichlet_v = {output_dirichlet_v.shape}")

    dirichlet_dict_input = {}
    dirichlet_dict_output = {}

    # assign value for each component
    dirichlet_dict_input[0] = tf.constant(input_dirichlet, dtype=dtype)
    dirichlet_dict_output[0] = tf.constant(output_dirichlet_u, dtype=dtype)

    dirichlet_dict_input[1] = tf.constant(input_dirichlet, dtype=dtype)
    dirichlet_dict_output[1] = tf.constant(output_dirichlet_v, dtype=dtype)

    # # print(f"Shape of cp = {cp.shape}")
    # cmp = sns.color_palette('RdBu_r', as_cmap=True)
    # for title, data in zip(["u", "v", "p"],[u, v, p]):
    #     plt.figure()
    #     plt.set_cmap(cmp)
    #     plt.contourf(x[:,:50*15], y[:,:50*15], data[:,:50*15], cmap = 'RdBu_r', levels = 50)
    #     # plot the color bar in horizontal orientation
    #     plt.colorbar(orientation = 'horizontal', pad = 0.1, aspect = 50)
    #     # custom aspect ratio of length to be 10 and height to be 2
    #     plt.gca().set_aspect(3, adjustable='box')
    #     plt.title(f"{title}")
    #     plt.show()

    return dirichlet_dict_input, dirichlet_dict_output, test_points, exact_solution, x.shape


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

    fespace_velocity = Fespace2D(
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

    fespace_pressure = Fespace2D(
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
        fespace_list=[fespace_velocity, fespace_velocity, fespace_pressure],
        fespace_name_list=["u", "v", "p"],
        domain=domain,
        dtype=i_dtype,
    )

    rhs_list = datahandler_vector.get_rhs_list(
        component_list=[0, 1], fespaces_list=[fespace_velocity, fespace_velocity]
    )

    # get the dirichlet input data from external file
    dirichlet_dict_input, dirichlet_dict_output, test_points, y_exact, original_shape = (
        get_dirichlet_and_test_data_external(
            "fem_solutions/RE_NR_200.000000_Backward_Step.csv", i_dtype
        )
    )

    # Assign the dirichlet data to the datahandler
    datahandler_vector.boundary_input_tensors_dict = dirichlet_dict_input
    datahandler_vector.boundary_actual_tensors_dict = dirichlet_dict_output

    params_dict = {}
    params_dict['n_cells'] = fespace_velocity.n_cells
    params_dict['output_scaling_max'] = datahandler_vector.get_output_scaling()
    input_scaling_min, input_scaling_max = datahandler_vector.get_input_scaling(domain.cell_points)
    params_dict['input_scaling_min'] = input_scaling_min
    params_dict['input_scaling_max'] = input_scaling_max

    print(f"Output Scaling Max = {params_dict['output_scaling_max']}")
    print(f"Input Scaling Min = {params_dict['input_scaling_min']}")
    print(f"Input Scaling Max = {params_dict['input_scaling_max']}")

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler_vector.get_bilinear_params_dict_as_tensors(
        get_bilinear_params_dict
    )

    model = DenseModel_NSE2D_Scaling(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_nse2d,
        input_tensors_list=datahandler_vector.datahandler_variables_dict["u"]["x_pde_list"],
        orig_factor_matrices=datahandler_vector.datahandler_variables_dict,
        force_function_list=rhs_list,
        dirichlet_list=[dirichlet_dict_input, dirichlet_dict_output],
        pressure_constraint=None,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
    )

    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")

    # get components
    y_exact_u = y_exact[:, 0].numpy()
    y_exact_v = y_exact[:, 1].numpy()
    y_exact_p = y_exact[:, 2].numpy()

    X = test_points[:, 0].numpy().reshape(original_shape)
    Y = test_points[:, 1].numpy().reshape(original_shape)
    Y_Exact_Matrix_u = y_exact_u.reshape(original_shape)
    Y_Exact_Matrix_v = y_exact_v.reshape(original_shape)
    Y_Exact_Matrix_p = y_exact_p.reshape(original_shape)

    # obtain penalty coefficients dict from datahandler
    penalty_coefficients_dict = datahandler_vector.get_penalty_coefficients(
        get_penalty_coefficients_dict
    )

    # plot the exact solution
    plot_contour_channel(
        x=X,
        y=Y,
        z=Y_Exact_Matrix_u,
        output_path=i_output_path,
        filename="exact_solution_u",
        title="Exact Solution",
        aspect_ratio=3,
    )
    plot_contour_channel(
        x=X,
        y=Y,
        z=Y_Exact_Matrix_v,
        output_path=i_output_path,
        filename="exact_solution_v",
        title="Exact Solution",
        aspect_ratio=3,
    )
    plot_contour_channel(
        x=X,
        y=Y,
        z=Y_Exact_Matrix_p,
        output_path=i_output_path,
        filename="exact_solution_p",
        title="Exact Solution",
        aspect_ratio=3,
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
    test_loss_array_p = []  # test loss
    dirichlet_loss_array = []  # dirichlet loss
    pde_loss_array = []  # pde loss
    divergence_loss_array = []  # divergence loss
    residual_x_loss_array = []  # residual loss
    residual_y_loss_array = []  # residual loss
    l2_regularization_loss_array = []  # regularization loss
    neumann_loss_array = []  # neumann loss

    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()
        loss = model.train_step(
            beta=beta,
            bilinear_params_dict=bilinear_params_dict,
            regularisation=None,
            penalty_coefficients_dict=penalty_coefficients_dict,
        )
        elapsed = time.time() - batch_start_time

        # print(elapsed)
        time_array.append(elapsed)

        loss_array.append(loss['loss'])
        pde_loss_array.append(loss['loss_pde'])
        dirichlet_loss_array.append(loss['loss_dirichlet'])
        l2_regularization_loss_array.append(loss['l2_regularisation'])
        divergence_loss_array.append(loss['divergence'])
        residual_x_loss_array.append(loss['residual_x'])
        residual_y_loss_array.append(loss['residual_y'])
        neumann_loss_array.append(loss['loss_neumann'])

        # ------ Intermediate results update ------ #
        if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:
            test_points_scalled = (test_points - input_scaling_min) / (
                input_scaling_max - input_scaling_min
            )

            y_pred = model(test_points_scalled).numpy()
            # rescaling
            y_pred = y_pred * params_dict['output_scaling_max']

            y_pred = y_pred.numpy()

            y_pred_u = y_pred[:, 0].reshape(-1)
            y_pred_v = y_pred[:, 1].reshape(-1)
            y_pred_p = y_pred[:, 2].reshape(-1)

            # Pressure correction
            difference = y_exact_p.mean() - y_pred_p.mean()
            y_pred_p = y_pred_p + difference

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

            # get errors
            (
                l2_error_p,
                linf_error_p,
                l2_error_relative_p,
                linf_error_relative_p,
                l1_error_p,
                l1_error_relative_p,
            ) = compute_errors_combined(y_exact_p, y_pred_p)

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())
            divergence_loss = float(loss['divergence'].numpy())
            residual_x_loss = float(loss['residual_x'].numpy())
            residual_y_loss = float(loss['residual_y'].numpy())
            neumann_loss = float(loss['loss_neumann'].numpy())

            # append test loss
            test_loss_array_u.append(l1_error_u)
            test_loss_array_v.append(l1_error_v)
            test_loss_array_p.append(l1_error_p)

            plot_loss_function(loss_array, i_output_path)  # plots NN loss
            plot_test_loss_function(test_loss_array_u, i_output_path, "u")  # plots test loss
            plot_test_loss_function(test_loss_array_v, i_output_path, "v")  # plots test loss
            plot_test_loss_function(test_loss_array_p, i_output_path, "p")  # plots test loss

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]", beta.numpy(), end=" ")
            console.print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
            console.print(
                f"Variational Losses || Pde : [red]{loss_pde:.3e}[/red] Dirichlet : [red]{loss_dirichlet:.3e}[/red] Neumann : [red]{neumann_loss:.3e}[/red] Total : [red]{total_loss:.3e}[/red]"
            )
            console.print(
                f"Residual Losses || Divergence Loss : [red]{divergence_loss:.3e}[/red] Residual X Loss : [red]{residual_x_loss:.3e}[/red] Residual Y Loss : [red]{residual_y_loss:.3e}[/red]"
            )
            console.print(f"Test Losses(U)     || L1 Error : {l1_error_u:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error_u:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error_u:.3e}", end="\n")
            console.print(f"Test Losses(V)     || L1 Error : {l1_error_v:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error_v:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error_v:.3e}", end="\n")
            console.print(f"Test Losses(P)     || L1 Error : {l1_error_p:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error_p:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error_p:.3e}", end="\n")

            plot_multiple_loss_function(
                loss_function_list=[
                    pde_loss_array,
                    l2_regularization_loss_array,
                    loss_array,
                    dirichlet_loss_array,
                    neumann_loss_array,
                ],
                legend_labels=[
                    "PDE Loss",
                    "Regularisation Loss",
                    "Total Loss",
                    "Dirichlet Loss",
                    "Neumann Loss",
                ],
                output_path=i_output_path,
                filename="loss_function_components",
                y_label="Loss",
                title="Loss Function Components",
                x_label="Epochs",
            )

            plot_multiple_loss_function(
                loss_function_list=[
                    divergence_loss_array,
                    residual_x_loss_array,
                    residual_y_loss_array,
                ],
                legend_labels=["Divergence Loss", "Residual X Loss", "Residual Y Loss"],
                output_path=i_output_path,
                filename="residual_loss_components",
                y_label="Loss",
                title="Residual Loss Components",
                x_label="Epochs",
            )

            # reshape y_pred into a 2D array
            y_pred_u = y_pred_u.reshape(original_shape)
            y_pred_v = y_pred_v.reshape(original_shape)
            y_pred_p = y_pred_p.reshape(original_shape)

            # Error
            error_u = np.abs(Y_Exact_Matrix_u - y_pred_u)
            error_v = np.abs(Y_Exact_Matrix_v - y_pred_v)
            error_p = np.abs(Y_Exact_Matrix_p - y_pred_p)

            # plot the prediction
            plot_contour_channel(
                x=X,
                y=Y,
                z=y_pred_u,
                output_path=i_output_path,
                filename=f"prediction_u_{epoch+1}",
                title="Prediction",
                aspect_ratio=3,
            )
            plot_contour_channel(
                x=X,
                y=Y,
                z=y_pred_v,
                output_path=i_output_path,
                filename=f"prediction_v_{epoch+1}",
                title="Prediction",
                aspect_ratio=3,
            )
            plot_contour_channel(
                x=X,
                y=Y,
                z=y_pred_p,
                output_path=i_output_path,
                filename=f"prediction_p_{epoch+1}",
                title="Prediction",
                aspect_ratio=3,
            )

            # plot the error
            plot_contour_channel(
                x=X,
                y=Y,
                z=error_u,
                output_path=i_output_path,
                filename=f"error_u_{epoch+1}",
                title="Error",
                aspect_ratio=3,
            )
            plot_contour_channel(
                x=X,
                y=Y,
                z=error_v,
                output_path=i_output_path,
                filename=f"error_v_{epoch+1}",
                title="Error",
                aspect_ratio=3,
            )
            plot_contour_channel(
                x=X,
                y=Y,
                z=error_p,
                output_path=i_output_path,
                filename=f"error_p_{epoch+1}",
                title="Error",
                aspect_ratio=3,
            )

        progress_bar.update(1)

    progress_bar.close()

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    # ---------------------------------------------------------------#

    # save the prediction as 2D arrays
    np.savetxt(str(Path(i_output_path) / "prediction_square_u.txt"), y_pred_u)
    np.savetxt(str(Path(i_output_path) / "prediction_square_v.txt"), y_pred_v)
    np.savetxt(str(Path(i_output_path) / "prediction_square_p.txt"), y_pred_p)

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

    # print the Error values in table
    print_table(
        "Error Values- p",
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
            l2_error_p,
            linf_error_p,
            l2_error_relative_p,
            linf_error_relative_p,
            l1_error_p,
            l1_error_relative_p,
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
    np.savetxt(str(Path(i_output_path) / "prediction_p.txt"), y_pred_p)
    np.savetxt(str(Path(i_output_path) / "exact_u.txt"), y_exact_u)
    np.savetxt(str(Path(i_output_path) / "exact_v.txt"), y_exact_v)
    np.savetxt(str(Path(i_output_path) / "exact_p.txt"), y_exact_p)
    np.savetxt(str(Path(i_output_path) / "error_u.txt"), error_u)
    np.savetxt(str(Path(i_output_path) / "error_v.txt"), error_v)
    np.savetxt(str(Path(i_output_path) / "error_p.txt"), error_p)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
    np.savetxt(str(Path(i_output_path) / "test_loss_u.txt"), np.array(test_loss_array_u))
    np.savetxt(str(Path(i_output_path) / "test_loss_v.txt"), np.array(test_loss_array_v))
    np.savetxt(str(Path(i_output_path) / "test_loss_p.txt"), np.array(test_loss_array_p))
    np.savetxt(str(Path(i_output_path) / "residual_x_loss.txt"), np.array(residual_x_loss_array))
    np.savetxt(str(Path(i_output_path) / "residual_y_loss.txt"), np.array(residual_y_loss_array))
    np.savetxt(str(Path(i_output_path) / "divergence_loss.txt"), np.array(divergence_loss_array))

    # copy the input file to the output folder
    os.system(f"cp {sys.argv[1]} {i_output_path}")

    # copy the main file to the output folder
    os.system(f"cp {__file__} {i_output_path}")

    # copy the example file to the output folder
    os.system(f"cp backward_step.py {i_output_path}")

    # copy the model file to the output folder
    os.system(f"cp ../../../../../fastvpinns/model/model_nse2d_scaling.py {i_output_path}")

    # copy the ns2d file to the output folder
    os.system(f"cp ../../../../../fastvpinns/physics/nse2d.py {i_output_path}")

    if i_use_wandb:
        wandb.log({"loss_function": wandb.Image(i_output_path + "/loss_function.png")})
        wandb.save(sys.argv[1])

        # log error values
        wandb.log(
            {
                "L2 Error U": l2_error_u,
                "Linf Error U": linf_error_u,
                "Relative L2 Error U": l2_error_relative_u,
                "Relative Linf Error U": linf_error_relative_u,
                "L1 Error U": l1_error_u,
                "Relative L1 Error U": l1_error_relative_u,
            }
        )
        wandb.log(
            {
                "L2 Error V": l2_error_v,
                "Linf Error V": linf_error_v,
                "Relative L2 Error V": l2_error_relative_v,
                "Relative Linf Error V": linf_error_relative_v,
                "L1 Error V": l1_error_v,
                "Relative L1 Error V": l1_error_relative_v,
            }
        )
        wandb.log(
            {
                "L2 Error P": l2_error_p,
                "Linf Error P": linf_error_p,
                "Relative L2 Error P": l2_error_relative_p,
                "Relative Linf Error P": linf_error_relative_p,
                "L1 Error P": l1_error_p,
                "Relative L1 Error P": l1_error_relative_p,
            }
        )

        # log the penalty coefficients
        wandb.log({"Penalty Coefficients": penalty_coefficients_dict})

        # save the numpy arrays
        wandb.save(str(Path(i_output_path) / "loss_function.txt"))
        wandb.save(str(Path(i_output_path) / "prediction_u.txt"))
        wandb.save(str(Path(i_output_path) / "prediction_v.txt"))
        wandb.save(str(Path(i_output_path) / "prediction_p.txt"))
        wandb.save(str(Path(i_output_path) / "exact_u.txt"))
        wandb.save(str(Path(i_output_path) / "exact_v.txt"))
        wandb.save(str(Path(i_output_path) / "exact_p.txt"))
        wandb.save(str(Path(i_output_path) / "error_u.txt"))
        wandb.save(str(Path(i_output_path) / "error_v.txt"))
        wandb.save(str(Path(i_output_path) / "error_p.txt"))
        wandb.save(str(Path(i_output_path) / "time_per_epoch.txt"))
        wandb.save(str(Path(i_output_path) / "test_loss_u.txt"))
        wandb.save(str(Path(i_output_path) / "test_loss_v.txt"))
        wandb.save(str(Path(i_output_path) / "test_loss_p.txt"))
        wandb.save(str(Path(i_output_path) / "residual_x_loss.txt"))
        wandb.save(str(Path(i_output_path) / "residual_y_loss.txt"))
        wandb.save(str(Path(i_output_path) / "divergence_loss.txt"))

        # save the sub files
        wandb.save(str(Path(i_output_path) / __file__))
        wandb.save(str(Path(i_output_path) / "backward_step.py"))
        wandb.save(str(Path(i_output_path) / "model_nse2d_scaling.py"))
        wandb.save(str(Path(i_output_path) / "nse2d.py"))
