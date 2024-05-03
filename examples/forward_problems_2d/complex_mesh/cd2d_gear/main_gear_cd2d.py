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


from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model import DenseModel
from fastvpinns.physics.cd2d import pde_loss_cd2d
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table

# import the example file
from cd2d_gear_example import *

# import all files from utility
from utility import *

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
    i_exact_solution_generation = config['geometry']['exact_solution']['exact_solution_generation']
    i_exact_solution_file_name = config['geometry']['exact_solution']['exact_solution_file_name']

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
    # cells, boundary_points = domain.generate_quad_mesh_internal(x_limits = [i_x_min, i_x_max], \
    #                                                                 y_limits = [i_y_min, i_y_max], \
    #                                                                 n_cells_x =  i_n_cells_x, \
    #                                                                 n_cells_y = i_n_cells_y, \
    #                                                                 num_boundary_points=i_n_boundary_points)

    cells, boundary_points = domain.read_mesh(
        i_mesh_file_name,
        i_boundary_refinement_level,
        i_boundary_sampling_method,
        refinement_level=1,
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

    # instantiate data handler
    datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    model = DenseModel(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_cd2d,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    test_points = domain.get_test_points()
    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    if i_exact_solution_generation == "internal":
        y_exact = exact_solution(test_points[:, 0], test_points[:, 1])
    else:
        exact_db = pd.read_csv(f"{i_exact_solution_file_name}", header=None, delimiter=",")
        y_exact = exact_db.iloc[:, 2].values.reshape(-1)

    # plot the exact solution
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

        # ------ Intermediate results update ------ #
        if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:
            y_pred = model(test_points).numpy()
            y_pred = y_pred.reshape(-1)

            error = np.abs(y_exact - y_pred)

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

            # Append test loss
            test_loss_array.append(l1_error)

            solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
            domain.write_vtk(
                solution_array,
                output_path=i_output_path,
                filename=f"prediction_{epoch+1}.vtk",
                data_names=["Sol", "Exact", "Error"],
            )

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]", beta.numpy(), end=" ")
            console.print(
                f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
            )
            console.print(
                f"Test Losses        || L1 Error : {l1_error:.3e} L2 Error : {l2_error:.3e} Linf Error : {linf_error:.3e}"
            )

        progress_bar.update(1)

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
    domain.write_vtk(
        solution_array,
        output_path=i_output_path,
        filename=f"prediction_{epoch+1}.vtk",
        data_names=["Sol", "Exact", "Error"],
    )
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
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
