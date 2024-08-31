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
from fastvpinns.physics.nse2d import pde_loss_nse2d
from fastvpinns.utils.plot_utils import (
    plot_contour,
    plot_loss_function,
    plot_test_loss_function,
    plot_multiple_loss_function,
)
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table

# import the example file
from fs import *


def get_dirichlet_and_test_data_external(filename, dtype):
    """
    Function is a custom function to get boundary and test data from the external file
    Reference: The data matrix and the function to load them are taken from
    Link : https://github.dev/KTH-FlowAI/Physics-informed-neural-networks-for-solving-Reynolds-averaged-Navier-Stokes-equations/blob/master/FS/train.py

    """
    d = np.load(filename)
    bc_step = 10
    u = d['u'].T
    v = d['v'].T
    x = d['x'].T
    y = d['y'].T
    p = d['p'].T
    x = x - x.min()
    y = y - y.min()
    ref = np.stack((u, v, p))
    # print(f"Shape of ref = {ref.shape}")

    test_points = np.vstack((x.flatten(), y.flatten())).T
    exact_solution = ref.reshape((3, -1)).T

    # convert to tensor
    test_points = tf.constant(test_points, dtype=dtype)
    exact_solution = tf.constant(exact_solution, dtype=dtype)

    # print(f"Shape of x = {x.shape}")
    ind_bc = np.zeros(x.shape, dtype=bool)
    ind_bc[[0, -1], ::bc_step] = True
    ind_bc[:, [0, -1]] = True

    x_bc = x[ind_bc].flatten()
    y_bc = y[ind_bc].flatten()

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
            project="ICCFD_KTH_PINNs",
            entity="starslab-iisc",
            name="FastVPINNs_Timing_Final",
            config=config,
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
        get_dirichlet_and_test_data_external("FalknerSkan_n0.08.npz", i_dtype)
    )

    # Assign the dirichlet data to the datahandler
    datahandler_vector.boundary_input_tensors_dict = dirichlet_dict_input
    datahandler_vector.boundary_actual_tensors_dict = dirichlet_dict_output

    print("Number of Dirichlet Points for u = ", dirichlet_dict_input[0].shape[0])
    print("Number of Dirichlet Points for v = ", dirichlet_dict_input[1].shape[0])
    print("Number of Test Points = ", test_points.shape[0])

    params_dict = {}
    params_dict['n_cells'] = fespace_velocity.n_cells
    params_dict['output_scaling_max'] = datahandler_vector.get_output_scaling()
    input_scaling_min, input_scaling_max = datahandler_vector.get_input_scaling(domain.cell_points)
    params_dict['input_scaling_min'] = input_scaling_min
    params_dict['input_scaling_max'] = input_scaling_max

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

    # obtain penalty coefficients dict from datahandler
    penalty_coefficients_dict = datahandler_vector.get_penalty_coefficients(
        get_penalty_coefficients_dict
    )

    num_epochs = i_epochs  # num_epochs
    num_epochs = 1000  # hardcoding the number of epochs to 1000
    progress_bar = tqdm(
        total=num_epochs,
        desc='Training',
        unit='epoch',
        bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
        colour="green",
        ncols=100,
    )

    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)
    elapsed = 0
    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    time_array = []
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
        time_array.append(elapsed)

        # update progress bar
        progress_bar.update(1)
    progress_bar.close()
    print("Training Complete")
    print("Time taken for training = ", np.sum(time_array))
    print("Time taken per iteration = ", np.mean(time_array))

    wandb.log({"Training Time": np.sum(time_array)})
    wandb.log({"Training Time per Iteration": np.mean(time_array)})

    # upload the current file and input.yaml to wandb

    wandb.save('main_nsed2d_fs_timing.py')
    wandb.save(sys.argv[1])
