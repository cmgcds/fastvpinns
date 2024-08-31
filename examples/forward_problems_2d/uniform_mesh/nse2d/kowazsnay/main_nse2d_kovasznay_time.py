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
from fastvpinns.model.model_nse2d import DenseModel_NSE2D
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
from kovasznay import *


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

    params_dict = {}
    params_dict['n_cells'] = fespace_velocity.n_cells

    # get the input data for the PDE
    train_dirichlet_input_list, train_dirichlet_output_list = (
        datahandler_vector.get_dirichlet_input(
            component_list=[0, 1], fespaces_list=[fespace_velocity, fespace_velocity]
        )
    )
    train_neumann_input, train_neumann_output, train_neumann_normal_x, train_neumann_normal_y = (
        datahandler_vector.get_neumann_input(
            component_list=[0, 1],
            fespaces_list=[fespace_velocity, fespace_velocity],
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

    model = DenseModel_NSE2D(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_nse2d,
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
        pressure_constraint=None,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    # obtain penalty coefficients dict from datahandler
    penalty_coefficients_dict = datahandler_vector.get_penalty_coefficients(
        get_penalty_coefficients_dict
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
    for epoch in range(i_epochs):

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

    print("Training Time (Total): ", np.sum(time_array))
    print("Training Time (Per Iteration): ", np.sum(time_array) / 1000)

    if i_use_wandb:
        wandb.save(sys.argv[1])

        wandb.log({"Adam Training Time (Total)": np.sum(time_array)})
        wandb.log({"Adam Training Time (Per Iteration)": np.sum(time_array) / 1000})

        # save the sub files
        wandb.save(str(Path(i_output_path) / "main_nse2d_kovasznay_time.py"))
