import numpy as np
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
import optuna
import argparse


from fastvpinns.Geometry.geometry_domain_decomposition import GeometryDomainDecomposition
from fastvpinns.FE.fespace_domain_decomposition import FESpaceDomainDecomposition
from fastvpinns.domain_decomposition.decompositions.uniform import UniformDomainDecomposition
from fastvpinns.data.datahandler_domain_decomposition import DataHandlerDomainDecomposition
from fastvpinns.domain_decomposition.window_functions.cosine import CosineWindowFunction
from fastvpinns.domain_decomposition.model import DenseModelDomainDecomposition
from fastvpinns.domain_decomposition.inference import InferenceDomainDecomposition
from fastvpinns.physics.poisson2d import pde_loss_poisson
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function, plot_array
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table
from fastvpinns.domain_decomposition.plot_utils import plot_subdomains

# import the example file
from sin_cos import *

# import all files from utility
from utility import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastVPINNs with YAML config or optimized hyperparameters"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="input.yaml",
        help="Path to YAML config file (default: input.yaml)",
    )
    parser.add_argument("--optimized", action="store_true", help="Use optimized hyperparameters")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5000,
        help="Number of epochs to train each model in the hyperparameter optimization",
    )
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')

    if args.optimized:
        from fastvpinns.hyperparameter_tuning.optuna_tuner import OptunaTuner

        print("Running with optimized hyperparameters")
        print("This may take a while...")
        print("Running OptunaTuner...")

        tuner = OptunaTuner(n_trials=args.n_trials, n_jobs=len(gpus), n_epochs=args.n_epochs)
        best_params = tuner.run()
        # Convert best_params to the format expected by your code
        # config = convert_best_params_to_config(best_params)
        print("OptunaTuner completed")
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        sys.exit(0)
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Please provide either a config file or use --optimized flag")
        sys.exit(1)

    console = Console()

    # # check input arguments
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <input file>")
    #     sys.exit(1)

    # Extract the values from the YAML file
    i_output_path = config['experimentation']['output_path']
    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_generate_mesh_plot = config['geometry']['generate_mesh_plot']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
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
    i_epochs = config['model']['epochs']
    i_set_memory_growth = config['model']['set_memory_growth']
    i_update_console_output = config['logging']['update_console_output']
    i_dtype = config['model']['dtype']

    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")

    # Values that are hyperparameters:
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']
    i_n_boundary_points = config['geometry']['internal_mesh_params']['n_boundary_points']

    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']

    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']

    i_learning_rate_dict = config['model']['learning_rate']

    i_beta = config['pde']['beta']

    i_kernel_size_x = config['domain_decomposition']['domain_decomposition_mesh_params'][
        'kernel_size_x'
    ]
    i_kernel_size_y = config['domain_decomposition']['domain_decomposition_mesh_params'][
        'kernel_size_y'
    ]
    i_stride_x = config['domain_decomposition']['domain_decomposition_mesh_params']['stride_x']
    i_stride_y = config['domain_decomposition']['domain_decomposition_mesh_params']['stride_y']

    kernel_size_dict = {
        'kernel_size_x': i_kernel_size_x,
        'kernel_size_y': i_kernel_size_y,
        'stride_x': i_stride_x,
        'stride_y': i_stride_y,
    }

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # ---------------------------------------------------------------#
    # ------------------------- Geometry --------------------------- #
    # ---------------------------------------------------------------#

    # Initiate a GeometryDomainDecomposition object
    domain = GeometryDomainDecomposition(
        i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path
    )

    # Generate the subdomains
    cells_in_domain, subdomains_in_domain, grid_x, grid_y = domain.generate_subdomains(
        x_limits=[i_x_min, i_x_max],
        y_limits=[i_y_min, i_y_max],
        n_cells_x=i_n_cells_x,
        n_cells_y=i_n_cells_y,
        kernel_size_dict=kernel_size_dict,
    )

    vertex_coordinates_in_cells, subdomain_boundary_limits = (
        domain.generate_quad_mesh_with_domain_decomposition(
            x_limits=[i_x_min, i_x_max],
            y_limits=[i_y_min, i_y_max],
            n_cells_x=i_n_cells_x,
            n_cells_y=i_n_cells_y,
        )
    )

    cells_points_subdomain = {}
    for i, block in enumerate(subdomains_in_domain):
        cells_points_subdomain[i] = domain.assign_sub_domain_coords(
            block, vertex_coordinates_in_cells
        )

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # ---------------------------------------------------------------#
    # --------------------- Domain Decomposition ------------------- #
    # ---------------------------------------------------------------#
    decomposed_domain = UniformDomainDecomposition(domain)
    decomposed_domain.subdomain_boundary_limits = subdomain_boundary_limits
    decomposed_domain.unnormalizing_factor = get_unnormalizing_factor()
    decomposed_domain.hard_constraints_factor = get_hard_constraints_factor()

    window_function = CosineWindowFunction(decomposed_domain)

    # ---------------------------------------------------------------#
    # -------------------------- FE Spaces ------------------------- #
    # ---------------------------------------------------------------#
    fe_spaces_for_subdomains = {}
    for i in range(len(subdomains_in_domain)):
        fe_spaces_for_subdomains[i] = FESpaceDomainDecomposition(
            cells=cells_points_subdomain[i],
            cell_type=domain.mesh_type,
            fe_order=i_fe_order,
            fe_type=i_fe_type,
            quad_order=i_quad_order,
            quad_type=i_quad_type,
            fe_transformation_type="bilinear",
            forcing_function=rhs,
            output_path=i_output_path,
        )

    # instantiate data handler
    datahandler = {}
    for i in range(len(subdomains_in_domain)):
        datahandler[i] = DataHandlerDomainDecomposition(
            fe_spaces_for_subdomains[i], domain, i, dtype=i_dtype
        )

    for i in range(len(subdomains_in_domain)):
        decomposed_domain.window_function_values[i] = datahandler[
            i
        ].get_window_function_values_for_training(window_function)

    # Initialise params

    for i in range(len(subdomains_in_domain)):
        decomposed_domain.params_dict[i] = {}
        decomposed_domain.params_dict[i]['n_cells'] = fe_spaces_for_subdomains[i].n_cells

    # obtain the boundary points for dirichlet boundary conditions
    train_dirichlet_input, train_dirichlet_output = {}, {}
    for i in range(len(subdomains_in_domain)):
        train_dirichlet_input[i], train_dirichlet_output[i] = tf.zeros(
            datahandler[i].x_pde_list.shape
        ), tf.zeros(datahandler[i].x_pde_list.shape)

    # obtain bilinear params dict
    for i in range(len(subdomains_in_domain)):
        decomposed_domain.bilinear_params_dict[i] = datahandler[
            i
        ].get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    # ---------------------------------------------------------------#
    # -------------------------- Model ----------------------------- #
    # ---------------------------------------------------------------#
    model = {}
    for i in range(len(subdomains_in_domain)):
        model[i] = DenseModelDomainDecomposition(
            layer_dims=i_model_architecture,
            learning_rate_dict=i_learning_rate_dict,
            subdomain_id=i,
            decomposed_domain=decomposed_domain,
            loss_function=pde_loss_poisson,
            datahandler=datahandler[i],
            use_attention=i_use_attention,
            activation=i_activation,
            hessian=False,
        )

    test_points = domain.get_test_points()
    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # save points for plotting
    X = test_points[:, 0].reshape(i_n_test_points_x, i_n_test_points_y)
    Y = test_points[:, 1].reshape(i_n_test_points_x, i_n_test_points_y)
    Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

    # plot the exact solution
    plot_contour(
        x=X,
        y=Y,
        z=Y_Exact_Matrix,
        output_path=i_output_path,
        filename="exact_solution",
        title="Exact Solution",
    )

    inference_obj = InferenceDomainDecomposition(
        decomposed_domain=decomposed_domain,
        model=model,
        datahandler=datahandler,
        window_function=window_function,
        test_points=test_points,
        exact_solution=exact_solution,
        num_test_points=[i_n_test_points_x, i_n_test_points_y],
        output_path=i_output_path,
    )


    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(
        total=num_epochs,
        desc='Training',
        unit='epoch',
        bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
        colour="red",
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
    time_array_pretrain = []
    time_array_train = []
    loss_subdomain = {}
    for i in range(len(subdomains_in_domain)):
        loss_subdomain[i] = []
    for epoch in range(num_epochs):

        # Train the model
        time_val = time.time()
        for i in range(len(subdomains_in_domain)):
            val, grad_x, grad_y = model[i].pretrain_step()
            datahandler[i].update_subdomain_solution(val, grad_x, grad_y)
            # console.print(f"[bold]Subdomain {i} Solution Values:[/bold]")
            # console.print(datahandler[i].subdomain_solution_values)
            # console.print("\n")



        time_array_pretrain.append(time.time() - time_val)

        decomposed_domain.update_overlap_values(datahandler, fe_spaces_for_subdomains)
        # for i in range(len(subdomains_in_domain)):
        #     console.print(f"[bold]Overlap Solution Values:[/bold]")
        #     console.print(datahandler[i].overlap_solution_values)
        #     console.print("\n")
        





        for i in range(len(subdomains_in_domain)):
            loss = model[i].train_step(
                bilinear_params_dict=decomposed_domain.bilinear_params_dict[i],
                overlap_val=datahandler[i].overlap_solution_values,
                overlap_grad_x=datahandler[i].overlap_solution_gradient_x,
                overlap_grad_y=datahandler[i].overlap_solution_gradient_y,
            )

            loss_subdomain[i].append(loss["loss"])
        time_array_train.append(time.time() - time_val)

        

        progress_bar.update(1)
        if epoch % 1000 == 0:
            inference_obj.compute_solution_nd_error()
            for i in range(len(subdomains_in_domain)):
                plot_array(loss_subdomain[i], i_output_path, f"loss_subdomain_{i}", title=f"Loss Subdomain {i}")


    exit()

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

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
