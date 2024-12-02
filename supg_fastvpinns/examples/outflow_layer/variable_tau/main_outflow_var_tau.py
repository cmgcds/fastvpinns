# Main File for running the Python code
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  30/Aug/2023


# import Libraries
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from datetime import datetime
import yaml
import sys
import os
import time
import keras

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

np.random.seed(812)
# tf.random.set_seed(1234)
keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()
# np.random.seed(1234)
# tf.random.set_seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}

console = Console()

# import the FE classes for 2D
from FE_2D.basis_function_2d import *
from FE_2D.fespace2d_supg import *

# Import the Geometry class
from Geometry.geometry_2d import *

# import the model class with hard constraints

from model.model_hard_adap_tau_outflow import *

# import the example file
from outflow_layer_example import *

#import physics for custom loss function
from physics.cd2d_supg_tau_domain import *

# import the data handler class
from data.datahandler2d_supg import *

# import the plot utils
from utils.plot_utils import *
from utils.compute_utils import *
from utils.print_utils import *


if __name__ == "__main__":
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
    i_boundary_refinement_level = config['geometry']['external_mesh_params']['boundary_refinement_level']
    i_boundary_sampling_method = config['geometry']['external_mesh_params']['boundary_sampling_method']

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
    i_lam = config['pde']['lam']
    i_pd_lam = config['pde']['pd_lam']
    # i_tau = config['pde']['tau']
    i_tau_growth = config['pde']['tau_growth']

    i_update_progress_bar = config['logging']['update_progress_bar']
    i_update_console_output = config['logging']['update_console_output']
    i_update_solution_images = config['logging']['update_solution_images']
    i_test_error_last_n_epochs = config['logging']['test_error_last_n_epochs']

    
    # For expansion of GPU Memory
    if i_set_memory_growth:
        console.print("[INFO] Setting memory growth for GPU")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
 

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # Initiate a Geometry_2D object
    domain = Geometry_2D(i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path)

    # Read mesh from a .mesh file
    if i_mesh_generation_method == "external":
        cells, boundary_points = domain.read_mesh(mesh_file = i_mesh_file_name, \
                                                  boundary_point_refinement_level=i_boundary_refinement_level, \
                                                  bd_sampling_method=i_boundary_sampling_method, \
                                                  refinement_level=0)

    
    elif i_mesh_generation_method == "internal":
        cells, boundary_points = domain.generate_quad_mesh_internal(x_limits = [i_x_min, i_x_max], \
                                                                    y_limits = [i_y_min, i_y_max], \
                                                                    n_cells_x =  i_n_cells_x, \
                                                                    n_cells_y = i_n_cells_y, \
                                                                    num_boundary_points=i_n_boundary_points)

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()
    
    
    # get fespace2d
    fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, 
                        cell_type=domain.mesh_type, fe_order=i_fe_order, fe_type =i_fe_type ,quad_order=i_quad_order, quad_type = i_quad_type, \
                        fe_transformation_type="bilinear", bound_function_dict = bound_function_dict, \
                        bound_condition_dict = bound_condition_dict, \
                        forcing_function=rhs, output_path=i_output_path, generate_mesh_plot = i_generate_mesh_plot)



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

    model = DenseModel(layer_dims = i_model_architecture, learning_rate_dict = i_learning_rate_dict, \
                       params_dict = params_dict, \
                       loss_function = pde_loss_cd2d, input_tensors_list = [datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output], \
                        orig_factor_matrices = [datahandler.shape_val_mat_list , datahandler.grad_x_mat_list, datahandler.grad_y_mat_list], \
                        force_function_list=datahandler.forcing_function_list, \
                        real_forcing_function = datahandler.real_forcing_function, \
                        tensor_dtype = i_dtype,
                        use_attention=i_use_attention, \
                        activation=i_activation, \
                        hessian=False)


    ## ------------- Need to do the below to print the summary of the custom model -------- ##

    # ---------------------------------------------------------------#
    # --------------    Get Testing points   ----------------------- #
    # ---------------------------------------------------------------#
    
    # test_points = np.c_[xx.ravel(), yy.ravel()]
    #code obtains the test points based on internal or external mesh
    test_points = domain.get_test_points()
    console.print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:,0], test_points[:,1])
    

    # save points for plotting
    if i_mesh_generation_method == "internal":
        X = test_points[:,0].reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:,1].reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

        # plot the exact solution
        plot_contour(x = X, y = Y, z = Y_Exact_Matrix, output_path = i_output_path, filename= "exact_solution", title = "Exact Solution")
    
    # ---------------------------------------------------------------#
    # ------------- PRE TRAINING INITIALISATIONS ------------------  #
    # ---------------------------------------------------------------#
    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(total=num_epochs, desc='Training', unit='epoch', bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}", colour="green", ncols=100)
    loss_array = []   # total loss
    pde_loss_array = [] # pde Loss
    regularisation_loss = [] # regularisation Loss
    pde_loss_wtsupg_array = []
    supg_loss_array = []
    tau_array = []
    l2_lamda_array = [] # l2_lambda
    test_l1_loss_array = []
    test_l2_loss_array = []
    test_linf_loss_array = []
    k_list = []
    k1_list = []
    epoch_array = [] # Epoch_array
    time_array = []   # time per epoc

    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)
    # Least Error Collection - Entry Condition
    least_l2_error = np.inf

    # function for exponential decay of l2_regularisation
    def exponential_decay(epoch):
        initial_lrate = 0.0001 # 1e-4
        final_lrate = 0.00006  # 6e-5 
        decay_rate = -np.log(final_lrate / initial_lrate) / 3e5  #300K
        lrate = initial_lrate * np.exp(-decay_rate * epoch)
        return lrate

    # tau_growth = 0.0
    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    # k = 10.0/1e-8
    # k = 30
    # k1 = 10 #.0/1e-8   
    # k1 = 1
    # k1 = 30
    # k = 10/1e-8
    # bb = tf.sin(tf.cast(tf.constant(math.pi),tf.float64)*test_points[:, 1:2] )* tf.sin(tf.cast(tf.constant(math.pi),tf.float64)*test_points[:, 0:1])
    # bb = tf.tanh(50*test_points[:, 0:1]) * tf.tanh(50*test_points[:, 1:2]) * tf.tanh(50*(1-test_points[:, 0:1])) *tf.tanh(50*(1-test_points[:, 1:2]))
    # # bb = (1 - tf.exp(-k1*test_points[:, 1:2])) * (1-tf.exp(-k1*test_points[:, 0:1])) \
    # #                                                * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))
    # aa = (1 - tf.exp(-10*k*test_points[:, 1:2])) * (1-tf.exp(-k*test_points[:, 0:1])) \
    #                                                * (1 - tf.exp(-k * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k * (1 - test_points[:, 1:2])))
    # aa = 1/(1+tf.exp(k*((((test_points[:, 0:1]-0.5)**2+(test_points[:, 1:2]-0.5)**2-0.05))))) * \
    #     (1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) \
    #                                * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))
   
    # bb = (1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) \
    #                                * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))
    #                                 # bb = (1/(1+tf.exp(k*((((test_points[:, 0:1]-0.5)**2+(test_points[:, 1:2]-0.5)**2-0.05)))))-1/(1+tf.exp(k*((((test_points[:, 0:1]-0.5)**2+(test_points[:, 1:2]-0.5)**2)-0.03))))) * \
    # aa = tf.exp(-k * (((test_points[:, 0:1] - 0.5)**2 + (test_points[:, 1:2] - 0.5)**2))) \
    # aa = (1/(1+tf.exp(k * (((test_points[:, 0:1] - 0.5)**2 + (test_points[:, 1:2] - 0.5)**2))))) \
    #               *(1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) \
    #                                                                 * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))

                    #                          * (1 - tf.exp(-k1 * (1 - self.input_tensor[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - self.input_tensor[:, 1:2])))

    # aa = tf.math.sin(tf.cast(tf.constant(math.pi),tf.float64)*test_points[:, 0:1]) * tf.math.sin(tf.cast(tf.constant(math.pi),tf.float64)*test_points[:, 1:2])

    # aa = (1 - tf.exp(-k*test_points[:, 0:1])) * (1 - tf.exp(-k*test_points[:, 1:2])) \
    #                                                 * (1 - tf.exp(-k * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k * (1 - test_points[:, 1:2])))
    # aa = tf.tanh(10.0*(test_points[:,0:1]))*tf.tanh(10.0*(test_points[:,1:2]))*tf.tanh(10.0*(1-test_points[:,0:1]))*tf.tanh(10.0*(1-test_points[:,1:2]))
    # bb = 0.5*(1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) \
                                                                    # * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))

    
    for epoch in range(num_epochs):   
        # tau_growth = 1.0/(1.0 + np.exp(-1.0*(epoch-10000)))
        tau_growth = i_tau_growth
        
        # compute l2_lambda
        # l2_lambda = exponential_decay(epoch+1)
        # l2_lamda_array.append(l2_lambda)
        l2_lambda = i_lam
        # print(l2_lambda)
        # sys.exit(1)
        pd_lambda = i_pd_lam
        # tau = i_tau

        # Train the model
        batch_start_time = time.time()
        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict, l2_lambda=tf.constant(l2_lambda, dtype=i_dtype), pd_lambda =tf.constant(pd_lambda, dtype=i_dtype), \
                                    tau =tf.constant(tau_growth, dtype=i_dtype)) #tau = float(tau_growth)) # tau =tf.constant(tau, dtype=i_dtype))  #tau =tf.constant(tau_growth, dtype=i_dtype))
        
        elapsed = time.time() - batch_start_time
        progress_bar.update(1)
        # print(elapsed)
        time_array.append(elapsed)
        
        
        loss_array.append(loss['loss'])
        regularisation_loss.append(loss['l2_regularisation'])
        pde_loss_array.append(loss['loss_pde'])
        pde_loss_wtsupg_array.append(loss['pde_loss(without_supg)'])
        supg_loss_array.append(loss['supg_loss'])
        k_list.append(loss['k'])
        k1_list.append(loss['k1'])
        

        # #------ Progress bar update ------ #
        # if (epoch+1) % i_update_progress_bar == 0 or epoch == num_epochs-1:
        #     progress_bar.update(i_update_progress_bar)
        
        # ------ Intermediate results update ------ #
        if (epoch+1) % i_update_console_output == 0  or epoch == num_epochs-1:
            
            # Mean time per epoch
            mean_time = np.mean(time_array[-i_update_console_output:])

            #total time
            total_time_per_intermediate = np.sum(time_array[-i_update_console_output:])
            k = loss['k'].numpy()
            k1 = loss['k1'].numpy()
            #epochs per second
            epochs_per_sec = i_update_console_output/np.sum(time_array[-i_update_console_output:])
            bb = tf.tanh(50*test_points[:, 0:1]) * tf.tanh(50*test_points[:, 1:2]) * tf.tanh(50*(1-test_points[:, 0:1])) *tf.tanh(50*(1-test_points[:, 1:2]))
        # bb = (1 - tf.exp(-k1*test_points[:, 1:2])) * (1-tf.exp(-k1*test_points[:, 0:1])) \
        #                                                * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))
            # aa = (1 - tf.exp(-(10**k)*test_points[:, 1:2])) * (1-tf.exp(-(10**k)*test_points[:, 0:1])) \
            #                                         * (1 - tf.exp(-(10**k1) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-(10**k1) * (1 - test_points[:, 1:2])))
            
            
            # aa = (1 - tf.exp(-(k)*test_points[:, 1:2])) * (1-tf.exp(-(k)*test_points[:, 0:1])) \
            #                                         * (1 - tf.exp(-(k1) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-(k1) * (1 - test_points[:, 1:2])))
            
            # aa = 1/(1+tf.exp(10**(k)*((test_points[:, 0:1]-0.5)**2+(test_points[:, 1:2]-0.5)**2+k1)))
            
            #Outflow Layer
            k = 10.0/1e-8
            k1 = 30
            aa = (1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) * (1 - tf.exp(-k * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k * (1 - test_points[:, 1:2])))

            # k = 3.95 #10.0/1e-8
            # k1 = 3.37 #30
            # k2 = 0.51
            # aa = (1 - tf.exp(-10**(k2)*test_points[:, 0:1])) * (1 - tf.exp(-10**(k1)*test_points[:, 1:2])) \
            #                                             * (1 - tf.exp(-10**(k) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-10**(k1) * (1 - test_points[:, 1:2])))


            y_pred = model(test_points)[:,0:1]
            y_pred = y_pred * aa
            y_pred = y_pred.numpy().reshape(-1,)

            # np.savetxt(f"{Path(i_output_path)}/prediction_{epoch}.txt", y_pred)
            tau = model(test_points)[:,1:2]
            tau = tf.sigmoid(tau)
            # tau - tf.square(tau)
            tau = tau_growth * tau * bb
            tau = tau.numpy().reshape(-1,)

            # get errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)
            
            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())
            Reggularisation_loss = float(loss['l2_regularisation'].numpy())
            pde_loss_wt_supg = float(loss['pde_loss(without_supg)'].numpy())
            supg_loss = float(loss['supg_loss'].numpy())

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]" , beta.numpy(), end=" ")
            console.print("[bold]L2_Lambda : [/bold]" , l2_lambda, end=" ")
            console.print("[bold]Pd_Lambda : [/bold]" , pd_lambda, end=" ")
            console.print("[bold]tau : [/bold]" , tau_growth, end=" ")
            console.print("[bold]k : [/bold]", k, end = " ")
            console.print("[bold]k1 : [/bold]", k1, end = " ")
            console.print(f"[bold]Time/epoch : [/bold] {mean_time:.5f} s", end=" ")
            console.print("[bold]Epochs/second : [/bold]" , int(epochs_per_sec), end=" ")
            console.print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
            console.print(f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Reg Loss : [red]{Reggularisation_loss:.3e}[/red] PDE_loss(without SUPG) :  [red]{pde_loss_wt_supg:.3e}[/red] SUPG loss : [red]{supg_loss:.3e}[/red]  \
                                Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]")
            console.print(f"Test Losses        || L1 Error : {l1_error:.3e}", end=" ")
            console.print(f"L2 Error : {l2_error:.3e}", end=" ")
            console.print(f"Linf Error : {linf_error:.3e}", end="\n")
        

            # Obtain the cell Loss array 
            cell_loss_array = loss['cells_residual'].numpy()

            # plot the cell residual heat map
            domain.plot_adaptive_mesh(cells_list = domain.cell_points, area_averaged_cell_loss_list = cell_loss_array, epoch = epoch+1)
        

            plot_loss_function(loss_array, i_output_path)  # plots NN loss 
            plot_array(k_list, output_path = i_output_path, filename="k_value", \
                    title="trainable_var_k", x_label="Epochs", y_label="k")
            plot_array(k1_list, output_path = i_output_path, filename="k1_value", \
                    title="trainable_var_k1", x_label="Epochs", y_label="k1")
            plot_test_loss_function(test_l2_loss_array, i_output_path) # plots test loss

            

            plot_multiple_loss_function(loss_function_list = [pde_loss_array, regularisation_loss, loss_array], \
                                        output_path = i_output_path, filename="loss_components_L2_reg", \
                                        legend_labels = ["PDE loss", "Regularisation Loss", "Total losss"], \
                                        y_label = "Loss", x_label = "Epochs", title="Loss Components")

            plot_multiple_loss_function(loss_function_list = [pde_loss_wtsupg_array, supg_loss_array, loss_array], \
                                        output_path = i_output_path, filename="loss_components_Supg", \
                                        legend_labels = ["PDE loss(without_supg)", "Supg Loss", "Total losss"], \
                                        y_label = "Loss", x_label = "Epochs", title="Loss Components")

            # magnitude of pde_loss / magnitude of regularisation_loss
            pde_reg_ratio = np.abs(np.array(regularisation_loss)/np.array(pde_loss_array))
            pde_supg_ratio = np.abs(np.array(supg_loss_array)/np.array(pde_loss_wtsupg_array))
    
            # plot pde/regularisation ratio
            plot_array(pde_reg_ratio, output_path = i_output_path, filename="pde_reg_ratio", \
                    title="PDE/Regularisation Ratio", x_label="Epochs", y_label="PDE/Regularisation Ratio")

            # plot pde/supg ratio
            plot_array(pde_supg_ratio, output_path = i_output_path, filename="pde_supg_ratio", \
                    title="PDE/Supg Ratio", x_label="Epochs", y_label="PDE/Supg Ratio")
            
            # plot l2_lambda array
            plot_array(l2_lamda_array, output_path = i_output_path, filename="l2_lambda", \
                        title="L2 Lambda", x_label="Epochs", y_label="L2 Lambda")
        

            # create a new array and perform cum_sum on time_array
            time_array_cum_sum = np.cumsum(time_array)
            
            #  Convert the three vectors into a single 2D matrix, where each vector is a column in the matrix
            if i_mesh_generation_method == "internal":
                if (epoch+1) % i_update_console_output == 0 or epoch == num_epochs-1:
                    # reshape y_pred into a 2D array
                    y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
                    tau = tau.reshape(i_n_test_points_x, i_n_test_points_y)
                    
                    #Error
                    error = np.abs(Y_Exact_Matrix - y_pred)
                    
                    # plot the prediction
                    plot_contour(x = X, y = Y, z = y_pred, output_path = i_output_path, filename= f"prediction_{epoch+1}", title = "Prediction")
                    # plot the error
                    plot_contour(x = X, y = Y, z = error, output_path = i_output_path, filename= f"error_{epoch+1}", title = "Error")
                    # plot the tau
                    plot_contour(x = X, y = Y, z = tau, output_path = i_output_path, filename= f"tau_{epoch+1}", title = "Tau")


            elif i_mesh_generation_method == "external":
                solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
                domain.write_vtk(solution_array, output_path = i_output_path, filename= f"prediction_{epoch+1}.vtk", data_names = ["Sol","Exact", "Error"] )

            
            

        if (epoch+1) >= num_epochs - i_test_error_last_n_epochs:
            # compute Errors
            # k = loss['k']
            # k1 = loss['k1']
            bb = tf.tanh(50*test_points[:, 0:1]) * tf.tanh(50*test_points[:, 1:2]) * tf.tanh(50*(1-test_points[:, 0:1])) *tf.tanh(50*(1-test_points[:, 1:2]))
        # bb = (1 - tf.exp(-k1*test_points[:, 1:2])) * (1-tf.exp(-k1*test_points[:, 0:1])) \
        #                                                * (1 - tf.exp(-k1 * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k1 * (1 - test_points[:, 1:2])))
            # aa = (1 - tf.exp(-(10**k)*test_points[:, 1:2])) * (1-tf.exp(-(10**k)*test_points[:, 0:1])) \
            #                                         * (1 - tf.exp(-(10**k1) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-(10**k1) * (1 - test_points[:, 1:2])))

            # aa = (1 - tf.exp(-(k)*test_points[:, 1:2])) * (1-tf.exp(-(k)*test_points[:, 0:1])) \
            #                                         * (1 - tf.exp(-(k1) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-(k1) * (1 - test_points[:, 1:2])))
            # aa = 1/(1+tf.exp(10**(k)*((test_points[:, 0:1]-0.5)**2+(test_points[:, 1:2]-0.5)**2+k1)))
            
            # Outflow layer
            k = 10.0/1e-8
            k1 = 30
            aa = (1 - tf.exp(-k1*test_points[:, 0:1])) * (1 - tf.exp(-k1*test_points[:, 1:2])) * (1 - tf.exp(-k * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-k * (1 - test_points[:, 1:2])))

            # k = 3.95 #10.0/1e-8
            # k1 = 3.37 #30
            # k2 = 0.51
            # aa = (1 - tf.exp(-10**(k2)*test_points[:, 0:1])) * (1 - tf.exp(-10**(k1)*test_points[:, 1:2])) \
            #                                             * (1 - tf.exp(-10**(k) * (1 - test_points[:, 0:1]))) * (1 - tf.exp(-10**(k1) * (1 - test_points[:, 1:2])))


            y_pred = model(tf.constant(test_points, dtype=i_dtype))[:,0:1]
            y_pred = y_pred*aa
            y_pred = y_pred.numpy()
            y_pred = y_pred.reshape(-1,)

            tau_pred = model(tf.constant(test_points, dtype=i_dtype))[:,1:2]
            tau_pred = tf.sigmoid(tau_pred)
            tau_pred = tau_growth*tau_pred*bb
            tau_pred = tau_pred.numpy()
            tau_pred = tau_pred.reshape(-1,)

            # get errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)

            if l2_error < least_l2_error:
                least_l2_error = l2_error
                least_l2_error_epoch = epoch+1
                linf_at_min = linf_error
                l1_at_min = l1_error
                l2_relative_at_min = l2_error_relative
                linf_relative_at_min = linf_error_relative
                l1_relative_at_min = l1_error_relative
                least_l2_error_pred = y_pred
                least_l2_error_tau = tau_pred
                pde_loss_at_min = loss_pde  # pde loss at min error

                # save the final prediction
                np.savetxt(str(Path(i_output_path) / "min_error_prediction.txt"), y_pred)
                np.savetxt(str(Path(i_output_path) / "min_error_tau.txt"), tau_pred)

                # save all the errors and epochs as a text in a single text file
                with open(str(Path(i_output_path) / "min_error_summary.txt"), "w") as f:
                    f.write(f"Least L2 Error = {least_l2_error}\n")
                    f.write(f"Least L2 Error Epoch = {least_l2_error_epoch}\n")
                    f.write(f"Linf at Min = {linf_at_min}\n")
                    f.write(f"L1 at Min = {l1_at_min}\n")
                    f.write(f"L2 Relative at Min = {l2_relative_at_min}\n")
                    f.write(f"Linf Relative at Min = {linf_relative_at_min}\n")
                    f.write(f"L1 Relative at Min = {l1_relative_at_min}\n")
                    f.write(f"PDE Loss at Min = {pde_loss_at_min}\n")

    

    # close the progress bar
    progress_bar.close()

    if i_mesh_generation_method == "internal":
        X = test_points[:,0].reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:,1].reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Pred_Matrix = least_l2_error_pred.reshape(i_n_test_points_x, i_n_test_points_y)
        SUPG_Pred_Matrix = least_l2_error_tau.reshape(i_n_test_points_x, i_n_test_points_y)
        # plot the final prediction
        plot_contour(x = X, y = Y, z = Y_Pred_Matrix, output_path = i_output_path, filename= "min_error_prediction", title = "")
        # plot the final error
        plot_contour(x = X, y = Y, z = np.abs(Y_Exact_Matrix - Y_Pred_Matrix), output_path = i_output_path, filename= "min_error", title = "")
        plot_contour(x = X, y = Y, z = SUPG_Pred_Matrix, output_path = i_output_path, filename= "min_error_SUPG", title = "")

        # print the Error values in table
    print_table("Error Values", ["Error Type", "Value"], \
                ["L2 Error", "Linf Error", "Relative L2 Error", "Relative Linf Error", "L1 Error", "Relative L1 Error"], \
                [least_l2_error, linf_at_min, l2_relative_at_min, linf_relative_at_min, l1_at_min, l1_relative_at_min])

    # print the time values in table
    print_table("Time Values", ["Time Type", "Value"], \
                ["Time per Epoch(s) - Median",   "Time per Epoch(s) IQR-25% ",   "Time per Epoch(s) IQR-75% ",  \
                "Mean (s)", "Epochs per second" , "Total Train Time"], \
                [np.median(time_array), np.percentile(time_array, 25),\
                np.percentile(time_array, 75), 
                np.mean(time_array),
                int(i_epochs/np.sum(time_array)) , np.sum(time_array) ])
    
    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    
    
    print("[INFO] Model Saved Successfully")

    # predict the values
    y_pred = model(test_points)[:,0:1]
    y_pred = y_pred * aa
    y_pred = y_pred.numpy().reshape(-1,)

    # plot the loss function, prediction, exact solution and error
    plot_loss_function(loss_array, i_output_path)
    
    # get errors
    l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)
    

    solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
    
    if i_mesh_generation_method == "internal":
        # reshape y_pred into a 2D array
        y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
        
        #Error
        error = np.abs(Y_Exact_Matrix - y_pred)
        
        # plot the prediction
        plot_contour(x = X, y = Y, z = y_pred, output_path = i_output_path, filename= f"final_prediction", title = "Prediction")
        # plot the error
        plot_contour(x = X, y = Y, z = error, output_path = i_output_path, filename= f"final_error", title = "Error")


    elif i_mesh_generation_method == "external":
        solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
        error = np.abs(y_exact - y_pred)
        domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )


    # domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )

    
    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))


    # copy the input file to the output folder
    os.system(f"cp {sys.argv[1]} {i_output_path}")