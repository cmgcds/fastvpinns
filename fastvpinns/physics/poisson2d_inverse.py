# This file contains the loss function for the poisson problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf


# PDE loss function for the poisson problem
# @tf.function
def pde_loss_poisson_inverse(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params,
    inverse_params_dict,
):
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # Compute PDE loss
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # print
    # tf.print("pde_diffusion_x = ", pde_diffusion_x.shape)

    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # tf.print("pde_diffusion_y = ", pde_diffusion_y.shape)

    pde_diffusion = inverse_params_dict["eps"] * (pde_diffusion_x + pde_diffusion_y)

    # tf.print("pde_diffusion = ", pde_diffusion.shape)

    # tf.print("pde_diffusion = ", pde_diffusion)

    residual_matrix = pde_diffusion - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells
