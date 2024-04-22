# This file contains the loss function for the NSE2d problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_burgers2d(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params,
):
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    test_velocity = test_shape_val_mat[0]  # φ

    test_velocity_grad_x = test_grad_x_mat[0]  # dφ/dx

    test_velocity_grad_y = test_grad_y_mat[0]  # dφ/dy

    u_nn = pred_nn[0]  # u
    v_nn = pred_nn[1]  # v

    u_grad_x_nn = pred_grad_x_nn[0]  # du/dx
    v_grad_x_nn = pred_grad_x_nn[1]  # dv/dx

    u_grad_y_nn = pred_grad_y_nn[0]  # du/dy
    v_grad_y_nn = pred_grad_y_nn[1]  # dv/dy

    # 1/Re * ∫ (du/dx. dφ/dx + du/dy. dφ/dy) dΩ
    diffusion_x = (
        1.0
        / bilinear_params["re_nr"]
        * (
            tf.transpose(tf.linalg.matvec(test_velocity_grad_x, u_grad_x_nn))
            + tf.transpose(tf.linalg.matvec(test_velocity_grad_y, u_grad_y_nn))
        )
    )

    # ∫u * du/dx. φ + v du/dy. φ dΩ
    conv_x = tf.transpose(
        tf.linalg.matvec(test_velocity, u_grad_x_nn * u_nn)
    ) + tf.transpose(tf.linalg.matvec(test_velocity, u_grad_y_nn * v_nn))

    # 1/Re * ∫ (dv/dx. dφ/dx + dv/dy. dφ/dy) dΩ
    diffusion_y = (
        1.0
        / bilinear_params["re_nr"]
        * (
            tf.transpose(tf.linalg.matvec(test_velocity_grad_x, v_grad_x_nn))
            + tf.transpose(tf.linalg.matvec(test_velocity_grad_y, v_grad_y_nn))
        )
    )

    # ∫u * dv/dx. φ + v dv/dy. φ dΩ
    conv_y = tf.transpose(
        tf.linalg.matvec(test_velocity, v_grad_x_nn * u_nn)
    ) + tf.transpose(tf.linalg.matvec(test_velocity, v_grad_y_nn * v_nn))

    # X component of the residual
    residual_x = diffusion_x + conv_x - forcing_function[0]
    residual_x = tf.reduce_mean(tf.square(residual_x), axis=0)

    # Y component of the residual
    residual_y = diffusion_y + conv_y - forcing_function[1]
    residual_y = tf.reduce_mean(tf.square(residual_y), axis=0)

    # Perform Reduce mean along the axis 0
    residual_cells = residual_x + residual_y

    return residual_cells
