# This file contains the loss function for the NSE2d problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_nse2d_inverse(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params,
    penalty_coefficients_dict=None,
    inverse_params_dict=None,
):
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    test_velocity = test_shape_val_mat[0]  # φ
    test_pressure = test_shape_val_mat[1]  # Ψ
    quad_weights_velocity = test_shape_val_mat[2]  # w_φ # used for pressure correction

    test_velocity_grad_x = test_grad_x_mat[0]  # dφ/dx
    test_pressure_grad_x = test_grad_x_mat[1]  # dΨ/dx

    test_velocity_grad_y = test_grad_y_mat[0]  # dφ/dy
    test_pressure_grad_y = test_grad_y_mat[1]  # dΨ/dy

    u_nn = pred_nn[0]  # u
    v_nn = pred_nn[1]  # v
    p_nn = pred_nn[2]  # p

    u_grad_x_nn = pred_grad_x_nn[0]  # du/dx
    v_grad_x_nn = pred_grad_x_nn[1]  # dv/dx
    p_grad_x_nn = pred_grad_x_nn[2]  # dp/dx

    u_grad_y_nn = pred_grad_y_nn[0]  # du/dy
    v_grad_y_nn = pred_grad_y_nn[1]  # dv/dy
    p_grad_y_nn = pred_grad_y_nn[2]  # dp/dy

    if penalty_coefficients_dict is not None:
        residual_u_penalty = penalty_coefficients_dict["residual_u"]
        residual_v_penalty = penalty_coefficients_dict["residual_v"]
        divergence_penalty = penalty_coefficients_dict["divergence"]
    else:
        residual_u_penalty = 1e-4
        residual_v_penalty = 1e-4
        divergence_penalty = 1e4

    # 1/Re * ∫ (du/dx. dφ/dx + du/dy. dφ/dy) dΩ
    diffusion_x = (
        1.0
        / inverse_params_dict["re_nr"]
        * (
            tf.transpose(tf.linalg.matvec(test_velocity_grad_x, u_grad_x_nn))
            + tf.transpose(tf.linalg.matvec(test_velocity_grad_y, u_grad_y_nn))
        )
    )

    # ∫u * du/dx. φ + v du/dy. φ dΩ
    conv_x = tf.transpose(tf.linalg.matvec(test_velocity, u_grad_x_nn * u_nn)) + tf.transpose(
        tf.linalg.matvec(test_velocity, u_grad_y_nn * v_nn)
    )

    # ∫p dφ/dx dΩ
    pressure_x = tf.transpose(tf.linalg.matvec(test_velocity_grad_x, p_nn))

    # 1/Re * ∫ (dv/dx. dφ/dx + dv/dy. dφ/dy) dΩ
    diffusion_y = (
        1.0
        / inverse_params_dict["re_nr"]
        * (
            tf.transpose(tf.linalg.matvec(test_velocity_grad_x, v_grad_x_nn))
            + tf.transpose(tf.linalg.matvec(test_velocity_grad_y, v_grad_y_nn))
        )
    )

    # ∫u * dv/dx. φ + v dv/dy. φ dΩ
    conv_y = tf.transpose(tf.linalg.matvec(test_velocity, v_grad_x_nn * u_nn)) + tf.transpose(
        tf.linalg.matvec(test_velocity, v_grad_y_nn * v_nn)
    )

    # ∫p dφ/dy dΩ
    pressure_y = tf.transpose(tf.linalg.matvec(test_velocity_grad_y, p_nn))

    # ∫(du/dx)φ   +  ∫(dv/dy)φ dΩ ---> Div(u)
    divergence = tf.transpose(tf.linalg.matvec(test_pressure, u_grad_x_nn)) + tf.transpose(
        tf.linalg.matvec(test_pressure, v_grad_y_nn)
    )

    # X component of the residual
    residual_x = diffusion_x + conv_x - pressure_x
    # residual_x = diffusion_x  - pressure_x
    residual_x = tf.reduce_mean(tf.square(residual_x), axis=0)

    # Y component of the residual
    # residual_y = diffusion_y  - pressure_y
    residual_y = diffusion_y + conv_y - pressure_y
    residual_y = tf.reduce_mean(tf.square(residual_y), axis=0)

    # divergence residual
    divergence = tf.reduce_mean(tf.square(divergence), axis=0)

    # Perform Reduce mean along the axis 0
    residual_cells = (
        residual_u_penalty * residual_x
        + residual_v_penalty * residual_y
        + divergence_penalty * divergence
    )

    return (
        residual_cells,
        tf.reduce_sum(divergence),
        tf.reduce_sum(residual_x),
        tf.reduce_sum(residual_y),
    )
