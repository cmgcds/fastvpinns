"""
This function is implemntation of our efficient tensor-based loss calculation for poisson equation with inverse problem (constant)
Author: Thivin Anandh D
Date: 21-Sep-2023
History: Initial implementation
Refer: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


# PDE loss function for the poisson problem inverse
# @tf.function - Commented due to compatibility issues
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
):  # pragma: no cover
    """
    Calculates and returns the loss for the  Poisson problem Inverse (constant)

    :param test_shape_val_mat: The test shape value matrix.
    :type test_shape_val_mat: tf.Tensor
    :param test_grad_x_mat: The x-gradient of the test matrix.
    :type test_grad_x_mat: tf.Tensor
    :param test_grad_y_mat: The y-gradient of the test matrix.
    :type test_grad_y_mat: tf.Tensor
    :param pred_nn: The predicted neural network output.
    :type pred_nn: tf.Tensor
    :param pred_grad_x_nn: The x-gradient of the predicted neural network output.
    :type pred_grad_x_nn: tf.Tensor
    :param pred_grad_y_nn: The y-gradient of the predicted neural network output.
    :type pred_grad_y_nn: tf.Tensor
    :param forcing_function: The forcing function used in the PDE.
    :type forcing_function: function
    :param bilinear_params_dict: The dictionary containing the bilinear parameters.
    :type bilinear_params_dict: dict
    :param inverse_param_dict: The dictionary containing the parameters for the inverse problem.
    :type inverse_param_dict: dict

    :return: The calculated loss.
    :rtype: tf.Tensor
    """
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = inverse_params_dict["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells
