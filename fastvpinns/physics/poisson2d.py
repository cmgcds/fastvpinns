"""
This file `poisson2d.py` is implementation of our efficient tensor-based loss calculation for poisson equation

Author: Thivin Anandh D

Date: 21/Sep/2023

History: Initial implementation

Refer: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_poisson(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params,
):  # pragma: no cover
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells
