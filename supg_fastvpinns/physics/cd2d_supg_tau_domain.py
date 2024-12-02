# This file contains the loss function for the poisson problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# PDE loss function for the poisson problem
@tf.function
def pde_loss_cd2d(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, \
                    pred_grad_grad_x_nn, pred_grad_grad_y_nn, \
                    forcing_function, real_forcing_function, bilinear_params, supg_tau):
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # Loss Function : ∫du/dx. dv/dx  +  ∫du/dy. dv/dy - ∫f.v
    
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

   

    # ∫du/dx. v dΩ
    conv_x = tf.transpose(tf.linalg.matvec(test_shape_val_mat, pred_grad_x_nn))
    
    # # ∫du/dy. v dΩ
    conv_y = tf.transpose(tf.linalg.matvec(test_shape_val_mat, pred_grad_y_nn))

    # # b(x) * ∫du/dx. v dΩ + b(y) * ∫du/dy. v dΩ
    conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    # c∫u. v dΩ
    pde_reaction = bilinear_params["c"]*tf.transpose(tf.linalg.matvec(test_shape_val_mat, pred_nn))


    # eps(d2u/dx2 + d2u/dy2)
    # double_xx_yy_u = bilinear_params["eps"]*(pred_grad_xx_nn + pred_grad_yy_nn)

    # b_x*dv/dx + b_y*dv/dy
    single_test_x_y_v = bilinear_params["b_x"]* test_grad_x_mat + bilinear_params["b_y"]* test_grad_y_mat

    # b_x*du/dx + b_y*du/dy
    #element wise multiplication of supg_tau and pred_grad_x_nn
    tau_pred_grad_x_nn = supg_tau*pred_grad_x_nn
    tau_pred_grad_y_nn = supg_tau*pred_grad_y_nn

    single_x_y_u = bilinear_params["b_x"]* tau_pred_grad_x_nn + bilinear_params["b_y"]* tau_pred_grad_y_nn

    pred_grad_grad_x_nn = supg_tau*pred_grad_grad_x_nn
    pred_grad_grad_y_nn = supg_tau*pred_grad_grad_y_nn
    double_deriv_sum = pred_grad_grad_x_nn + pred_grad_grad_y_nn

    # SUPG diffusion ∫(d^2u/dx^2 +d^2u/dy^2) . (b_x*du/dx + b_y*du/dy) dΩ
    supg_diffusion = -bilinear_params["eps"]*tf.transpose(tf.linalg.matvec(single_test_x_y_v, double_deriv_sum))


    #SUPG reaction
    tau_pred_nn = supg_tau*pred_nn
    supg_reaction = bilinear_params["c"]*tf.transpose(tf.linalg.matvec(single_test_x_y_v, tau_pred_nn))

    # supg_residual_matrix = tf.transpose(tf.linalg.matvec(single_test_x_y_v, double_xx_yy_u)) 

    supg_residual_matrix =  tf.transpose(tf.linalg.matvec(single_test_x_y_v, single_x_y_u)) 
    # # print(real_forcing_function.shape)
    supg_forcing_function = supg_tau*real_forcing_function

    supg_forcing = tf.transpose(tf.linalg.matvec(single_test_x_y_v, supg_forcing_function))
    
    # tau = 0.3*(1/real_forcing_function.shape[0])*2

    # tau = 0.02

    # residual_matrix = (pde_diffusion + conv + pde_reaction) - forcing_function + supg_tau*(supg_residual_matrix-supg_forcing + supg_reaction) # + supg_diffusion + supg_reaction) 

    pde_loss = (pde_diffusion + conv + pde_reaction) - forcing_function
    supg_loss = (supg_residual_matrix-supg_forcing + supg_reaction + supg_diffusion)
    residual_matrix = pde_loss + supg_loss
    # Perform Reduce mean along the axis 0
    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)
    pde_loss = tf.reduce_mean(tf.square(pde_loss), axis=0)
    supg_loss = tf.reduce_mean(tf.square(supg_loss), axis=0)


    # print(single_test_x_y_v.shape, single_x_y_u.shape, forcing_function.shape) 
    # print(real_forcing_function.shape, supg_forcing.shape)
    # exit(1)
    return residual_cells, pde_loss, supg_loss
