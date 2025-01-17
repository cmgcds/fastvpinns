# A method which hosts the NN model and the training loop for variational Pinns
# This focusses only on the model architecture and the training loop, and not on the loss functions
# Author : Thivin Anandh D
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic model architecture and training loop

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy
import numpy as np


# Custom Loss Functions
def custom_loss1(y_true1, y_pred1):
    return tf.reduce_mean(tf.square(y_pred1 - y_true1))


def custom_loss2(y_true2, y_pred2):
    return tf.reduce_mean(tf.square(y_pred2 - y_true2))


# Custom Model
class DenseModelDomainDecomposition(tf.keras.Model):
    """
    This class defines the Dense Model for the Neural Network for solving Variational PINNs

    Attributes:
    - layer_dims (list): List of dimensions of the dense layers
    - use_attention (bool): Flag to use attention layer after input
    - activation (str): The activation function to be used for the dense layers
    - layer_list (list): List of dense layers
    - loss_function (function): The loss function for the PDE
    - hessian (bool): Flag to use hessian loss
    - input_tensor (tf.Tensor): The input tensor for the PDE
    - dirichlet_input (tf.Tensor): The input tensor for the Dirichlet boundary data
    - dirichlet_actual (tf.Tensor): The actual values for the Dirichlet boundary data
    - optimizer (tf.keras.optimizers): The optimizer for the model
    - gradients (tf.Tensor): The gradients of the loss function wrt the trainable variables
    - learning_rate_dict (dict): The dictionary containing the learning rate parameters
    - orig_factor_matrices (list): The list containing the original factor matrices
    - shape_function_mat_list (tf.Tensor): The shape function matrix
    - shape_function_grad_x_factor_mat_list (tf.Tensor): The shape function derivative with respect to x matrix
    - shape_function_grad_y_factor_mat_list (tf.Tensor): The shape function derivative with respect to y matrix
    - force_function_list (tf.Tensor): The force function matrix
    - input_tensors_list (list): The list containing the input tensors
    - params_dict (dict): The dictionary containing the parameters
    - pre_multiplier_val (tf.Tensor): The pre-multiplier for the shape function matrix
    - pre_multiplier_grad_x (tf.Tensor): The pre-multiplier for the shape function derivative with respect to x matrix
    - pre_multiplier_grad_y (tf.Tensor): The pre-multiplier for the shape function derivative with respect to y matrix
    - force_matrix (tf.Tensor): The force function matrix
    - n_cells (int): The number of cells in the domain
    - tensor_dtype (tf.DType): The tensorflow dtype to be used for all the tensors

    Methods:
    - call(inputs): The call method for the model
    - get_config(): Returns the configuration of the model
    - train_step(beta, bilinear_params_dict): The train step method for the model
    """

    def __init__(
        self,
        layer_dims,
        learning_rate_dict,
        subdomain_id,
        decomposed_domain,
        loss_function,
        datahandler,
        use_attention=False,
        activation='tanh',
        hessian=False,
    ):
        super(DenseModelDomainDecomposition, self).__init__()
        self.layer_dims = layer_dims
        self.use_attention = use_attention
        self.activation = activation
        self.layer_list = []
        self.loss_function = loss_function
        self.hessian = hessian
        self.subdomain_id = subdomain_id

        self.tensor_dtype = datahandler.dtype

        self.window_func_vals = decomposed_domain.window_function_values[self.subdomain_id]
        self.x_min_limit = decomposed_domain.subdomain_boundary_limits[self.subdomain_id][0]
        self.x_max_limit = decomposed_domain.subdomain_boundary_limits[self.subdomain_id][1]
        self.y_min_limits = decomposed_domain.subdomain_boundary_limits[self.subdomain_id][2]
        self.y_max_limits = decomposed_domain.subdomain_boundary_limits[self.subdomain_id][3]

        self.mean_x = (self.x_max_limit + self.x_min_limit) / 2.0
        self.std_x = (self.x_max_limit - self.x_min_limit) / 2.0

        self.mean_y = (self.y_max_limits + self.y_min_limits) / 2.0
        self.std_y = (self.y_max_limits - self.y_min_limits) / 2.0

        # if dtype is not a valid tensorflow dtype, raise an error
        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        self.shape_function_mat_list = copy.deepcopy(datahandler.shape_val_mat_list)
        self.shape_function_grad_x_factor_mat_list = copy.deepcopy(datahandler.grad_x_mat_list)
        self.shape_function_grad_y_factor_mat_list = copy.deepcopy(datahandler.grad_y_mat_list)

        self.force_function_list = datahandler.forcing_function_list

        self.input_tensor = copy.deepcopy(datahandler.x_pde_list)
        # self.input_tensor = t3f.to_tt_tensor(self.input_tensor, max_tt_rank=4)

        self.params_dict = decomposed_domain.params_dict[self.subdomain_id]

        self.pre_multiplier_val = self.shape_function_mat_list
        self.pre_multiplier_grad_x = self.shape_function_grad_x_factor_mat_list
        self.pre_multiplier_grad_y = self.shape_function_grad_y_factor_mat_list

        self.force_matrix = self.force_function_list

        self.unnormalizing_factor = decomposed_domain.unnormalizing_factor
        self.hard_constraints_factor = decomposed_domain.hard_constraints_factor

        print(f"{'-'*74}")
        print(f"| {'PARAMETER':<25} | {'SHAPE':<25} |")
        print(f"{'-'*74}")
        print(
            f"| {'input_tensor':<25} | {str(self.input_tensor.shape):<25} | {self.input_tensor.dtype}"
        )
        print(
            f"| {'force_matrix':<25} | {str(self.force_matrix.shape):<25} | {self.force_matrix.dtype}"
        )
        print(
            f"| {'pre_multiplier_grad_x':<25} | {str(self.pre_multiplier_grad_x.shape):<25} | {self.pre_multiplier_grad_x.dtype}"
        )
        print(
            f"| {'pre_multiplier_grad_y':<25} | {str(self.pre_multiplier_grad_y.shape):<25} | {self.pre_multiplier_grad_y.dtype}"
        )
        print(
            f"| {'pre_multiplier_val':<25} | {str(self.pre_multiplier_val.shape):<25} | {self.pre_multiplier_val.dtype}"
        )
        print(f"{'-'*74}")

        self.n_cells = self.params_dict['n_cells']

        ## ----------------------------------------------------------------- ##
        ## ---------- LEARNING RATE AND OPTIMISER FOR THE MODEL ------------ ##
        ## ----------------------------------------------------------------- ##

        # parse the learning rate dictionary
        self.learning_rate_dict = learning_rate_dict
        initial_learning_rate = learning_rate_dict['initial_learning_rate']
        use_lr_scheduler = learning_rate_dict['use_lr_scheduler']
        decay_steps = learning_rate_dict['decay_steps']
        decay_rate = learning_rate_dict['decay_rate']
        staircase = learning_rate_dict['staircase']

        if use_lr_scheduler:
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=True
            )
        else:
            learning_rate_fn = initial_learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        ## ----------------------------------------------------------------- ##
        ## --------------------- MODEL ARCHITECTURE ------------------------ ##
        ## ----------------------------------------------------------------- ##

        # Build dense layers based on the input list
        for dim in range(len(self.layer_dims) - 2):
            self.layer_list.append(
                layers.Dense(
                    self.layer_dims[dim + 1],
                    activation=self.activation,
                    kernel_initializer="glorot_uniform",
                    dtype=self.tensor_dtype,
                    bias_initializer="zeros",
                )
            )

        # Add a output layer with no activation
        self.layer_list.append(
            layers.Dense(
                self.layer_dims[-1],
                activation=None,
                kernel_initializer="glorot_uniform",
                dtype=self.tensor_dtype,
                bias_initializer="zeros",
            )
        )

        # Add attention layer if required
        if self.use_attention:
            self.attention_layer = layers.Attention()

        # Compile the model
        self.compile(optimizer=self.optimizer)
        self.build(input_shape=(None, self.layer_dims[0]))

        # print the summary of the model
        self.summary()

    # def build(self, input_shape):
    #     super(DenseModel, self).build(input_shape)

    def call(self, inputs):
        """
        The call method for the model.

        :param inputs: The input tensor for the model.
        :type inputs: tf.Tensor
        :return: The output tensor of the model.
        :rtype: tf.Tensor
        """
        x = inputs

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        return x

    def get_config(self):
        # Get the base configuration
        base_config = super().get_config()

        # Add the non-serializable arguments to the configuration
        base_config.update(
            {
                'learning_rate_dict': self.learning_rate_dict,
                'loss_function': self.loss_function,
                'input_tensors_list': self.input_tensors_list,
                'orig_factor_matrices': self.orig_factor_matrices,
                'force_function_list': self.force_function_list,
                'params_dict': self.params_dict,
                'use_attention': self.use_attention,
                'activation': self.activation,
                'hessian': self.hessian,
                'layer_dims': self.layer_dims,
                'tensor_dtype': self.tensor_dtype,
            }
        )

        return base_config

    @tf.function
    def pretrain_step(self):
        """
        This function is used to compute the values and gradients of the subdomains and store them in the dictionary
        These values will be used in main train step as an update parameter
        """
        with tf.GradientTape(persistent=True) as pretrain_tape:
            # tape gradient
            pretrain_tape.watch(self.input_tensor)

            # Compute the predicted values from the model
            x_values = self.input_tensor[:, 0:1]
            y_values = self.input_tensor[:, 1:2]

            normalized_x_values = (x_values - self.mean_x) / self.std_x
            normalized_y_values = (y_values - self.mean_y) / self.std_y

            normalized_input_tensor = tf.concat([normalized_x_values, normalized_y_values], axis=1)

            predicted_values = self(normalized_input_tensor)

            # unnormalisation
            predicted_values = self.unnormalizing_factor * predicted_values

            predicted_values = predicted_values * self.window_func_vals

   

            # compute the gradients of the predicted values wrt the input which is (x, y)
        gradients = pretrain_tape.gradient(predicted_values, self.input_tensor)
        pred_grad_x = gradients[:, 0:1]
        pred_grad_y = gradients[:, 1:2]

        # Split the gradients into x and y components and reshape them to (-1, 1)
        # the reshaping is done for the tensorial operations purposes (refer Notebook)
        pred_grad_x = tf.reshape(
            gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
        )
        pred_grad_y = tf.reshape(
            gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
        )

        pred_val = tf.reshape(predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]])

        return pred_val, pred_grad_x, pred_grad_y

    @tf.function
    def train_step(
        self, bilinear_params_dict=None, overlap_val=None, overlap_grad_x=None, overlap_grad_y=None
    ):

        with tf.GradientTape(persistent=True) as optimizer_tape:
            total_pde_loss = 0.0
            with tf.GradientTape(persistent=True) as training_tape:
                # tape gradient
                training_tape.watch(self.input_tensor)

                # Compute the predicted values from the model
                x_values = self.input_tensor[:, 0:1]
                y_values = self.input_tensor[:, 1:2]

                normalized_x_values = (x_values - self.mean_x) / self.std_x
                normalized_y_values = (y_values - self.mean_y) / self.std_y


                normalized_input_tensor = tf.concat(
                    [normalized_x_values, normalized_y_values], axis=1
                )

                predicted_values = self(normalized_input_tensor)

                # unnormalisation - perform element wise multiplication with the scalar value
                predicted_values = self.unnormalizing_factor * predicted_values

                predicted_values = predicted_values * self.window_func_vals



            # compute the gradients of the predicted values wrt the input which is (x, y)
            gradients = training_tape.gradient(predicted_values, self.input_tensor)
            # tf.print("Shape of gradients : ", gradients.shape)
            pred_grad_x = gradients[:, 0:1]
            # tf.print("Shape of pred_grad_x : ", pred_grad_x.shape)
            pred_grad_y = gradients[:, 1:2]
            # tf.print("Shape of pred_grad_y : ", pred_grad_y.shape)


            # # Split the gradients into x and y components and reshape them to (-1, 1)
            # # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_grad_x = tf.reshape(
                pred_grad_x, [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(
                pred_grad_y, [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            pred_val = tf.reshape(
                predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            # # tf.print("Shape of pred_val Final : ", pred_val.shape)
            # # tf.print("Shape of pred_grad_x Final : ", pred_grad_x.shape)
            # # tf.print("Shape of pred_grad_y Final : ", pred_grad_y.shape)

            pred_val = pred_val + tf.stop_gradient(overlap_val)
            pred_grad_x = pred_grad_x + tf.stop_gradient(overlap_grad_x)
            pred_grad_y = pred_grad_y + tf.stop_gradient(overlap_grad_y)

            pred_val= tf.reshape(pred_val, [-1, 1])
            pred_grad_x = tf.reshape(pred_grad_x, [-1, 1])
            pred_grad_y = tf.reshape(pred_grad_y, [-1, 1])

            pred_val = (
                    tf.tanh((self.hard_constraints_factor) * x_values)
                    * tf.tanh((self.hard_constraints_factor) * y_values)
                    * tf.tanh((self.hard_constraints_factor) * (x_values - 1.0))
                    * tf.tanh((self.hard_constraints_factor) * (y_values - 1.0))
                    * pred_val
                )
            
            ansatz = tf.tanh((self.hard_constraints_factor) * x_values) * tf.tanh((self.hard_constraints_factor) * y_values) * tf.tanh((self.hard_constraints_factor) * (x_values - 1.0)) * tf.tanh((self.hard_constraints_factor) * (y_values - 1.0))

            c = self.hard_constraints_factor
            
            diff_ansatz_x = self.hard_constraints_factor * (1 - tf.tanh(self.hard_constraints_factor * x_values)**2) * tf.tanh(self.hard_constraints_factor * y_values) * tf.tanh(self.hard_constraints_factor * (1 - x_values)) * tf.tanh(self.hard_constraints_factor * (1 - y_values)) - self.hard_constraints_factor * (1 - tf.tanh(self.hard_constraints_factor * (1 - x_values))**2) * tf.tanh(self.hard_constraints_factor * x_values) * tf.tanh(self.hard_constraints_factor * y_values) * tf.tanh(self.hard_constraints_factor * (1 - y_values))
            

            diff_ansatz_y = self.hard_constraints_factor * (1 - tf.tanh(self.hard_constraints_factor * y_values)**2) * tf.tanh(self.hard_constraints_factor * x_values) * tf.tanh(self.hard_constraints_factor * (x_values - 1)) * tf.tanh(self.hard_constraints_factor * (y_values - 1)) + self.hard_constraints_factor * (1 - tf.tanh(self.hard_constraints_factor * (y_values - 1))**2) * tf.tanh(self.hard_constraints_factor * x_values) * tf.tanh(self.hard_constraints_factor * y_values) * tf.tanh(self.hard_constraints_factor * (x_values - 1))


            pred_grad_x = pred_grad_x * ansatz + pred_val * diff_ansatz_x
            pred_grad_y = pred_grad_y * ansatz + pred_val * diff_ansatz_y
            
            # pred_val = tf.reshape(pred_val, [self.n_cells, self.pre_multiplier_val.shape[-1]])
            # pred_grad_x = tf.reshape(pred_grad_x, [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])
            # pred_grad_y = tf.reshape(pred_grad_y, [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])

            # # # # sum the gradients with the overlap values
            # # # # the overlap values should not be involved in the gradient tape
            # cells_residual = self.loss_function(
            #     test_shape_val_mat=self.pre_multiplier_val,
            #     test_grad_x_mat=self.pre_multiplier_grad_x,
            #     test_grad_y_mat=self.pre_multiplier_grad_y,
            #     pred_nn=pred_val,
            #     pred_grad_x_nn=pred_grad_x,
            #     pred_grad_y_nn=pred_grad_y,
            #     forcing_function=self.force_matrix,
            #     bilinear_params=bilinear_params_dict,
            # )

            residual = pred_grad_x + pred_grad_y
            
            residual = tf.reduce_mean(tf.square(residual))

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

            # Compute Total Loss
            total_loss = total_pde_loss

        trainable_vars = self.trainable_variables
        self.gradients = optimizer_tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {"loss_pde": total_pde_loss, "loss": total_loss}

    @tf.function
    def train_step_new(
        self, bilinear_params_dict=None, overlap_val=None, overlap_grad_x=None, overlap_grad_y=None,
        solution_values=None, solution_grad_x=None, solution_grad_y=None
    ):

        with tf.GradientTape(persistent=True) as optimizer_tape:
            total_pde_loss = 0.0
            with tf.GradientTape(persistent=True) as training_tape:
                # tape gradient
                training_tape.watch(self.input_tensor)

                # Compute the predicted values from the model
                x_values = self.input_tensor[:, 0:1]
                y_values = self.input_tensor[:, 1:2]

                # normalized_x_values = (x_values - self.mean_x) / self.std_x
                # normalized_y_values = (y_values - self.mean_y) / self.std_y

                normalized_x_values = x_values
                normalized_y_values = y_values

                normalized_input_tensor = tf.concat(
                    [normalized_x_values, normalized_y_values], axis=1
                )

                predicted_values = self(normalized_input_tensor)

                # unnormalisation - perform element wise multiplication with the scalar value
                predicted_values = self.unnormalizing_factor * predicted_values

                predicted_values = predicted_values * self.window_func_vals

            # compute the gradients of the predicted values
            gradients = training_tape.gradient(predicted_values, self.input_tensor)
            pred_grad_x = gradients[:, 0:1]
            pred_grad_y = gradients[:, 1:2]

            
            # # Split the gradients into x and y components and reshape them to (-1, 1)
            # # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_grad_x = tf.reshape(
                pred_grad_x, [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(
                pred_grad_y, [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            pred_val = tf.reshape(
                predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            # # tf.print("Shape of pred_val Final : ", pred_val.shape)
            # # tf.print("Shape of pred_grad_x Final : ", pred_grad_x.shape)
            # # tf.print("Shape of pred_grad_y Final : ", pred_grad_y.shape)

            pred_val = pred_val + tf.stop_gradient(overlap_val)
            pred_grad_x = pred_grad_x + tf.stop_gradient(overlap_grad_x)
            pred_grad_y = pred_grad_y + tf.stop_gradient(overlap_grad_y)

            pred_val= tf.reshape(pred_val, [-1, 1])

            pred_val = (
                    tf.tanh((self.hard_constraints_factor) * x_values)
                    * tf.tanh((self.hard_constraints_factor) * y_values)
                    * tf.tanh((self.hard_constraints_factor) * (x_values - 1.0))
                    * tf.tanh((self.hard_constraints_factor) * (y_values - 1.0))
                    * pred_val
                )
            
            pred_val = tf.reshape(pred_val, [self.n_cells, self.pre_multiplier_val.shape[-1]])

            # # # sum the gradients with the overlap values
            # # # the overlap values should not be involved in the gradient tape
            cells_residual = self.loss_function(
                test_shape_val_mat=self.pre_multiplier_val,
                test_grad_x_mat=self.pre_multiplier_grad_x,
                test_grad_y_mat=self.pre_multiplier_grad_y,
                pred_nn=pred_val,
                pred_grad_x_nn=pred_grad_x,
                pred_grad_y_nn=pred_grad_y,
                forcing_function=self.force_matrix,
                bilinear_params=bilinear_params_dict,
            )

            residual = tf.reduce_sum(cells_residual)

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

            # Compute Total Loss
            total_loss = total_pde_loss

        trainable_vars = self.trainable_variables
        self.gradients = optimizer_tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {"loss_pde": total_pde_loss, "loss": total_loss}
