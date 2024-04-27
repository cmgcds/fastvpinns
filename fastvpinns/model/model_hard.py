"""
file: model_hard.py
description: This file contains the DenseModel class which is a custom model for the Neural Network
                for solving Variational PINNs. This model is used for enforcing hard boundary constraints
                on the solution.
author: Thivin Anandh D, Divij Ghose, Sashikumaar Ganesan
date: 22/01/2024
changelog: 22/01/2024 - file created
           22/01/2024 - 

known issues: None
"""

import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers


# Custom Model
class DenseModel_Hard(tf.keras.Model):
    """The DenseModel_Hard class is a custom model class that hosts the neural network model.

    The class inherits from the tf.keras.Model class and is used
    to define the neural network model architecture and the training loop for FastVPINNs.
    :param layer_dims: List of integers representing the number of neurons in each layer
    :type layer_dims: list
    :param learning_rate_dict: Dictionary containing the learning rate parameters
    :type learning_rate_dict: dict
    :param params_dict: Dictionary containing the parameters for the model
    :type params_dict: dict
    :param loss_function: Loss function for the model
    :type loss_function: function
    :param input_tensors_list: List of input tensors for the model
    :type input_tensors_list: list
    :param orig_factor_matrices: List of original factor matrices
    :type orig_factor_matrices: list
    :param force_function_list: List of force functions
    :type force_function_list: list
    :param tensor_dtype: Tensor data type
    :type tensor_dtype: tf.DType
    :param use_attention: Flag to use attention layer
    :type use_attention: bool
    :param activation: Activation function for the model
    :type activation: str
    :param hessian: Flag to compute hessian
    :type hessian: bool

    Methods
    -------
    call(inputs)
        This method is used to define the forward pass of the model.
    get_config()
        This method is used to get the configuration of the model.
    train_step(beta=10, bilinear_params_dict=None)
        This method is used to define the training step of the model.

    """

    def __init__(
        self,
        layer_dims,
        learning_rate_dict,
        params_dict,
        loss_function,
        input_tensors_list,
        orig_factor_matrices,
        force_function_list,
        tensor_dtype,
        use_attention=False,
        activation="tanh",
        hessian=False,
        hard_constraint_function=None,
    ):
        super(DenseModel_Hard, self).__init__()
        self.layer_dims = layer_dims
        self.use_attention = use_attention
        self.activation = activation
        self.layer_list = []
        self.loss_function = loss_function
        self.hessian = hessian
        if hard_constraint_function is None:
            self.hard_constraint_function = lambda x, y: y
        else:
            self.hard_constraint_function = hard_constraint_function

        self.tensor_dtype = tensor_dtype

        # if dtype is not a valid tensorflow dtype, raise an error
        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        self.orig_factor_matrices = orig_factor_matrices
        self.shape_function_mat_list = copy.deepcopy(orig_factor_matrices[0])
        self.shape_function_grad_x_factor_mat_list = copy.deepcopy(orig_factor_matrices[1])
        self.shape_function_grad_y_factor_mat_list = copy.deepcopy(orig_factor_matrices[2])

        self.force_function_list = force_function_list

        self.input_tensors_list = input_tensors_list
        self.input_tensor = copy.deepcopy(input_tensors_list[0])
        self.dirichlet_input = copy.deepcopy(input_tensors_list[1])
        self.dirichlet_actual = copy.deepcopy(input_tensors_list[2])

        self.params_dict = params_dict

        self.pre_multiplier_val = self.shape_function_mat_list
        self.pre_multiplier_grad_x = self.shape_function_grad_x_factor_mat_list
        self.pre_multiplier_grad_y = self.shape_function_grad_y_factor_mat_list

        self.force_matrix = self.force_function_list

        self.gradients = None

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
        print(
            f"| {'dirichlet_input':<25} | {str(self.dirichlet_input.shape):<25} | {self.dirichlet_input.dtype}"
        )
        print(
            f"| {'dirichlet_actual':<25} | {str(self.dirichlet_actual.shape):<25} | {self.dirichlet_actual.dtype}"
        )
        print(f"{'-'*74}")

        self.n_cells = params_dict["n_cells"]

        ## ----------------------------------------------------------------- ##
        ## ---------- LEARNING RATE AND OPTIMISER FOR THE MODEL ------------ ##
        ## ----------------------------------------------------------------- ##

        # parse the learning rate dictionary
        self.learning_rate_dict = learning_rate_dict
        initial_learning_rate = learning_rate_dict["initial_learning_rate"]
        use_lr_scheduler = learning_rate_dict["use_lr_scheduler"]
        decay_steps = learning_rate_dict["decay_steps"]
        decay_rate = learning_rate_dict["decay_rate"]
        staircase = learning_rate_dict["staircase"]

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
        """This method is used to define the forward pass of the model.
        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :return: Output tensor from the model
        :rtype: tf.Tensor
        """
        x = inputs

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        x = self.hard_constraint_function(inputs, x)

        return x

    def get_config(self):
        """This method is used to get the configuration of the model.
        :return: Configuration of the model
        :rtype: dict
        """
        # Get the base configuration
        base_config = super().get_config()

        # Add the non-serializable arguments to the configuration
        base_config.update(
            {
                "learning_rate_dict": self.learning_rate_dict,
                "loss_function": self.loss_function,
                "input_tensors_list": self.input_tensors_list,
                "orig_factor_matrices": self.orig_factor_matrices,
                "force_function_list": self.force_function_list,
                "params_dict": self.params_dict,
                "use_attention": self.use_attention,
                "activation": self.activation,
                "hessian": self.hessian,
                "layer_dims": self.layer_dims,
                "tensor_dtype": self.tensor_dtype,
            }
        )

        return base_config

    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None):  # pragma: no cover
        """This method is used to define the training step of the mode.
        :param bilinear_params_dict: Dictionary containing the bilinear parameters
        :type bilinear_params_dict: dict
        :return: Dictionary containing the loss values
        :rtype: dict
        """

        with tf.GradientTape(persistent=True) as tape:
            # Predict the values for dirichlet boundary conditions

            # initialize total loss as a tensor with shape (1,) and value 0.0
            total_pde_loss = 0.0

            with tf.GradientTape(persistent=True) as tape1:
                # tape gradient
                tape1.watch(self.input_tensor)
                # Compute the predicted values from the model
                predicted_values = self(self.input_tensor)

            # compute the gradients of the predicted values wrt the input which is (x, y)
            gradients = tape1.gradient(predicted_values, self.input_tensor)

            # Split the gradients into x and y components and reshape them to (-1, 1)
            # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_grad_x = tf.reshape(
                gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(
                gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            pred_val = tf.reshape(
                predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

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

            # tf.print("Residual : ", residual)
            # tf.print("Residual Shape : ", residual.shape)

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

            # convert predicted_values_dirichlet to tf.float64
            # predicted_values_dirichlet = tf.cast(predicted_values_dirichlet, tf.float64)

            # tf.print("Boundary Loss : ", boundary_loss)
            # tf.print("Boundary Loss Shape : ", boundary_loss.shape)
            # tf.print("Total PDE Loss : ", total_pde_loss)
            # tf.print("Total PDE Loss Shape : ", total_pde_loss.shape)
            boundary_loss = 0.0
            # Compute Total Loss
            total_loss = total_pde_loss

        trainable_vars = self.trainable_variables
        self.gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {"loss_pde": total_pde_loss, "loss_dirichlet": boundary_loss, "loss": total_loss}
