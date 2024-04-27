"""
file: model_inverse_domain.py
description: This file hosts the Neural Network (NN) model and the training loop for variational Physics-Informed Neural Networks (PINNs).
             This focuses on training variational PINNs for inverse problems where the inverse parameter is constant over the entire domain.
             The focus is on the model architecture and the training loop, and not on the loss functions.
authors: Thivin Anandh D
changelog: 22/Sep/2023 - Initial implementation with basic model architecture and training loop
known_issues: None
dependencies: None specified.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy


# Custom Model
class DenseModel_Inverse_Domain(tf.keras.Model):
    """
    A subclass of tf.keras.Model that defines a dense model for an inverse problem.

    :param list layer_dims: The dimensions of the layers in the model.
    :param dict learning_rate_dict: A dictionary containing the learning rates.
    :param dict params_dict: A dictionary containing the parameters of the model.
    :param function loss_function: The loss function to be used in the model.
    :param list input_tensors_list: A list of input tensors.
    :param list orig_factor_matrices: The original factor matrices.
    :param list force_function_list: A list of force functions.
    :param list sensor_list: A list of sensors for the inverse problem.
    :param dict inverse_params_dict: A dictionary containing the parameters for the inverse problem.
    :param tf.DType tensor_dtype: The data type of the tensors.
    :param bool use_attention: Whether to use attention mechanism in the model. Defaults to False.
    :param str activation: The activation function to be used in the model. Defaults to 'tanh'.
    :param bool hessian: Whether to use Hessian in the model. Defaults to False.
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
        sensor_list,  # for inverse problem
        tensor_dtype,
        use_attention=False,
        activation="tanh",
        hessian=False,
    ):
        super(DenseModel_Inverse_Domain, self).__init__()
        self.layer_dims = layer_dims
        self.use_attention = use_attention
        self.activation = activation
        self.layer_list = []
        self.loss_function = loss_function
        self.hessian = hessian

        self.tensor_dtype = tensor_dtype

        self.sensor_list = sensor_list
        # obtain sensor values
        self.sensor_points = sensor_list[0]
        self.sensor_values = sensor_list[1]

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
        # staircase = learning_rate_dict["staircase"]

        if use_lr_scheduler:
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=True
            )
        else:
            learning_rate_fn = initial_learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        # build the model using the input shape of the first layer in self.layer_dims
        input_shape = (None, self.layer_dims[0])
        # build the model
        self.build(input_shape=input_shape)
        # Compile the model
        self.compile(optimizer=self.optimizer)
        # print model summary
        self.summary()

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
        """
        Get the configuration of the model.

        Returns:
            dict: The configuration of the model.
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
                "sensor_list": self.sensor_list,
            }
        )

        return base_config

    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None):  # pragma: no cover
        """
        The train step method for the model.

        :param beta: The beta parameter for the training step, defaults to 10.
        :type beta: int, optional
        :param bilinear_params_dict: The dictionary containing the bilinear parameters, defaults to None.
        :type bilinear_params_dict: dict, optional
        :return: The output of the training step.
        :rtype: varies based on implementation
        """

        with tf.GradientTape(persistent=True) as tape:
            # Predict the values for dirichlet boundary conditions
            predicted_values_dirichlet = self(self.dirichlet_input)
            # reshape the predicted values to (, 1)
            predicted_values_dirichlet = tf.reshape(predicted_values_dirichlet[:, 0], [-1, 1])

            # predict the sensor values
            predicted_sensor_values = self(self.sensor_points)
            # reshape the predicted values to (, 1)
            predicted_sensor_values = tf.reshape(predicted_sensor_values[:, 0], [-1, 1])

            # initialize total loss as a tensor with shape (1,) and value 0.0
            total_pde_loss = 0.0

            with tf.GradientTape(persistent=True) as tape1:
                # tape gradient
                tape1.watch(self.input_tensor)
                # Compute the predicted values from the model
                predicted_values_actual = self(self.input_tensor)

                predicted_values = predicted_values_actual[:, 0]
                inverse_param_values = predicted_values_actual[:, 1]

            # compute the gradients of the predicted values wrt the input which is (x, y)
            # First column of the predicted values is the predicted value of the PDE
            gradients = tape1.gradient(predicted_values, self.input_tensor)

            # obtain inverse param gradients
            inverse_param_gradients = tape1.gradient(inverse_param_values, self.input_tensor)

            # Split the gradients into x and y components and reshape them to (-1, 1)
            # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_grad_x = tf.reshape(
                gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(
                gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            # First column of the predicted values is the predicted value of the PDE and reshape it to (N_cells, N_quadrature_points)
            pred_val = tf.reshape(
                predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            # reshape the second column of the predicted value and reshape it to (N_cells, N_quadrature_points)
            inverse_param_values = tf.reshape(
                inverse_param_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
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
                inverse_params_list=[inverse_param_values],
            )

            residual = tf.reduce_sum(cells_residual)

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

            # print shapes of the predicted values and the actual values
            boundary_loss = tf.reduce_mean(
                tf.square(predicted_values_dirichlet - self.dirichlet_actual), axis=0
            )

            # Sensor loss
            sensor_loss = tf.reduce_mean(
                tf.square(predicted_sensor_values - self.sensor_values), axis=0
            )

            # Compute Total Loss
            total_loss = total_pde_loss + beta * boundary_loss + 10 * sensor_loss

        trainable_vars = self.trainable_variables
        self.gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {
            "loss_pde": total_pde_loss,
            "loss_dirichlet": boundary_loss,
            "loss": total_loss,
            "sensor_loss": sensor_loss,
        }
