# A method which hosts the NN model and the training loop for variational Pinns
# This focusses only on the model architecture and the training loop, and not on the loss functions
# Author : Thivin Anandh D
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic model architecture and training loop

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy


# Custom Model
class DenseModel_Burgers(tf.keras.Model):
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
        params_dict,
        loss_function,
        input_tensors_list,
        orig_factor_matrices,
        force_function_list,
        dirichlet_list,
        tensor_dtype,
        use_attention=False,
        activation='tanh',
        hessian=False,
    ):
        super(DenseModel_Burgers, self).__init__()
        self.layer_dims = layer_dims
        self.use_attention = use_attention
        self.activation = activation
        self.layer_list = []
        self.loss_function = loss_function
        self.hessian = hessian

        self.tensor_dtype = tensor_dtype

        # if dtype is not a valid tensorflow dtype, raise an error
        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        self.orig_factor_matrices = copy.deepcopy(orig_factor_matrices)

        self.force_function_list = force_function_list

        self.input_tensor = input_tensors_list

        self.dirichlet_input_dict = copy.deepcopy(dirichlet_list[0])
        self.dirichlet_actual_dict = copy.deepcopy(dirichlet_list[1])

        if len(dirichlet_list) > 2:
            self.neumann_input_dict = copy.deepcopy(dirichlet_list[2])
            self.neumann_actual_dict = copy.deepcopy(dirichlet_list[3])
            self.neumann_normal_x = copy.deepcopy(dirichlet_list[4])
            self.neumann_normal_y = copy.deepcopy(dirichlet_list[5])
        else:
            self.neumann_input_dict = {}
            self.neumann_actual_dict = {}
            self.neumann_normal_x = {}
            self.neumann_normal_y = {}

        self.params_dict = params_dict

        print(f"{'-'*74}")
        print(f"| {'PARAMETER':<25} | {'SHAPE':<25} |")
        print(f"{'-'*74}")
        print(
            f"| {'input_tensor':<25} | {str(self.input_tensor.shape):<25} | {self.input_tensor.dtype}"
        )

        # loop over the number of components in the force_matrix_list
        for i in range(len(self.force_function_list)):
            print(
                f"| {'force_matrix['+str(i)+']':<25} | {str(self.force_function_list[i].shape):<25} | {self.force_function_list[i].dtype}"
            )

        # loop over fespaces to obtain the shape of the shape function and derivative matrices
        for k, v in self.orig_factor_matrices.items():
            for kk, vv in v.items():
                print(f"| {str(kk) + '['+str(k)+']':<25} | {str(vv.shape):<25} | {vv.dtype}")

        # loop over the boundary conditions to obtain the shape of the boundary data
        for component_id, dirichlet_input in self.dirichlet_input_dict.items():
            print(
                f"| {'dirichlet_input['+str(component_id)+']':<25} | {str(dirichlet_input.shape):<25} | {dirichlet_input.dtype}"
            )

        # loop over the boundary conditions to obtain the shape of the Neumann boundary data
        for component_id, neumann_input in self.neumann_input_dict.items():
            print(
                f"| {'neumann_input['+str(component_id)+']':<25} | {str(neumann_input.shape):<25} | {neumann_input.dtype}"
            )

        # loop over the force function list to obtain the shape of the force function matrices
        for i in range(len(self.force_function_list)):
            print(
                f"| {'force_function['+str(i)+']':<25} | {str(self.force_function_list[i].shape):<25} | {self.force_function_list[i].dtype}"
            )

        print(f"{'-'*74}")

        self.n_cells = params_dict['n_cells']

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
                    kernel_initializer='glorot_uniform',
                    dtype=self.tensor_dtype,
                    bias_initializer='zeros',
                )
            )

        # Add a output layer with no activation
        self.layer_list.append(
            layers.Dense(
                self.layer_dims[-1],
                activation=None,
                kernel_initializer='glorot_uniform',
                dtype=self.tensor_dtype,
                bias_initializer='zeros',
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
        x = inputs

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                'layer_dims': self.layer_dims,
                'use_attention': self.use_attention,
                'activation': self.activation,
                'loss_function': self.loss_function,
                'hessian': self.hessian,
                'tensor_dtype': self.tensor_dtype.name,  # tf.DType is not JSON serializable, so we store its name
                'orig_factor_matrices': self.orig_factor_matrices,
                'force_function_list': self.force_function_list,
                'input_tensor': self.input_tensor,
                'dirichlet_input_list': self.dirichlet_input_list,
                'dirichlet_actual_list': self.dirichlet_actual_list,
                'params_dict': self.params_dict,
            }
        )

        return base_config

    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None):

        with tf.GradientTape(persistent=True) as tape:

            # For Dirichlet Boundary
            boundary_loss = 0.0

            # Predict the values for dirichlet boundary conditions - u & v
            for component_id, dirichlet_input in self.dirichlet_input_dict.items():
                predicted_values_dirichlet = self(dirichlet_input)[
                    :, component_id : component_id + 1
                ]
                boundary_loss += tf.reduce_mean(
                    tf.square(
                        predicted_values_dirichlet - self.dirichlet_actual_dict[component_id]
                    ),
                    axis=0,
                )

            # instantiate for Neumann Loss
            neumann_loss = 0.0

            for component_id, neumann_input in self.neumann_input_dict.items():
                with tf.GradientTape(persistent=True) as tape_neumann:
                    tape_neumann.watch(neumann_input)
                    # Predict the values for neumann boundary conditions - u
                    predicted_neumann_i = self(neumann_input)[:, component_id : component_id + 1]

                # Compute the neumann gradients of the predicted values wrt the input which is (x, y)
                gradients_i_neumann = tape_neumann.gradient(predicted_neumann_i, neumann_input)

                grad_i_neumann_x = gradients_i_neumann[:, 0:1]
                grad_i_neumann_y = gradients_i_neumann[:, 1:2]

                # now perform du_i/dx. n1 + du_i/dy. n2 - g
                neumann_residual_x = grad_i_neumann_x * self.neumann_normal_x[component_id]
                neumann_residual_y = grad_i_neumann_y * self.neumann_normal_y[component_id]

                neumann_loss += tf.reduce_sum(
                    tf.square(
                        (neumann_residual_x + neumann_residual_y)
                        - self.neumann_actual_dict[component_id]
                    )
                )

            # initialize total loss as a tensor with shape (1,) and value 0.0
            total_pde_loss = 0.0

            with tf.GradientTape(persistent=True) as tape1:
                # tape gradient
                tape1.watch(self.input_tensor)
                # Compute the predicted values from the model
                predicted_values = self(self.input_tensor)

                predicted_u = predicted_values[:, 0]
                predicted_v = predicted_values[:, 1]

                # compute the gradients of the predicted values wrt the input which is (x, y)
                gradients_u = tape1.gradient(predicted_u, self.input_tensor)
                gradients_v = tape1.gradient(predicted_v, self.input_tensor)

            # compute the gradients of the predicted values wrt the input which is (x, y)
            du_dx = tf.reshape(gradients_u[:, 0], [self.n_cells, -1])
            du_dy = tf.reshape(gradients_u[:, 1], [self.n_cells, -1])

            dv_dx = tf.reshape(gradients_v[:, 0], [self.n_cells, -1])
            dv_dy = tf.reshape(gradients_v[:, 1], [self.n_cells, -1])

            # reshape predicted_values to [self.n_cells, self.pre_multiplier_val.shape[-1]]
            u_predicted = tf.reshape(predicted_u, [self.n_cells, -1])
            v_predicted = tf.reshape(predicted_v, [self.n_cells, -1])

            cells_residual = self.loss_function(
                test_shape_val_mat=[
                    self.orig_factor_matrices["u"]["shape_val_mat_list"],
                    self.orig_factor_matrices["u"]["quad_weight_list"],
                ],
                test_grad_x_mat=[self.orig_factor_matrices["u"]["grad_x_mat_list"]],
                test_grad_y_mat=[self.orig_factor_matrices["u"]["grad_y_mat_list"]],
                pred_nn=[u_predicted, v_predicted],
                pred_grad_x_nn=[du_dx, dv_dx],
                pred_grad_y_nn=[du_dy, dv_dy],
                forcing_function=self.force_function_list,
                bilinear_params=bilinear_params_dict,
            )

            residual = tf.reduce_sum(cells_residual)

            # tf.print("Residual : ", residual)
            # tf.print("Residual Shape : ", residual.shape)

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

            # tf.print("Boundary Loss : ", boundary_loss)
            # tf.print("Boundary Loss Shape : ", boundary_loss.shape)
            # tf.print("Total PDE Loss : ", total_pde_loss)
            # tf.print("Total PDE Loss Shape : ", total_pde_loss.shape)

            # Compute Total Loss
            total_loss = total_pde_loss + beta * boundary_loss + beta * neumann_loss

        trainable_vars = self.trainable_variables
        self.gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {
            "loss_pde": total_pde_loss,
            "loss_dirichlet": boundary_loss,
            "loss": total_loss,
            "loss_neumann": neumann_loss,
        }
