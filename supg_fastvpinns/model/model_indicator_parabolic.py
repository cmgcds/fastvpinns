# A method which hosts the NN model and the training loop for variational Pinns with HARD CONSTRAINTS
# This focusses only on the model architecture and the training loop, and not on the loss functions
# Author : Thivin Anandh D
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic model architecture and training loop

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy
import math
# tf.config.run_functions_eagerly(True)

# Custom Loss Functions
def custom_loss1(y_true1, y_pred1):
    return tf.reduce_mean(tf.square(y_pred1 - y_true1))

def custom_loss2(y_true2, y_pred2):
    return tf.reduce_mean(tf.square(y_pred2 - y_true2))

# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, num_cells):
#         super(AttentionLayer, self).__init__()
#         self.num_cells = num_cells

#     def build(self, input_shape):
#         self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, x):
#         attention_weights = tf.matmul(x, self.W)
#         return attention_weights

# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, num_cells):
#         super(AttentionLayer, self).__init__()
#         self.W = self.add_weight(shape=(num_cells, 1), initializer='random_normal', trainable=True)
#         self.W = tf.cast(self.W, dtype=tf.float64)
#         # def build(self, input_shape):
#         # self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)

#     def call(self, x):
#         scores = tf.multiply(x, self.W)
#         # attention_weights = tf.nn.softmax(scores, axis=1)
#         # print()
#         return scores

# Custom Model
class DenseModel(tf.keras.Model):
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
    def __init__(self, layer_dims, learning_rate_dict, params_dict, loss_function, input_tensors_list, orig_factor_matrices , force_function_list, \
                 tensor_dtype, k, k1, k2,
                 use_attention=False, activation='tanh', hessian=False):
        super(DenseModel, self).__init__()
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

        self.orig_factor_matrices = orig_factor_matrices
        self.shape_function_mat_list = copy.deepcopy(orig_factor_matrices[0])
        self.shape_function_grad_x_factor_mat_list = copy.deepcopy(orig_factor_matrices[1])
        self.shape_function_grad_y_factor_mat_list = copy.deepcopy(orig_factor_matrices[2])
        
        self.force_function_list = force_function_list

        # self.k = tf.Variable(2, dtype=tf.float64, trainable=True, name='k')
        # self.k = tf.constant(10.0/1e-8, dtype = tf.float64)
        # self.k = tf.Variable(6, dtype=tf.float64, trainable=True, name='k')
        # self.k1 = tf.Variable(2, dtype=tf.float64, trainable=True, name='k1')

        self.k = tf.Variable(k, dtype=tf.float64, trainable=True, name='k')
        self.k1 = tf.Variable(k1, dtype=tf.float64, trainable=True, name='k1')
        self.k2 = tf.Variable(k2, dtype=tf.float64, trainable=True, name='k2')
        # self.k2 = tf.Variable(2, dtype=tf.float64, trainable=True, name='k2')

        # self.k2 = tf.constant(30, dtype = tf.float64)
        # self.k2 = tf.constant(0, d        self.k3 = tf.Variable(0.5, dtype=tf.float64, trainable=True, name='k2')
        # self.k3 = tf.constant(0, dtype = tf.float64)
        # self.k3 = tf.Variable(1, dtype=tf.float64, trainable=True, name='k3')

        self.input_tensors_list = input_tensors_list
        self.input_tensor = copy.deepcopy(input_tensors_list[0])
        self.dirichlet_input = copy.deepcopy(input_tensors_list[1])
        self.dirichlet_actual = copy.deepcopy(input_tensors_list[2])
        

        self.params_dict = params_dict

        self.pre_multiplier_val   = self.shape_function_mat_list
        self.pre_multiplier_grad_x = self.shape_function_grad_x_factor_mat_list
        self.pre_multiplier_grad_y = self.shape_function_grad_y_factor_mat_list

        self.force_matrix = self.force_function_list
        
        print(f"{'-'*74}")
        print(f"| {'PARAMETER':<25} | {'SHAPE':<25} |")
        print(f"{'-'*74}")
        print(f"| {'input_tensor':<25} | {str(self.input_tensor.shape):<25} | {self.input_tensor.dtype}")
        print(f"| {'force_matrix':<25} | {str(self.force_matrix.shape):<25} | {self.force_matrix.dtype}")
        print(f"| {'pre_multiplier_grad_x':<25} | {str(self.pre_multiplier_grad_x.shape):<25} | {self.pre_multiplier_grad_x.dtype}")
        print(f"| {'pre_multiplier_grad_y':<25} | {str(self.pre_multiplier_grad_y.shape):<25} | {self.pre_multiplier_grad_y.dtype}")
        print(f"| {'pre_multiplier_val':<25} | {str(self.pre_multiplier_val.shape):<25} | {self.pre_multiplier_val.dtype}")
        print(f"| {'dirichlet_input':<25} | {str(self.dirichlet_input.shape):<25} | {self.dirichlet_input.dtype}")
        print(f"| {'dirichlet_actual':<25} | {str(self.dirichlet_actual.shape):<25} | {self.dirichlet_actual.dtype}")
        print(f"{'-'*74}")
        
        self.n_cells = params_dict['n_cells']

        # self.attention_layer = AttentionLayer(self.n_cells)

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

        if(use_lr_scheduler):
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
            self.layer_list.append(layers.Dense(self.layer_dims[dim+1], activation=self.activation, \
                                                    kernel_initializer='glorot_uniform', \
                                                    dtype=self.tensor_dtype, bias_initializer='zeros'))
        
        # Add a output layer with no activation
        self.layer_list.append(layers.Dense(self.layer_dims[-1], activation=None, 
                                    kernel_initializer='glorot_uniform',
                                    dtype=self.tensor_dtype, bias_initializer='zeros'))
        

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

     
        # For Parabolic Layer Problem - ansatz 2 , 3 parameters
        aa = (1 - tf.exp(-10**(self.k1)*tf.cast(inputs[:, 0:1], tf.float64))) * (1 - tf.exp(-10**(self.k)*tf.cast(inputs[:, 1:2], tf.float64))) * (1 - tf.exp(-10**(self.k2)*tf.cast((1 - inputs[:, 0:1]), tf.float64))) * (1 - tf.exp(-10**(self.k)*tf.cast((1 - inputs[:, 1:2]), tf.float64)))

        x = x*aa
        return x
    
    def get_config(self):
        # Get the base configuration
        base_config = super().get_config()

        # Add the non-serializable arguments to the configuration
        base_config.update({
            'learning_rate_dict': self.learning_rate_dict,
            'loss_function': self.loss_function,
            'input_tensors_list':  self.input_tensors_list,
            'orig_factor_matrices': self.orig_factor_matrices,
            'force_function_list': self.force_function_list,
            'params_dict': self.params_dict,
            'use_attention': self.use_attention,
            'activation': self.activation,
            'hessian': self.hessian,
            'layer_dims': self.layer_dims,
            'tensor_dtype': self.tensor_dtype
        })

        return base_config
    


    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None, l2_lambda = 0.0001, pd_lambda = 0.0001):

        with tf.GradientTape(persistent=True) as tape:
            # Predict the values for dirichlet boundary conditions
            predicted_values_dirichlet = self(self.dirichlet_input)

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
            pred_grad_x = tf.reshape(gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])  # shape : (N_cells , N_quadrature_points)
            
            pred_val = tf.reshape(predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]])  # shape : (N_cells , N_quadrature_points)

            

            cells_residual = self.loss_function(test_shape_val_mat = self.pre_multiplier_val, test_grad_x_mat = self.pre_multiplier_grad_x, \
                                test_grad_y_mat = self.pre_multiplier_grad_y, pred_nn = pred_val, \
                                pred_grad_x_nn = pred_grad_x, pred_grad_y_nn = pred_grad_y ,\
                                forcing_function = self.force_matrix, bilinear_params = bilinear_params_dict)


            residual = tf.reduce_sum(cells_residual) 

            # tf.print("Residual : ", residual)
            # tf.print("Residual Shape : ", residual.shape)

            # Compute the total loss for the PDE
            total_pde_loss = pd_lambda*(total_pde_loss + residual)
            # tf.print(total_pde_loss)

            # convert predicted_values_dirichlet to tf.float64
            predicted_values_dirichlet = tf.cast(predicted_values_dirichlet, tf.float64)

            # print shapes of the predicted values and the actual values
            boundary_loss = tf.reduce_mean(tf.square(predicted_values_dirichlet - self.dirichlet_actual), axis=0)

            # Compute Total Loss with L2 Regularization (only for the weights and not the biases)
            l2_regularisation = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in self.trainable_variables if 'kernel' in var.name])
            total_loss = total_pde_loss + l2_regularisation


        trainable_vars = self.trainable_variables
        self.gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        # backward compatibility, adding the loss values to a dictionary
        # boundary_loss = 0.0
        
        return {"loss_pde": total_pde_loss, "loss_dirichlet": boundary_loss, "loss": total_loss, "cells_residual": cells_residual, "l2_regularisation": l2_regularisation, "k": self.k, "k1": self.k1 , "k2": self.k2}
