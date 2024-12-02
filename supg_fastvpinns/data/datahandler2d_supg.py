# This class is to handle data for 2D problems, convert them into tensors using custom tf functions
# and make them available for the model to train
# @Author : Thivin Anandh D
# @Date : 22/Sep/2023
# @History : 22/Sep/2023 - Initial implementation with basic data handling

from FE_2D.fespace2d_supg import *
from Geometry.geometry_2d import *
import tensorflow as tf

class DataHandler2D():
    """
    This class is to handle data for 2D problems, convert them into tensors using custom tf functions
    Responsible for all type conversions and data handling
    Note: All inputs to these functions are generally numpy arrays with dtype np.float64
            So we can either maintain the same dtype or convert them to tf.float32 ( for faster computation )
    """
    
    def __init__(self, fespace, domain, dtype):
        

        self.fespace = fespace
        self.domain = domain
        self.shape_val_mat_list = []
        self.grad_x_mat_list = []
        self.grad_y_mat_list = []
        self.x_pde_list = []
        self.forcing_function_list = []
        self.real_forcing_function_list = []
        self.dtype = dtype

        # check if the given dtype is a valid tensorflow dtype
        if not isinstance(self.dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")


        for cell_index in range(self.fespace.n_cells):
            shape_val_mat = tf.constant(self.fespace.get_shape_function_val(cell_index), dtype=self.dtype)
            grad_x_mat = tf.constant(self.fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype)
            grad_y_mat = tf.constant(self.fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype)
            x_pde = tf.constant(self.fespace.get_quadrature_actual_coordinates(cell_index), dtype=self.dtype)
            forcing_function = tf.constant(self.fespace.get_forcing_function_values(cell_index), dtype=self.dtype)
            real_forcing_function_new = tf.constant(self.fespace.real_f_values(cell_index), dtype = self.dtype)
            # print(real_forcing_function_new.shape)
            self.shape_val_mat_list.append(shape_val_mat) 
            self.grad_x_mat_list.append(grad_x_mat)
            self.grad_y_mat_list.append(grad_y_mat)
            self.x_pde_list.append(x_pde)
            self.forcing_function_list.append(forcing_function)
            self.real_forcing_function_list.append(real_forcing_function_new)
        
        # now convert all the shapes into 3D tensors for easy multiplication
        # input tensor - x_pde_list
        self.x_pde_list = tf.reshape(self.x_pde_list, [-1, 2])

        self.forcing_function_list = tf.concat(self.forcing_function_list, axis=1)
        # print(self.forcing_function_list.shape)
        # print("------------")

        self.real_forcing_function = tf.transpose(tf.concat(self.real_forcing_function_list, axis=1))
        # print(self.real_forcing_function.shape)

        self.shape_val_mat_list = tf.stack(self.shape_val_mat_list, axis=0)
        self.grad_x_mat_list = tf.stack(self.grad_x_mat_list, axis=0)
        self.grad_y_mat_list = tf.stack(self.grad_y_mat_list, axis=0)
        # self.real_forcing_function = fespace.real_f_values(0)


    def get_pde_input(self):

        return self.fespace.get_pde_training_data()
    
    def get_dirichlet_input(self):
        input_dirichlet, actual_dirichlet = self.fespace.generate_dirichlet_boundary_data()
        
        # convert to tensors
        input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

        return input_dirichlet, actual_dirichlet

    def get_test_points(self, num_test_points):
        """
        This function will return the test points
        """
        

        self.test_points = self.domain.generate_test_points(num_test_points)
        self.test_points = tf.constant(self.test_points, dtype=self.dtype)
        return self.test_points

    def get_bilinear_params_dict_as_tensors(self, function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype
        """
        
        # get the dictionary of bilinear parameters
        bilinear_params_dict = function()

        # loop over all keys and convert the values to tensors
        for key in bilinear_params_dict.keys():
            bilinear_params_dict[key] = tf.constant(bilinear_params_dict[key], dtype=self.dtype)
        
        return bilinear_params_dict
    

    # to be used only in inverse problems
    def get_sensor_data(self, exact_sol, num_sensor_points, mesh_type):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype
        """
        print(f"mesh_type = {mesh_type}")
        if (mesh_type != "internal"):
            # print an error of the file and function name
            print(f"ERROR: get_sensor_data() is implemented only for internal mesh")
            print(f"ERROR: At the file name {__file__} ")

            # raise NotImplementedError
            raise NotImplementedError(f"ERROR: get_sensor_data() is implemented only for internal mesh")
        

        # Call the method to get the sensor data
        points, sensor_values = self.fespace.get_sensor_data(exact_sol, num_sensor_points)

        # convert the points and sensor values into tensors
        points = tf.constant(points, dtype=self.dtype)
        sensor_values = tf.constant(sensor_values, dtype=self.dtype)

        sensor_values = tf.reshape(sensor_values, [-1, 1])
        points = tf.reshape(points, [-1, 2])



        return points, sensor_values
    

    # get inverse param dict as tensors
    def get_inverse_params(self, inverse_params_dict_function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype
        """
        # loop over all keys and convert the values to tensors

        inverse_params_dict = inverse_params_dict_function()

        for key in inverse_params_dict.keys():
            inverse_params_dict[key] = tf.constant(inverse_params_dict[key], dtype=self.dtype)
        
        return inverse_params_dict