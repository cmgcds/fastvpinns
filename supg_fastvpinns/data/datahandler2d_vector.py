# This class is to handle data for 2D problems, convert them into tensors using custom tf functions
# and make them available for the model to train
# @Author : Thivin Anandh D
# @Date : 22/Sep/2023
# @History : 22/Sep/2023 - Initial implementation with basic data handling

from FE_2D.fespace2d import *
from Geometry.geometry_2d import *
import tensorflow as tf

class DataHandler2D_Vector():
    """
    This class is to handle data for 2D problems, convert them into tensors using custom tf functions
    Responsible for all type conversions and data handling
    Note: All inputs to these functions are generally numpy arrays with dtype np.float64
            So we can either maintain the same dtype or convert them to tf.float32 ( for faster computation )
    """
    
    def __init__(self, fespace_list, fespace_name_list, domain, dtype):
        
        # Store the corresponding variables list in a dict
        self.datahandler_variables_dict = {}

        
        self.fespace_list = fespace_list
        self.fespace_name_list = fespace_name_list


    
        self.domain = domain
        self.dtype = dtype
        
        # Dict for storing dirichlet boundary data

        # check the length of the fespace_list and fespace_dict
        if (len(self.fespace_list) != len(self.fespace_name_list)):
            print("[ERROR] The length of the fespace_list and fespace_dict are not equal, File :", __file__)
            raise ValueError("The length of the fespace_list and fespace_dict are not equal")


        # create a dictionary of variables for each fespace
        self.fespace_dict = {}
        for fespace, fespace_name in zip(self.fespace_list, self.fespace_name_list):
            self.fespace_dict[fespace_name] = fespace


        for fespace_name, fespace in self.fespace_dict.items():

            # get the variables list
            self.shape_val_mat_list = []
            self.grad_x_mat_list = []
            self.grad_y_mat_list = []
            self.quad_weight_list = []
            self.x_pde_list = []
            self.forcing_function_list = []
            

            # check if the given dtype is a valid tensorflow dtype
            if not isinstance(self.dtype, tf.DType):
                raise TypeError("The given dtype is not a valid tensorflow dtype")


            for cell_index in range(fespace.n_cells):
                shape_val_mat = tf.constant(fespace.get_shape_function_val(cell_index), dtype=self.dtype)
                grad_x_mat = tf.constant(fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype)
                grad_y_mat = tf.constant(fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype)
                x_pde = tf.constant(fespace.get_quadrature_actual_coordinates(cell_index), dtype=self.dtype)
                quad_weight = tf.constant(fespace.get_quadrature_weights(cell_index), dtype=self.dtype)

                # forcing_function = tf.constant(fespace.get_forcing_function_values(cell_index), dtype=self.dtype)
                self.shape_val_mat_list.append(shape_val_mat) 
                self.grad_x_mat_list.append(grad_x_mat)
                self.grad_y_mat_list.append(grad_y_mat)
                self.x_pde_list.append(x_pde)
                self.quad_weight_list.append(quad_weight)

                
                # forcing function will be calculated seperately for each component
                # self.forcing_function_list.append(forcing_function)
            
            # now convert all the shapes into 3D tensors for easy multiplication
            # input tensor - x_pde_list
            self.x_pde_list = tf.reshape(self.x_pde_list, [-1, 2])

            # self.forcing_function_list = tf.concat(self.forcing_function_list, axis=1)

            self.shape_val_mat_list = tf.stack(self.shape_val_mat_list, axis=0)
            self.grad_x_mat_list = tf.stack(self.grad_x_mat_list, axis=0)
            self.grad_y_mat_list = tf.stack(self.grad_y_mat_list, axis=0)

            # Stack all the quadrature weights of dimension (N_quad_points, 1) into a matrix of dimension (N_quad_points, N_cells)
            self.quad_weight_list = tf.transpose(tf.stack(self.quad_weight_list, axis=1))


            self.datahandler_variables_dict[fespace_name] = {}
            self.datahandler_variables_dict[fespace_name]["shape_val_mat_list"] = self.shape_val_mat_list
            self.datahandler_variables_dict[fespace_name]["grad_x_mat_list"] = self.grad_x_mat_list
            self.datahandler_variables_dict[fespace_name]["grad_y_mat_list"] = self.grad_y_mat_list
            self.datahandler_variables_dict[fespace_name]["x_pde_list"] = self.x_pde_list
            self.datahandler_variables_dict[fespace_name]["quad_weight_list"] = self.quad_weight_list
            



    def get_pde_input(self):

        return self.fespace.get_pde_training_data()
    
    def get_dirichlet_input(self, component_list, fespaces_list):
        """
        This function will return the dirichlet input and actual dirichlet data as tensors for the given component list

        Parameters
        component_list : list of components for which the dirichlet data is required
        fespaces_list : list of fespaces corresponding to the components

        Returns

        """
        
        self.boundary_input_tensors_list = []
        self.boundary_actual_tensors_list = []


        if (len(component_list) != len(fespaces_list)):
            print("[ERROR] The length of the component_list and fespaces_list are not equal, File :", __file__)
            raise ValueError("The length of the component_list and fespaces_list are not equal")
        
        for fespace, component in zip(fespaces_list, component_list):
            input_dirichlet, actual_dirichlet = fespace.generate_dirichlet_boundary_data_vector(component)
            
            # convert to tensors
            input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
            actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
            actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

            # append to the list
            self.boundary_input_tensors_list.append(input_dirichlet)
            self.boundary_actual_tensors_list.append(actual_dirichlet)


        return self.boundary_input_tensors_list, self.boundary_actual_tensors_list



    def get_pressure_constraint_boundary_input(self, fespace, component_id):
        """
        For Navier Stokes problems, we cannot impose the pressure boundary conditions directly. However for the problems with 
        dirichlet boundary conditions on all the components, Pressure can take any solution, To constrain that pressure, we can constrain a single
        point in a cell to be equal to a constant value.

        Parameters
        fespace : fespace object
        component_id : component id for which the pressure constraint is required

        Returns

        """

        # get a random point from first cell
        point = fespace.get_quadrature_actual_coordinates(0)
        
        len_point = point.shape[0]
        
        # pick a random point from the first cell
        point = point[np.random.randint(0, len_point)]
        
        # provide value
        value = 0.0

        # convert to tensors
        point = tf.constant(point.reshape(-1,2), dtype=self.dtype)
        value = tf.constant(value, dtype=self.dtype)
        value = tf.reshape(value, [-1, 1])

        return point, value


    def get_rhs_list(self, component_list, fespaces_list):
        """
        This function is used to obtain the necessary RHS component ( in a vector valued problem )
        using the corresponding fespace and component id

        Parameters
        component_list : list of components for which the rhs is required
        fespaces_list : list of fespaces corresponding to the components

        Returns


        """
        self.forcing_function_matrices_list = []

        print("[INFO] Setting up RHS list")
        for fespace, component in zip(fespaces_list, component_list):
            forcing_list_per_cell = []
            # for a given component, get the forcing function values by looping over all the cells
            for cell_index in tqdm(range(fespace.n_cells)):
                forcing_function = tf.constant(fespace.get_forcing_function_values_vector(cell_index = cell_index, component = component), dtype=self.dtype)
                forcing_list_per_cell.append(forcing_function)


            # convert the forcing function list into a tensor
            forcing_matrix = tf.concat(forcing_list_per_cell, axis=1)

            # Append to the list
            self.forcing_function_matrices_list.append(forcing_matrix)

        return self.forcing_function_matrices_list


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
    def get_sensor_data(self, exact_sol, num_sensor_points, mesh_type, file_name=None):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype
        """
        print(f"mesh_type = {mesh_type}")
        if (mesh_type == "internal"):
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data(exact_sol, num_sensor_points)
        elif (mesh_type == "external"):
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data_external(exact_sol, num_sensor_points, file_name)
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