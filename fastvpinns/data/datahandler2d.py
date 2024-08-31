# This class is to handle data for 2D problems, convert them into tensors using custom tf functions
# and make them available for the model to train
# @Author : Thivin Anandh D
# @Date : 22/Sep/2023
# @History : 22/Sep/2023 - Initial implementation with basic data handling

from ..FE.fespace2d import *
from ..Geometry.geometry_2d import *
import tensorflow as tf

from .datahandler import DataHandler


class DataHandler2D(DataHandler):
    """
    This class is to handle data for 2D problems, convert them into tensors using custom tf functions.
    It is responsible for all type conversions and data handling.

    .. note:: All inputs to these functions are generally numpy arrays with dtype np.float64.
              So we can either maintain the same dtype or convert them to tf.float32 ( for faster computation ).

    :param fespace: The FESpace2D object.
    :type fespace: FESpace2D
    :param domain: The Domain2D object.
    :type domain: Domain2D
    :param shape_val_mat_list: List of shape function values for each cell.
    :type shape_val_mat_list: list
    :param grad_x_mat_list: List of shape function derivatives with respect to x for each cell.
    :type grad_x_mat_list: list
    :param grad_y_mat_list: List of shape function derivatives with respect to y for each cell.
    :type grad_y_mat_list: list
    :param x_pde_list: List of actual coordinates of the quadrature points for each cell.
    :type x_pde_list: list
    :param forcing_function_list: List of forcing function values for each cell.
    :type forcing_function_list: list
    :param dtype: The tensorflow dtype to be used for all the tensors.
    :type dtype: tf.DType
    """

    def __init__(self, fespace, domain, dtype):
        """
        Constructor for the DataHandler2D class

        :param fespace: The FESpace2D object.
        :type fespace: FESpace2D
        :param domain: The Domain2D object.
        :type domain: Domain2D
        :param shape_val_mat_list: List of shape function values for each cell.
        :type shape_val_mat_list: list
        :param grad_x_mat_list: List of shape function derivatives with respect to x for each cell.
        :type grad_x_mat_list: list
        :param grad_y_mat_list: List of shape function derivatives with respect to y for each cell.
        :type grad_y_mat_list: list
        :param x_pde_list: List of actual coordinates of the quadrature points for each cell.
        :type x_pde_list: list
        :param forcing_function_list: List of forcing function values for each cell.
        :type forcing_function_list: list
        :param dtype: The tensorflow dtype to be used for all the tensors.
        :type dtype: tf.DType
        """
        # call the parent class constructor
        super().__init__(fespace=fespace, domain=domain, dtype=dtype)

        self.shape_val_mat_list = []
        self.grad_x_mat_list = []
        self.grad_y_mat_list = []
        self.x_pde_list = []
        self.forcing_function_list = []

        # check if the given dtype is a valid tensorflow dtype
        if not isinstance(self.dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        for cell_index in range(self.fespace.n_cells):
            shape_val_mat = tf.constant(
                self.fespace.get_shape_function_val(cell_index), dtype=self.dtype
            )
            grad_x_mat = tf.constant(
                self.fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype
            )
            grad_y_mat = tf.constant(
                self.fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype
            )
            x_pde = tf.constant(
                self.fespace.get_quadrature_actual_coordinates(cell_index), dtype=self.dtype
            )
            forcing_function = tf.constant(
                self.fespace.get_forcing_function_values(cell_index), dtype=self.dtype
            )
            self.shape_val_mat_list.append(shape_val_mat)
            self.grad_x_mat_list.append(grad_x_mat)
            self.grad_y_mat_list.append(grad_y_mat)
            self.x_pde_list.append(x_pde)
            self.forcing_function_list.append(forcing_function)

        # now convert all the shapes into 3D tensors for easy multiplication
        # input tensor - x_pde_list
        self.x_pde_list = tf.reshape(self.x_pde_list, [-1, 2])

        self.forcing_function_list = tf.concat(self.forcing_function_list, axis=1)

        self.shape_val_mat_list = tf.stack(self.shape_val_mat_list, axis=0)
        self.grad_x_mat_list = tf.stack(self.grad_x_mat_list, axis=0)
        self.grad_y_mat_list = tf.stack(self.grad_y_mat_list, axis=0)

        # test points
        self.test_points = None

    def get_dirichlet_input(self):
        """
        This function will return the input for the Dirichlet boundary data

        :return:
            - input_dirichlet (tf.Tensor): The input for the Dirichlet boundary data
            - actual_dirichlet (tf.Tensor): The actual Dirichlet boundary data
        """
        input_dirichlet, actual_dirichlet = self.fespace.generate_dirichlet_boundary_data()

        # convert to tensors
        input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

        return input_dirichlet, actual_dirichlet

    def get_test_points(self):
        """
        Get the test points for the given domain.

        :return: The test points for the given domain.
        :rtype: tf.Tensor
        """
        self.test_points = self.domain.get_test_points()
        self.test_points = tf.constant(self.test_points, dtype=self.dtype)
        return self.test_points

    def get_bilinear_params_dict_as_tensors(self, function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Parameters:
        - function (function): The function from the example file which returns the bilinear parameters dictionary

        Returns:
        - bilinear_params_dict (dict): The bilinear parameters dictionary with all the values converted to tensors

        :param function: The function from the example file which returns the bilinear parameters dictionary
        :type function: function
        :return: The bilinear parameters dictionary with all the values converted to tensors
        :rtype: dict
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

        :param exact_sol: The function from the example file which returns the exact solution
        :type exact_sol: function
        :param num_sensor_points: The number of sensor points to be generated
        :type num_sensor_points: int
        :param mesh_type: The type of mesh to be used for sensor data generation
        :type mesh_type: str
        :param file_name: The name of the file to be used for external mesh generation, defaults to None
        :type file_name: str, optional
        :return: The sensor points and sensor values as tensors
        :rtype: tuple[tf.Tensor, tf.Tensor]
        """
        print(f"mesh_type = {mesh_type}")
        if mesh_type == "internal":
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data(exact_sol, num_sensor_points)
        elif mesh_type == "external":
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data_external(
                exact_sol, num_sensor_points, file_name
            )
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

        :param inverse_params_dict_function: The function from the example file which returns the inverse parameters dictionary
        :type inverse_params_dict_function: function
        :return: The inverse parameters dictionary with all the values converted to tensors
        :rtype: dict
        """
        # loop over all keys and convert the values to tensors

        inverse_params_dict = inverse_params_dict_function()

        for key in inverse_params_dict.keys():
            inverse_params_dict[key] = tf.constant(inverse_params_dict[key], dtype=self.dtype)

        return inverse_params_dict
