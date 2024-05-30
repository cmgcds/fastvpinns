"""
The file `datahandler.py` is an abstract class for Datahandler.

Author : Thivin Anandh D

Date : 03/May/2024

History : 03/May/2024 - Initial implementation with basic data handling
"""

from abc import abstractmethod


class DataHandler:
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
        Constructor for the DataHandler class

        :param fespace: The FESpace2D object.
        :type fespace: FESpace2D
        :param domain: The Domain object.
        :type domain: Domain2D
        :param dtype: The tensorflow dtype to be used for all the tensors.
        :type dtype: tf.DType
        """

        self.fespace = fespace
        self.domain = domain
        self.dtype = dtype

    @abstractmethod
    def get_dirichlet_input(self):
        """
        This function will return the input for the Dirichlet boundary data

        :return:
            - input_dirichlet (tf.Tensor): The input for the Dirichlet boundary data
            - actual_dirichlet (tf.Tensor): The actual Dirichlet boundary data
        """

    @abstractmethod
    def get_test_points(self):
        """
        Get the test points for the given domain.

        :return: The test points for the given domain.
        :rtype: tf.Tensor
        """

    @abstractmethod
    def get_bilinear_params_dict_as_tensors(self, function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        :param function: The function from the example file which returns the bilinear parameters dictionary
        :type function: function
        :return: The bilinear parameters dictionary with all the values converted to tensors
        :rtype: dict
        """

    @abstractmethod
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

    @abstractmethod
    def get_inverse_params(self, inverse_params_dict_function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        :param inverse_params_dict_function: The function from the example file which returns the inverse parameters dictionary
        :type inverse_params_dict_function: function
        :return: The inverse parameters dictionary with all the values converted to tensors
        :rtype: dict
        """
