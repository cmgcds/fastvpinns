from .datahandler2d import *


class DataHandlerDomainDecomposition(DataHandler2D):
    """
    This class is to handle data for domain decomposition problems, convert them into tensors using custom tf functions.
    """

    def __init__(self, fespace, domain, block_id, dtype):
        """
        Constructor for the DataHandlerDomainDecomposition class
        """
        super().__init__(fespace=fespace, domain=domain, dtype=dtype)
        self.block_id = block_id

        num_quadrature_points = tf.shape(self.shape_val_mat_list)[-1]
        self.subdomain_solution_values = tf.zeros(
            (self.fespace.n_cells, num_quadrature_points), dtype=self.dtype
        )
        self.subdomain_solution_gradient_x = tf.zeros(
            (self.fespace.n_cells, num_quadrature_points), dtype=self.dtype
        )
        self.subdomain_solution_gradient_y = tf.zeros(
            (self.fespace.n_cells, num_quadrature_points), dtype=self.dtype
        )

        self.overlap_solution_values = tf.zeros_like(self.subdomain_solution_values)
        self.overlap_solution_gradient_x = tf.zeros_like(self.subdomain_solution_gradient_x)
        self.overlap_solution_gradient_y = tf.zeros_like(self.subdomain_solution_gradient_y)

    def reset_overlap_values(self):
        """
        Reset the overlap values to zero.
        """
        self.overlap_solution_values = tf.zeros_like(self.subdomain_solution_values)
        self.overlap_solution_gradient_x = tf.zeros_like(self.subdomain_solution_gradient_x)
        self.overlap_solution_gradient_y = tf.zeros_like(self.subdomain_solution_gradient_y)

    def update_subdomain_solution(
        self,
        subdomain_solution_values,
        subdomain_solution_gradient_x,
        subdomain_solution_gradient_y,
    ):
        """
        Update the subdomain solution values and gradients.

        :param subdomain_solution_values: The solution values on the subdomain.
        :type subdomain_solution_values: tf.Tensor
        :param subdomain_solution_gradient_x: The solution gradients with respect to x on the subdomain.
        :type subdomain_solution_gradient_x: tf.Tensor
        :param subdomain_solution_gradient_y: The solution gradients with respect to y on the subdomain.
        :type subdomain_solution_gradient_y: tf.Tensor
        """
        self.subdomain_solution_values = subdomain_solution_values
        self.subdomain_solution_gradient_x = subdomain_solution_gradient_x
        self.subdomain_solution_gradient_y = subdomain_solution_gradient_y

    def get_window_function_values(self, window_function):
        """
        Get the window function values for the given block.

        :param window_function: The window function to be used.
        :type window_function: WindowFunction
        :param block_id: The block id.
        :type block_id: int
        :return: The window function values.
        :rtype: tf.Tensor
        """
        window_function_values = window_function.evaluate_window_function(
            self.x_pde_list[:, 0:1], self.x_pde_list[:, 0:1], self.block_id
        )
        window_function_values = tf.convert_to_tensor(window_function_values, dtype=self.dtype)
        return window_function_values
