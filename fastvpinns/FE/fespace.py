"""
file: fespace.py
description: Abstract class for the FEspace Routines
authors: Thivin Anandh D
changelog: 03-May-2024
known_issues: None
dependencies: None specified.
"""

import numpy as np
from abc import abstractmethod


class Fespace:
    """
    Represents a finite element space.

    :param mesh: The mesh object.
    :type mesh: Mesh
    :param cells: The array of cell indices.
    :type cells: ndarray
    :param boundary_points: The dictionary of boundary points.
    :type boundary_points: dict
    :param cell_type: The type of cell.
    :type cell_type: str
    :param fe_order: The order of the finite element.
    :type fe_order: int
    :param fe_type: The type of finite element.
    :type fe_type: str
    :param quad_order: The order of the quadrature.
    :type quad_order: int
    :param quad_type: The type of quadrature.
    :type quad_type: str
    :param fe_transformation_type: The type of finite element transformation.
    :type fe_transformation_type: str
    :param bound_function_dict: The dictionary of boundary functions.
    :type bound_function_dict: dict
    :param bound_condition_dict: The dictionary of boundary conditions.
    :type bound_condition_dict: dict
    :param forcing_function: The forcing function.
    :type forcing_function: function
    :param output_path: The output path.
    :type output_path: str
    :param generate_mesh_plot: Whether to generate a plot of the mesh. Defaults to False.
    :type generate_mesh_plot: bool, optional
    """

    def __init__(
        self,
        mesh,
        cells,
        boundary_points,
        cell_type: str,
        fe_order: int,
        fe_type: str,
        quad_order: int,
        quad_type: str,
        fe_transformation_type: str,
        bound_function_dict: dict,
        bound_condition_dict: dict,
        forcing_function,
        output_path: str,
    ) -> None:
        """
        The constructor of the Fespace2D class.
        """
        self.mesh = mesh
        self.boundary_points = boundary_points
        self.cells = cells
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type

        self.fe_transformation_type = fe_transformation_type
        self.output_path = output_path
        self.bound_function_dict = bound_function_dict
        self.bound_condition_dict = bound_condition_dict
        self.forcing_function = forcing_function

    @abstractmethod
    def set_finite_elements(self) -> None:
        """
        Assigns the finite elements to each cell.

        This method initializes the finite element objects for each cell in the mesh.
        It creates an instance of the `FE2D_Cell` class for each cell, passing the necessary parameters.
        The finite element objects store information about the basis functions, gradients, Jacobians,
        quadrature points, weights, actual coordinates, and forcing functions associated with each cell.

        After initializing the finite element objects, this method prints the shape details of various matrices
        and updates the total number of degrees of freedom (dofs) for the entire mesh.

        :return: None
        """

    @abstractmethod
    def generate_dirichlet_boundary_data(self) -> np.ndarray:
        """
        Generate Dirichlet boundary data.

        This function returns the boundary points and their corresponding values.

        :return: A tuple containing two arrays:
            - The first array contains the boundary points as numpy arrays.
            - The second array contains the values of the boundary points as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

    @abstractmethod
    def get_shape_function_val(self, cell_index) -> np.ndarray:
        """
        Get the actual values of the shape functions on a given cell.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the actual values of the shape functions.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_x(self, cell_index) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the gradient of the shape function with respect to the x-coordinate.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.

        This function returns the actual values of the gradient of the shape function on a given cell.
        """

    @abstractmethod
    def get_shape_function_grad_x_ref(self, cell_index) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate on the reference element.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the gradient of the shape function with respect to the x-coordinate.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_y(self, cell_index) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to y at the given cell index.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: The gradient of the shape function with respect to y.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the total number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_y_ref(self, cell_index):
        """
        Get the gradient of the shape function with respect to y at the reference element.

        :param cell_index: The index of the cell.
        :type cell_index: int
        :return: The gradient of the shape function with respect to y at the reference element.
        :rtype: np.ndarray
        :raises ValueError: If cell_index is greater than the number of cells.

        This function returns the gradient of the shape function with respect to y at the reference element
        for a given cell. The shape function gradient values are stored in the `basis_grady_at_quad_ref` array
        of the corresponding finite element cell. The `cell_index` parameter specifies the index of the cell
        for which the shape function gradient is required. If the `cell_index` is greater than the total number
        of cells, a `ValueError` is raised.

        .. note::
            The returned gradient values are copied from the `basis_grady_at_quad_ref` array to ensure immutability.
        """

    @abstractmethod
    def get_quadrature_actual_coordinates(self, cell_index) -> np.ndarray:
        """
        Get the actual coordinates of the quadrature points for a given cell.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the actual coordinates of the quadrature points.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_forcing_function_values(self, cell_index) -> np.ndarray:
        """
        Get the forcing function values at the quadrature points.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: The forcing function values at the quadrature points.
        :rtype: np.ndarray

        :raises ValueError: If cell_index is greater than the number of cells.

        This function computes the forcing function values at the quadrature points for a given cell.
        It loops over all the basis functions and computes the integral using the actual coordinates
        and the basis functions at the quadrature points. The resulting values are stored in the
        `forcing_at_quad` attribute of the corresponding `fe_cell` object.

        Note: The forcing function is evaluated using the `forcing_function` method of the `fe_cell`
        object.
        """

    @abstractmethod
    def get_sensor_data(self, exact_solution, num_points):
        """
        Obtain sensor data (actual solution) at random points.

        This method is used in the inverse problem to obtain the sensor data at random points within the domain.
        Currently, it only works for problems with an analytical solution.
        Methodologies to obtain sensor data for problems from a file are not implemented yet.
        It is also not implemented for external or complex meshes.

        :param exact_solution: A function that computes the exact solution at a given point.
        :type exact_solution: function
        :param num_points: The number of random points to generate.
        :type num_points: int
        :return: A tuple containing the generated points and the exact solution at those points.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """

    @abstractmethod
    def get_sensor_data_external(self, exact_sol, num_points, file_name):
        """
        This method is used to obtain the sensor data from an external file.

        :param exact_sol: The exact solution values.
        :type exact_sol: array-like
        :param num_points: The number of points to sample from the data.
        :type num_points: int
        :param file_name: The path to the file containing the sensor data.
        :type file_name: str

        :return: A tuple containing two arrays:
            - points (ndarray): The sampled points from the data.
            - exact_sol (ndarray): The corresponding exact solution values.
        :rtype: tuple
        """
