# This class is to handle data for 2D problems, convert them into tensors using custom tf functions
# and make them available for the model to train
# @Author : Thivin Anandh D
# @Date : 28/May/2024
# @History : 22/May/2024 - Initial implementation with basic data handling

from ..FE.fespace2d import *
from ..Geometry.geometry_2d import *
import tensorflow as tf

from .datahandler import DataHandler


class DataHandler2D_Vector(DataHandler):
    """
    This class is to handle data for 2D Vector valued problems, convert them into tensors using custom tf functions.
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

    def __init__(self, fespace_list, fespace_name_list, domain, dtype):
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
        super().__init__(fespace=fespace_list[0], domain=domain, dtype=dtype)

        # Store the variables in a dict
        self.datahandler_variables_dict = {}

        # Fespace list
        self.fespace_list = []
        self.fespace_name_list = []

        # Scaling parameters
        self.output_scaling_max = None
        self.input_scaling_min = None
        self.input_scaling_max = None

        # check if the given dtype is a valid tensorflow dtype
        if not isinstance(self.dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        # Check the list of Fespace list and fespace dict to make sure that both are of same length
        if len(self.fespace_list) != len(self.fespace_name_list):
            print("The length of the fespace list and fespace name list are not equal")
            raise ValueError("The length of the fespace list and fespace name list are not equal")

        # create a dictionay of variables for each fespace
        self.fespace_dict = {}
        for fespace, fespace_name in zip(fespace_list, fespace_name_list):
            self.fespace_dict[fespace_name] = fespace

        # Assemble the fespace for each fespace
        for fespace_name, fespace in self.fespace_dict.items():

            # Initialise the list to store the tensors
            shape_val_mat_list = []
            grad_x_mat_list = []
            grad_y_mat_list = []
            x_pde_list = []
            quad_weights_list = []

            # loop over all cells to get the shape function values, gradients and quadrature points
            for cell_index in range(fespace.n_cells):
                shape_val_mat = tf.constant(
                    fespace.get_shape_function_val(cell_index), dtype=self.dtype
                )
                grad_x_mat = tf.constant(
                    fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype
                )
                grad_y_mat = tf.constant(
                    fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype
                )
                x_pde = tf.constant(
                    fespace.get_quadrature_actual_coordinates(cell_index), dtype=self.dtype
                )
                quad_weights = tf.constant(
                    fespace.get_quadrature_weights(cell_index), dtype=self.dtype
                )

                shape_val_mat_list.append(shape_val_mat)
                grad_x_mat_list.append(grad_x_mat)
                grad_y_mat_list.append(grad_y_mat)
                x_pde_list.append(x_pde)
                quad_weights_list.append(quad_weights)

            # Convert the lists to tensors
            x_pde_list = tf.reshape(x_pde_list, [-1, 2])
            shape_val_mat_list = tf.stack(shape_val_mat_list, axis=0)
            grad_x_mat_list = tf.stack(grad_x_mat_list, axis=0)
            grad_y_mat_list = tf.stack(grad_y_mat_list, axis=0)

            # stack all quadrature weights frmo list of n_cells (where each elem has n-quad * 1) into
            # matrix (n_cells * n_quad, 1)
            self.quad_weights_list = tf.reshape(tf.stack(quad_weights_list, axis=0), [-1, 1])

            self.datahandler_variables_dict[fespace_name] = {}
            self.datahandler_variables_dict[fespace_name]["shape_val_mat_list"] = shape_val_mat_list
            self.datahandler_variables_dict[fespace_name]["grad_x_mat_list"] = grad_x_mat_list
            self.datahandler_variables_dict[fespace_name]["grad_y_mat_list"] = grad_y_mat_list
            self.datahandler_variables_dict[fespace_name]["x_pde_list"] = x_pde_list
            self.datahandler_variables_dict[fespace_name][
                "quad_weight_list"
            ] = self.quad_weights_list

    def get_dirichlet_input(self, component_list, fespaces_list):
        """
        This function will return the dirichlet input and actual dirichlet data as tensors for the given component list

        Parameters
        component_list : list of components for which the dirichlet data is required
        fespaces_list : list of fespaces corresponding to the components

        Returns
        """

        self.boundary_input_tensors_dict = {}
        self.boundary_actual_tensors_dict = {}

        if len(component_list) != len(fespaces_list):
            print(
                "[ERROR] The length of the component_list and fespaces_list are not equal, File :",
                __file__,
            )
            raise ValueError("The length of the component_list and fespaces_list are not equal")

        for fespace, component in zip(fespaces_list, component_list):
            input_dirichlet, actual_dirichlet = fespace.generate_dirichlet_boundary_data_vector(
                component
            )
            # if a component is empty Do not fill the value for the component
            if input_dirichlet == []:
                continue

            # use vstack to stack all the lists vertically
            input_dirichlet = np.vstack(input_dirichlet)
            actual_dirichlet = np.vstack(actual_dirichlet)

            # convert to tensors
            input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
            actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
            actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

            # append to the list
            self.boundary_input_tensors_dict[component] = input_dirichlet
            self.boundary_actual_tensors_dict[component] = actual_dirichlet

        return self.boundary_input_tensors_dict, self.boundary_actual_tensors_dict

    def get_neumann_input(
        self, component_list, fespaces_list, bd_cell_info_dict, bd_joint_info_dict
    ):
        """
        This function will return the dirichlet input and actual dirichlet data as tensors for the given component list

        Parameters
        component_list : list of components for which the dirichlet data is required
        fespaces_list : list of fespaces corresponding to the components

        Returns

        """

        self.boundary_neumann_input_tensors_dict = {}
        self.boundary_neumann_actual_tensors_dict = {}
        self.boundary_neumann_normal_x_tensors_dict = {}
        self.boundary_neumann_normal_y_tensors_dict = {}

        if len(component_list) != len(fespaces_list):
            print(
                "[ERROR] The length of the component_list and fespaces_list are not equal, File :",
                __file__,
            )
            raise ValueError("The length of the component_list and fespaces_list are not equal")

        for fespace, component in zip(fespaces_list, component_list):
            input_neumann, actual_neumann, normal_x, normal_y = (
                fespace.generate_neumann_boundary_data_vector(
                    component, bd_cell_info_dict, bd_joint_info_dict
                )
            )

            # if a component is empty Do not fill the value for the component
            if input_neumann == []:
                continue

            # concatenate the lists into a single tensor by stacking all list elements vertically
            input_neumann = np.vstack(input_neumann)
            actual_neumann = np.vstack(actual_neumann)
            normal_x = np.vstack(normal_x)
            normal_y = np.vstack(normal_y)

            # convert to tensors
            input_neumann = tf.constant(input_neumann, dtype=self.dtype)
            actual_neumann = tf.constant(actual_neumann, dtype=self.dtype)

            normal_x = tf.constant(normal_x, dtype=self.dtype)
            normal_y = tf.constant(normal_y, dtype=self.dtype)

            # append to the dict
            self.boundary_neumann_input_tensors_dict[component] = input_neumann
            self.boundary_neumann_actual_tensors_dict[component] = actual_neumann
            self.boundary_neumann_normal_x_tensors_dict[component] = normal_x
            self.boundary_neumann_normal_y_tensors_dict[component] = normal_y

        return (
            self.boundary_neumann_input_tensors_dict,
            self.boundary_neumann_actual_tensors_dict,
            self.boundary_neumann_normal_x_tensors_dict,
            self.boundary_neumann_normal_y_tensors_dict,
        )

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
        point = tf.constant(point.reshape(-1, 2), dtype=self.dtype)
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
            for cell_index in tqdm(
                range(fespace.n_cells), desc=f"RHS Assembly for comp {component}", ncols=80
            ):
                forcing_function = tf.constant(
                    fespace.get_forcing_function_values_vector(
                        cell_index=cell_index, component=component
                    ),
                    dtype=self.dtype,
                )
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
        """
        # loop over all keys and convert the values to tensors

        inverse_params_dict = inverse_params_dict_function()

        for key in inverse_params_dict.keys():
            inverse_params_dict[key] = tf.constant(inverse_params_dict[key], dtype=self.dtype)

        return inverse_params_dict

    def get_penalty_coefficients(self, penalty_coefficients):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype
        """
        # loop over all keys and convert the values to tensors
        penalty_coefficients_dict = penalty_coefficients()

        for key in penalty_coefficients_dict.keys():
            penalty_coefficients_dict[key] = tf.constant(
                penalty_coefficients_dict[key], dtype=self.dtype
            )

        return penalty_coefficients_dict

    def get_output_scaling(self, override_value_list=None):
        """
        Function which obtains the output max based on the dirichlet boundary values
        """
        output_max_list = []
        # get the max dirichlet values
        for component, output_dirichlet in self.boundary_actual_tensors_dict.items():
            scale_val = tf.reduce_max(tf.abs(output_dirichlet))
            if tf.abs(scale_val) < 1e-6:
                output_max_list.append(1.0)
            else:
                if override_value_list is not None and override_value_list[component] is not None:
                    output_max_list.append(override_value_list[component])
                else:
                    output_max_list.append(scale_val)

        # Add a value for pressure
        output_max_list.append(1.0)

        # convert the list into a 1, 3 tensor
        output_max = tf.reshape(tf.stack(output_max_list), [1, 3])

        # convert into data type
        self.output_scaling_max = tf.cast(output_max, dtype=self.dtype)

        return self.output_scaling_max

    def get_input_scaling(self, cell_points):
        """
        Function which obtains the x_min and x_max from the cell_points array
        """
        # compute the max value of all columns in the cell_points
        x_max = tf.reduce_max(tf.reduce_max(tf.abs(cell_points), axis=0), axis=0)  # [x_max, y_max]
        x_min = tf.reduce_min(tf.reduce_min(cell_points, axis=0), axis=0)  # [x_min, y_min]

        x_max = tf.reshape(x_max, [1, 2])
        x_min = tf.reshape(x_min, [1, 2])

        # convert into data type
        self.input_scaling_min = tf.cast(x_min, dtype=self.dtype)
        self.input_scaling_max = tf.cast(x_max, dtype=self.dtype)

        # return the values
        return self.input_scaling_min, self.input_scaling_max
