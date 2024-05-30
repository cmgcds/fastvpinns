"""
The file `fespace2d.py` contains the main class that holds the information of all the 
Finite Element (FE) spaces of all the cells within the given mesh.

Author: Thivin Anandh D

Changelog: 30/Aug/2023 - Initial version

Known issues: None

Dependencies: None specified
"""

import numpy as np
import meshio
from .FE2D_Cell import FE2D_Cell

# from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm

# import plotting
import matplotlib.pyplot as plt

# import path
from pathlib import Path

# import tensorflow
import tensorflow as tf

from ..utils.print_utils import print_table

from pyDOE import lhs
import pandas as pd

from matplotlib import rc
from cycler import cycler


plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20

plt.rcParams["legend.fontsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "darkblue",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#bcbd22",
        "#8c564b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#7f7f7f",
    ]
)

from .fespace import Fespace


class Fespace2D(Fespace):
    """
    Represents a finite element space in 2D.

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
        generate_mesh_plot: bool = False,
    ) -> None:
        """
        The constructor of the Fespace2D class.
        """
        # call the constructor of the parent class
        super().__init__(
            mesh=mesh,
            cells=cells,
            boundary_points=boundary_points,
            cell_type=cell_type,
            fe_order=fe_order,
            fe_type=fe_type,
            quad_order=quad_order,
            quad_type=quad_type,
            fe_transformation_type=fe_transformation_type,
            bound_function_dict=bound_function_dict,
            bound_condition_dict=bound_condition_dict,
            forcing_function=forcing_function,
            output_path=output_path,
        )

        if self.cell_type == "triangle":
            raise ValueError(
                "Triangle Mesh is not supported yet"
            )  # added by thivin - to remove support for triangular mesh

        self.generate_mesh_plot = generate_mesh_plot

        # to be calculated in the plot function
        self.total_dofs = 0
        self.total_boundary_dofs = 0

        # to be calculated on get_boundary_data_dirichlet function
        self.total_dirichlet_dofs = 0

        # get the number of cells
        self.n_cells = self.cells.shape[0]

        self.fe_cell = []

        # Function which assigns the fe_cell for each cell
        self.set_finite_elements()

        # generate the plot of the mesh
        if self.generate_mesh_plot:
            self.generate_plot(self.output_path)
        # self.generate_plot(self.output_path)

        # Obtain boundary Data
        self.dirichlet_boundary_data = self.generate_dirichlet_boundary_data()

        title = [
            "Number of Cells",
            "Number of Quadrature Points",
            "Number of Dirichlet Boundary Points",
            "Quadrature Order",
            "FE Order",
            "FE Type",
            "FE Transformation Type",
        ]
        values = [
            self.n_cells,
            self.total_dofs,
            self.total_dirichlet_dofs,
            self.quad_order,
            self.fe_order,
            self.fe_type,
            self.fe_transformation_type,
        ]
        # print the table
        print_table("FE Space Information", ["Property", "Value"], title, values)

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
        progress_bar = tqdm(
            total=self.n_cells,
            desc="Fe2D_cell Setup",
            unit="cells_assembled",
            bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
            colour="blue",
            ncols=100,
        )

        dof = 0
        for i in range(self.n_cells):
            self.fe_cell.append(
                FE2D_Cell(
                    self.cells[i],
                    self.cell_type,
                    self.fe_order,
                    self.fe_type,
                    self.quad_order,
                    self.quad_type,
                    self.fe_transformation_type,
                    self.forcing_function,
                )
            )

            # obtain the shape of the basis function (n_test, N_quad)
            dof += self.fe_cell[i].basis_at_quad.shape[1]

            progress_bar.update(1)
        # print the Shape details of all the matrices from cell 0 using print_table function
        title = [
            "Shape function Matrix Shape",
            "Shape function Gradient Matrix Shape",
            "Jacobian Matrix Shape",
            "Quadrature Points Shape",
            "Quadrature Weights Shape",
            "Quadrature Actual Coordinates Shape",
            "Forcing Function Shape",
        ]
        values = [
            self.fe_cell[0].basis_at_quad.shape,
            self.fe_cell[0].basis_gradx_at_quad.shape,
            self.fe_cell[0].jacobian.shape,
            self.fe_cell[0].quad_xi.shape,
            self.fe_cell[0].quad_weight.shape,
            self.fe_cell[0].quad_actual_coordinates.shape,
            self.fe_cell[0].forcing_at_quad.shape,
        ]
        print_table("FE Matrix Shapes", ["Matrix", "Shape"], title, values)

        # update the total number of dofs
        self.total_dofs = dof

    def generate_plot(self, output_path) -> None:
        """
        Generate a plot of the mesh.

        :param output_path: The path to save the generated plot.
        :type output_path: str
        """
        total_quad = 0
        marker_list = [
            "o",
            ".",
            ",",
            "x",
            "+",
            "P",
            "s",
            "D",
            "d",
            "^",
            "v",
            "<",
            ">",
            "p",
            "h",
            "H",
        ]

        print(f"[INFO] : Generating the plot of the mesh")
        # Plot the mesh
        plt.figure(figsize=(6.4, 4.8), dpi=300)

        # label flag ( to add the label only once)
        label_set = False

        # plot every cell as a quadrilateral
        # loop over all the cells
        for i in range(self.n_cells):
            # get the coordinates of the cell
            x = self.fe_cell[i].cell_coordinates[:, 0]
            y = self.fe_cell[i].cell_coordinates[:, 1]

            # add the first point to the end of the array
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            plt.plot(x, y, "k-", linewidth=0.5)

            # plot the quadrature points
            x_quad = self.fe_cell[i].quad_actual_coordinates[:, 0]
            y_quad = self.fe_cell[i].quad_actual_coordinates[:, 1]

            total_quad += x_quad.shape[0]

            if not label_set:
                plt.scatter(x_quad, y_quad, marker="x", color="b", s=2, label="Quad Pts")
                label_set = True
            else:
                plt.scatter(x_quad, y_quad, marker="x", color="b", s=2)

        self.total_dofs = total_quad

        bound_dof = 0
        # plot the boundary points
        # loop over all the boundary tags
        for i, (bound_id, bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            x = bound_pts[:, 0]
            y = bound_pts[:, 1]

            # add the first point to the end of the array
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            bound_dof += x.shape[0]

            plt.scatter(x, y, marker=marker_list[i + 1], s=2, label=f"Bd-id : {bound_id}")

        self.total_boundary_dofs = bound_dof

        plt.legend(bbox_to_anchor=(0.85, 1.02))
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(str(Path(output_path) / "mesh.png"), bbox_inches="tight")
        plt.savefig(str(Path(output_path) / "mesh.svg"), bbox_inches="tight")

        # print the total number of quadrature points
        print(f"Plots generated")
        print(f"[INFO] : Total number of cells = {self.n_cells}")
        print(f"[INFO] : Total number of quadrature points = {self.total_dofs}")
        print(f"[INFO] : Total number of boundary points = {self.total_boundary_dofs}")

    def generate_dirichlet_boundary_data(self) -> np.ndarray:
        """
        Generate Dirichlet boundary data.

        This function returns the boundary points and their corresponding values.

        :return: A tuple containing two arrays:
            - The first array contains the boundary points as numpy arrays.
            - The second array contains the values of the boundary points as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        x = []
        y = []
        for bound_id, bound_pts in self.boundary_points.items():
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0], pt[1]], dtype=np.float64)
                x.append(pt_new)
                val = np.array(
                    self.bound_function_dict[bound_id](pt[0], pt[1]), dtype=np.float64
                ).reshape(-1, 1)
                y.append(val)

        print(f"[INFO] : Total number of Dirichlet boundary points = {len(x)}")
        self.total_dirichlet_dofs = len(x)
        print(f"[INFO] : Shape of Dirichlet-X = {np.array(x).shape}")
        print(f"[INFO] : Shape of Y = {np.array(y).shape}")

        return x, y

    def generate_dirichlet_boundary_data_vector(self, component) -> np.ndarray:
        """
        Generate the boundary data vector for the Dirichlet boundary condition.

        This function returns the boundary points and their corresponding values for a specific component.

        :param component: The component for which the boundary data vector is generated.
        :type component: int

        :return: The boundary points and their values as numpy arrays.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        x = []
        y = []
        for bound_id, bound_pts in self.boundary_points.items():
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0], pt[1]], dtype=np.float64)
                x.append(pt_new)
                val = np.array(
                    self.bound_function_dict[bound_id](pt[0], pt[1])[component], dtype=np.float64
                ).reshape(-1, 1)
                y.append(val)

        return x, y

    def get_shape_function_val(self, cell_index) -> np.ndarray:
        """
        Get the actual values of the shape functions on a given cell.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the actual values of the shape functions.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_at_quad.copy()

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
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_gradx_at_quad.copy()

    def get_shape_function_grad_x_ref(self, cell_index) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate on the reference element.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the gradient of the shape function with respect to the x-coordinate.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_gradx_at_quad_ref.copy()

    def get_shape_function_grad_y(self, cell_index) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to y at the given cell index.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: The gradient of the shape function with respect to y.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the total number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_grady_at_quad.copy()

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
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_grady_at_quad_ref.copy()

    def get_quadrature_actual_coordinates(self, cell_index) -> np.ndarray:
        """
        Get the actual coordinates of the quadrature points for a given cell.

        :param cell_index: The index of the cell.
        :type cell_index: int

        :return: An array containing the actual coordinates of the quadrature points.
        :rtype: np.ndarray

        :raises ValueError: If the cell_index is greater than the number of cells.

        :example:
        >>> fespace = FESpace2D()
        >>> fespace.get_quadrature_actual_coordinates(0)
        array([[0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6]])
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].quad_actual_coordinates.copy()

    def get_quadrature_weights(self, cell_index) -> np.ndarray:
        """
        Return the quadrature weights for a given cell.

        :param cell_index: The index of the cell for which the quadrature weights are needed.
        :type cell_index: int
        :return: The quadrature weights for the given cell  of dimension (N_Quad_Points, 1).
        :rtype: np.ndarray
        :raises ValueError: If cell_index is greater than the number of cells.
        Example
        -------
        >>> fespace = FESpace2D()
        >>> weights = fespace.get_quadrature_weights(0)
        >>> print(weights)
        [0.1, 0.2, 0.3, 0.4]
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].mult.copy()

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

        Example usage:
            >>> fespace = FESpace2D()
            >>> cell_index = 0
            >>> forcing_values = fespace.get_forcing_function_values(cell_index)
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        # Changed by Thivin: To assemble the forcing function at the quadrature points here in the fespace
        # so that it can be used to handle multiple dimensions on a vector valud problem

        # get number of shape functions
        n_shape_functions = self.fe_cell[cell_index].basis_function.num_shape_functions

        # Loop over all the basis functions and compute the integral
        f_integral = np.zeros((n_shape_functions, 1), dtype=np.float64)

        for i in range(n_shape_functions):
            val = 0
            for q in range(self.fe_cell[cell_index].basis_at_quad.shape[1]):
                x = self.fe_cell[cell_index].quad_actual_coordinates[q, 0]
                y = self.fe_cell[cell_index].quad_actual_coordinates[q, 1]
                # print("f_values[q] = ",f_values[q])

                # the Jacobian and the quadrature weights are pre multiplied to the basis functions
                val += (self.fe_cell[cell_index].basis_at_quad[i, q]) * self.fe_cell[
                    cell_index
                ].forcing_function(x, y)
                # print("val = ", val)

            f_integral[i] = val

        self.fe_cell[cell_index].forcing_at_quad = f_integral

        return self.fe_cell[cell_index].forcing_at_quad.copy()

    def get_forcing_function_values_vector(self, cell_index, component) -> np.ndarray:
        """
        This function will return the forcing function values at the quadrature points
        based on the Component of the RHS Needed, for vector valued problems

        :param cell_index: The index of the cell
        :type cell_index: int
        :param component: The component of the RHS needed
        :type component: int
        :return: The forcing function values at the quadrature points
        :rtype: np.ndarray
        :raises ValueError: If cell_index is greater than the number of cells
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        # get the coordinates
        x = self.fe_cell[cell_index].quad_actual_coordinates[:, 0]
        y = self.fe_cell[cell_index].quad_actual_coordinates[:, 1]

        # compute the forcing function values
        f_values = self.fe_cell[cell_index].forcing_function(x, y)[component]

        # compute the integral
        f_integral = np.sum(self.fe_cell[cell_index].basis_at_quad * f_values, axis=1)

        self.fe_cell[cell_index].forcing_at_quad = f_integral.reshape(-1, 1)

        return self.fe_cell[cell_index].forcing_at_quad.copy()

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
        # generate random points within the bounds of the domain
        # get the bounds of the domain
        x_min = np.min(self.mesh.points[:, 0])
        x_max = np.max(self.mesh.points[:, 0])
        y_min = np.min(self.mesh.points[:, 1])
        y_max = np.max(self.mesh.points[:, 1])
        # sample n random points within the bounds of the domain
        # Generate points in the unit square

        num_internal_points = int(num_points * 0.9)

        points = lhs(2, samples=num_internal_points)
        points[:, 0] = x_min + (x_max - x_min) * points[:, 0]
        points[:, 1] = y_min + (y_max - y_min) * points[:, 1]
        # get the exact solution at the points
        exact_sol = exact_solution(points[:, 0], points[:, 1])

        # print the shape of the points and the exact solution
        print(f"[INFO] : Number of sensor points = {points.shape[0]}")
        print(f"[INFO] : Shape of sensor points = {points.shape}")

        # plot the points
        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.scatter(points[:, 0], points[:, 1], marker="x", color="r", s=2)
        plt.axis("equal")
        plt.title("Sensor Points")
        plt.tight_layout()
        plt.savefig("sensor_points.png", bbox_inches="tight")

        return points, exact_sol

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
        # use pandas to read the file
        df = pd.read_csv(file_name)

        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        exact_sol = df.iloc[:, 2].values

        # now sample num_points from the data
        indices = np.random.randint(0, x.shape[0], num_points)

        x = x[indices]
        y = y[indices]
        exact_sol = exact_sol[indices]

        # stack them together
        points = np.stack((x, y), axis=1)

        return points, exact_sol
