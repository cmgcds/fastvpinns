"""
This module, `geometry_2d.py`, contains the `Geometry_2D` class which defines functions to read mesh from Gmsh and 
generate internal mesh for 2D problems. It supports different types of meshes and mesh generation methods. 
The class also allows for specifying the number of test points in the x and y directions, and the output folder for storing results.

Author: Thivin Anandh D
Date: 21-Sep-2023
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import meshio
from pyDOE import lhs

import gmsh

from .geometry import Geometry


class Geometry_2D(Geometry):
    """
    Defines functions to read mesh from Gmsh and internal mesh for 2D problems.

    :param mesh_type: The type of mesh to be used.
    :type mesh_type: str
    :param mesh_generation_method: The method used to generate the mesh.
    :type mesh_generation_method: str
    :param n_test_points_x: The number of test points in the x-direction.
    :type n_test_points_x: int
    :param n_test_points_y: The number of test points in the y-direction.
    :type n_test_points_y: int
    :param output_folder: The path to the output folder.
    :type output_folder: str
    """

    def __init__(
        self,
        mesh_type: str,
        mesh_generation_method: str,
        n_test_points_x: int,
        n_test_points_y: int,
        output_folder: str,
    ):
        """
        Constructor for Geometry_2D class.
        :param mesh_type: The type of mesh to be used.
        :type mesh_type: str
        :param mesh_generation_method: The method used to generate the mesh.
        :type mesh_generation_method: str
        :param n_test_points_x: The number of test points in the x-direction.
        :type n_test_points_x: int
        :param n_test_points_y: The number of test points in the y-direction.
        :type n_test_points_y: int
        :param output_folder: The path to the output folder.
        :type output_folder: str
        """
        # Call the super class constructor
        super().__init__(mesh_type, mesh_generation_method)
        self.mesh_type = mesh_type
        self.mesh_generation_method = mesh_generation_method
        self.n_test_points_x = n_test_points_x
        self.n_test_points_y = n_test_points_y
        self.output_folder = output_folder

        # to be filled in mesh/grid generation routines
        self.bd_cell_info_dict = None
        self.bd_joint_info_dict = None

        if self.mesh_generation_method not in ["internal", "external"]:
            print(
                f"Invalid mesh generation method {self.mesh_generation_method} in {self.__class__.__name__} from {__name__}."
            )
            raise ValueError("Mesh generation method should be either internal or external.")

        if self.mesh_type not in ["quadrilateral"]:
            print(
                f"Invalid mesh type {self.mesh_type} in {self.__class__.__name__} from {__name__}."
            )
            raise ValueError("Mesh type should be quadrilateral only.")

        # To be filled - only when mesh is internal
        self.n_cells_x = None
        self.n_cells_y = None
        self.x_limits = None
        self.y_limits = None

        # to be filled by external
        self.mesh_file_name = None
        self.mesh = None
        self.bd_dict = None
        self.cell_points = None
        self.test_points = None

    def find_cells_containing_indices(self, cells, index):
        """
        Helper function, which is used to identify in which cells, the current edge co-ordinates are present in the mesh
        """
        # Initialize an empty list to store the cell indices containing the given indices
        containing_cells = []

        # Iterate through each cell in the cells list
        for i, cell in enumerate(cells):
            # Check if both indices are present in the current cell
            if all(idx in cell for idx in index):
                containing_cells.append(i)

        return containing_cells

    def read_mesh(
        self,
        mesh_file: str,
        boundary_point_refinement_level: int,
        bd_sampling_method: str,
        refinement_level: int,
    ):
        """
        Reads mesh from a Gmsh .msh file and extracts cell information.

        :param mesh_file: The path to the mesh file.
        :type mesh_file: str
        :param boundary_point_refinement_level: The number of boundary points to be generated.
        :type boundary_point_refinement_level: int
        :param bd_sampling_method: The method used to generate the boundary points.
        :type bd_sampling_method: str
        :param refinement_level: The number of times the mesh should be refined.
        :type refinement_level: int
        :return: The cell points and the dictionary of boundary points.
        :rtype: tuple
        """

        self.mesh_file_name = mesh_file

        # bd_sampling_method = "uniform"  # "uniform" or "lhs"

        file_extension = Path(mesh_file).suffix

        if file_extension != ".mesh":
            raise ValueError("Mesh file should be in .mesh format.")

        # Read mesh using meshio
        self.mesh = meshio.read(self.mesh_file_name)

        if self.mesh_type == "quadrilateral":
            # Extract cell information
            cells = self.mesh.cells_dict["quad"]

        num_cells = cells.shape[0]
        print(f"[INFO] : Number of cells = {num_cells}")
        cell_points = self.mesh.points[cells][
            :, :, 0:2
        ]  # remove the z coordinate, which is 0 for all points

        # loop over all cells and rearrange the points in anticlockwise direction
        for i in range(num_cells):
            cell = cell_points[i]
            # get the centroid of the cell
            centroid = np.mean(cell, axis=0)
            # get the angle of each point with respect to the centroid
            angles = np.arctan2(cell[:, 1] - centroid[1], cell[:, 0] - centroid[0])

            # collect the sorted index based on the angles
            sorted_index = np.argsort(angles)

            # sort the points based on the angles
            cell_points[i] = cell[sorted_index]

            # similarly sort the cells
            cells[i] = cells[i][sorted_index]

        # Extract number of points within each cell
        print(f"[INFO] : Number of points per cell = {cell_points.shape}")

        # Collect the Boundary point id's within the domain
        boundary_edges = self.mesh.cells_dict["line"]

        # Using the point id, collect the coordinates of the boundary points
        boundary_coordinates = self.mesh.points[boundary_edges]

        # Number of Existing Boundary points
        # print(
        #     f"[INFO] : Number of Bound points before refinement = {np.unique(boundary_coordinates.reshape(-1,3)).shape[0] * 0.5 + 1}"
        # )

        # now Get the physical tag of the boundary edges
        boundary_tags = self.mesh.cell_data["medit:ref"][0]

        # Generate a Dictionary of boundary tags and boundary coordinates
        # Keys will be the boundary tags and values will be the list of coordinates
        boundary_dict = {}
        boundary_cell_info_dict = {}
        boundary_joint_info_dict = {}

        # refine the boundary points based on the number of boundary points needed
        for i in range(boundary_coordinates.shape[0]):

            # obtain the point index of the boundary points
            edge_index = self.mesh.cells_dict["line"][i]

            # the index will be a list of two indices, Now check in which index, both of these index are present in list of cell indices
            cell_index_of_edge = self.find_cells_containing_indices(cells, edge_index)

            # print(f"[INFO] : Neumann Boundary Edge {i} is present in cells {cell_index_of_edge}")

            # Throw error, if the boundary edge is either not present in any cell or present in more than one cell
            if len(cell_index_of_edge) != 1:
                print(
                    f"[Error] : Boundary edge {i} is not present in any cell or present in more than one cell."
                )
                raise ValueError("Boundary edge should be present in exactly one cell.")

            # get the cell index
            cell_index_of_edge = cell_index_of_edge[0]

            # now look for orientation of the edge within the given cell
            # based on the orientation, we can assign the edge and assign the points p1 and p2
            edge_pt_0_index_in_cell = np.where(cells[cell_index_of_edge] == edge_index[0])[0][0]
            edge_pt_1_index_in_cell = np.where(cells[cell_index_of_edge] == edge_index[1])[0][0]
            index_list = [edge_pt_0_index_in_cell, edge_pt_1_index_in_cell]

            # the below are the valid combinations of indices for an edge based on how quad cells are arranged
            # the Quad points are already rearranged in anticlockwise direction, to match the orientation from ParMooN.
            valid_combinations_quad_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
            valid_combination_inverted_quad_edges = [[1, 0], [2, 1], [3, 2], [0, 3]]

            is_valid_index = False  # entry check
            is_valid_reverse_index = False  # entry check

            if index_list in valid_combinations_quad_edges:
                is_valid_index = True
                # joint id is where the index is present in the valid_combinations_quad_edges
                joint_id = valid_combinations_quad_edges.index(index_list)
            elif index_list in valid_combination_inverted_quad_edges:
                is_valid_reverse_index = True
                joint_id = valid_combination_inverted_quad_edges.index(index_list)
            else:
                is_valid_index = False
                is_valid_reverse_index = False

            if is_valid_index:
                p1 = cell_points[cell_index_of_edge][edge_pt_0_index_in_cell, :]
                p2 = cell_points[cell_index_of_edge][edge_pt_1_index_in_cell, :]
            elif is_valid_reverse_index:
                p1 = cell_points[cell_index_of_edge][edge_pt_1_index_in_cell, :]
                p2 = cell_points[cell_index_of_edge][edge_pt_0_index_in_cell, :]
            else:
                print(
                    f"[Error] : Invalid combination of Edge indices on cell number {cell_index_of_edge}"
                )
                print(f"[Error] : The edge indices are {index_list}")
                print(
                    f"[Error] : The valid combinations are {valid_combinations_quad_edges} and valid inverted indices are {valid_combination_inverted_quad_edges}"
                )
                raise ValueError(
                    "Invalid combination of Edge indices on cell number {cell_index_of_edge}"
                )

            if bd_sampling_method == "uniform":
                # take the current point and next point and then perform a uniform sampling
                new_points = np.linspace(p1, p2, pow(2, boundary_point_refinement_level) + 1)
            elif bd_sampling_method == "lhs":
                # take the current point and next point and then perform a uniform sampling
                new_points = lhs(2, pow(2, boundary_point_refinement_level) + 1)
                new_points[:, 0] = new_points[:, 0] * (p2[0] - p1[0]) + p1[0]
                new_points[:, 1] = new_points[:, 1] * (p2[1] - p1[1]) + p1[1]
            else:
                print(
                    f"Invalid sampling method {bd_sampling_method} in {self.__class__.__name__} from {__name__}."
                )
                raise ValueError("Sampling method should be either uniform or lhs.")

            # fill the Cell information and joint information
            new_points_cells = np.ones_like(new_points[:, 0]) * cell_index_of_edge
            new_points_joints = np.ones_like(new_points[:, 0]) * joint_id

            # conver the new points cells and new points joints to int
            new_points_cells = new_points_cells.astype(np.int64)
            new_points_joints = new_points_joints.astype(np.int64)

            # get the boundary tag
            tag = boundary_tags[i]

            if tag not in boundary_dict:
                boundary_dict[tag] = new_points
                boundary_cell_info_dict[tag] = new_points_cells
                boundary_joint_info_dict[tag] = new_points_joints
            else:
                current_val = new_points
                prev_val = boundary_dict[tag]
                final = np.vstack([prev_val, current_val])
                boundary_dict[tag] = final

                # append the current cell info to the existing cell info
                boundary_cell_info_dict[tag] = np.append(
                    boundary_cell_info_dict[tag], new_points_cells
                )

                # append the current joint info to the existing joint info
                boundary_joint_info_dict[tag] = np.append(
                    boundary_joint_info_dict[tag], new_points_joints
                )

        # get unique
        for tag, val in boundary_dict.items():
            # find the unique values and also their indices
            unique_val, unique_idx = np.unique(val, axis=0, return_index=True)

            # based on the unique indices, get the cell and joint information
            cell_info = boundary_cell_info_dict[tag][unique_idx]
            joint_info = boundary_joint_info_dict[tag][unique_idx]
            boundary_dict[tag] = unique_val
            boundary_cell_info_dict[tag] = cell_info
            boundary_joint_info_dict[tag] = joint_info

        self.bd_dict = boundary_dict
        self.bd_cell_info_dict = boundary_cell_info_dict
        self.bd_joint_info_dict = boundary_joint_info_dict
        # print the new boundary points  on each boundary tag (key) in a tabular format

        total_bound_points = 0
        print(f"| {'Boundary ID':<12} | {'Number of Points':<16} |")
        print(f"| {'-'*12:<12}---{'-'*16:<16} |")
        for k, v in self.bd_dict.items():
            print(f"| {k:<12} | {v.shape[0]:<16} |")
            total_bound_points += v.shape[0]

        print(f"[INFO] : No of bound pts after refinement:  {total_bound_points}")

        # Assign to class values
        self.cell_points = cell_points

        # generate testvtk
        self.generate_vtk_for_test()

        return cell_points, self.bd_dict

    def generate_quad_mesh_internal(
        self,
        x_limits: tuple,
        y_limits: tuple,
        n_cells_x: int,
        n_cells_y: int,
        num_boundary_points: int,
    ):
        """
        Generate and save a quadrilateral mesh with physical curves.

        :param x_limits: The lower and upper limits in the x-direction (x_min, x_max).
        :type x_limits: tuple
        :param y_limits: The lower and upper limits in the y-direction (y_min, y_max).
        :type y_limits: tuple
        :param n_cells_x: The number of cells in the x-direction.
        :type n_cells_x: int
        :param n_cells_y: The number of cells in the y-direction.
        :type n_cells_y: int
        :param num_boundary_points: The number of boundary points.
        :type num_boundary_points: int
        :return: The cell points and the dictionary of boundary points.
        :rtype: tuple[numpy.ndarray, dict]
        """

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.x_limits = x_limits
        self.y_limits = y_limits

        # generate linspace of points in x and y direction
        x = np.linspace(x_limits[0], x_limits[1], n_cells_x + 1)
        y = np.linspace(y_limits[0], y_limits[1], n_cells_y + 1)

        # Generate quad cells from the points
        # the output should be a list of 4 points for each cell , each being a list of 2 points [x,y]
        cells = []

        for i in range(n_cells_x):
            for j in range(n_cells_y):
                # get the four points of the cell
                p1 = [x[i], y[j]]
                p2 = [x[i + 1], y[j]]
                p3 = [x[i + 1], y[j + 1]]
                p4 = [x[i], y[j + 1]]

                # append the points to the cells
                cells.append([p1, p2, p3, p4])

        # convert to numpy array
        cells = np.array(cells, dtype=np.float64)

        # use arctan2 to sort the points in anticlockwise direction
        # loop over all cells and rearrange the points in anticlockwise direction
        for i in range(cells.shape[0]):
            cell = cells[i]
            # get the centroid of the cell
            centroid = np.mean(cell, axis=0)
            # get the angle of each point with respect to the centroid
            angles = np.arctan2(cell[:, 1] - centroid[1], cell[:, 0] - centroid[0])
            # sort the points based on the angles
            cells[i] = cell[np.argsort(angles)]

        # generate a meshio mesh object using the cells
        self.mesh = meshio.Mesh(points=cells.reshape(-1, 2), cells=[("quad", cells.reshape(-1, 4))])

        # lets generate the boundary points, this function will return a dictionary of boundary points
        # the keys will be the boundary tags and values will be the list of boundary points
        bd_points = {}
        bd_cells = {}
        bd_joints = {}

        num_bound_per_side = int(num_boundary_points / 4)

        def _temp_bd_func(start, end, num_pts):
            """
            This function returns the boundary points between the start and end points
            using lhs sampling.

            :param start: The starting point of the boundary.
            :type start: float
            :param end: The ending point of the boundary.
            :type end: float
            :param num_pts: The number of points to generate.
            :type num_pts: int
            :return: The boundary points as a 1D numpy array.
            :rtype: numpy.ndarray
            """
            # generate the boundary points using lhs as a np.float64 array
            bd_pts = lhs(1, num_pts).astype(np.float64)
            # scale the points
            bd_pts = bd_pts * (end - start) + start

            return bd_pts.reshape(-1)

        # For internal mesh generation routines, the cell values are hardcoded to be zero, as the normals are computed
        # based on the boundary id directly.

        # bottom boundary
        y_bottom = (np.ones(num_bound_per_side, dtype=np.float64) * y_limits[0]).reshape(-1)
        x_bottom = _temp_bd_func(x_limits[0], x_limits[1], num_bound_per_side)
        bd_points[1000] = np.vstack([x_bottom, y_bottom]).T
        bd_cells[1000] = np.ones_like(x_bottom, dtype=np.int64) * 0
        bd_joints[1000] = np.ones_like(x_bottom, dtype=np.int64) * 0

        # right boundary
        x_right = (np.ones(num_bound_per_side, dtype=np.float64) * x_limits[1]).reshape(-1)
        y_right = _temp_bd_func(y_limits[0], y_limits[1], num_bound_per_side)
        bd_points[1001] = np.vstack([x_right, y_right]).T
        bd_cells[1001] = np.ones_like(x_right, dtype=np.int64) * 0
        bd_joints[1001] = np.ones_like(x_right, dtype=np.int64) * 1

        # top boundary
        y_top = (np.ones(num_bound_per_side, dtype=np.float64) * y_limits[1]).reshape(-1)
        x_top = _temp_bd_func(x_limits[0], x_limits[1], num_bound_per_side)
        bd_points[1002] = np.vstack([x_top, y_top]).T
        bd_cells[1002] = np.ones_like(x_top, dtype=np.int64) * 0
        bd_joints[1002] = np.ones_like(x_top, dtype=np.int64) * 2

        # left boundary
        x_left = (np.ones(num_bound_per_side, dtype=np.float64) * x_limits[0]).reshape(-1)
        y_left = _temp_bd_func(y_limits[0], y_limits[1], num_bound_per_side)
        bd_points[1003] = np.vstack([x_left, y_left]).T
        bd_cells[1003] = np.ones_like(x_left, dtype=np.int64) * 0
        bd_joints[1003] = np.ones_like(x_left, dtype=np.int64) * 3

        # Assign local variables to class variables
        self.cell_points = cells
        self.bd_dict = bd_points
        self.bd_cell_info_dict = bd_cells
        self.bd_joint_info_dict = bd_joints

        # generate vtk
        self.generate_vtk_for_test()

        return self.cell_points, self.bd_dict

    def generate_vtk_for_test(self):
        """
        Generates a VTK from Mesh file (External) or using gmsh (for Internal).

        :return: None
        """

        if self.mesh_generation_method == "internal":
            # initialise the mesh
            gmsh.initialize()

            # Now, lets generate the mesh with the points.
            x_range = self.x_limits[1] - self.x_limits[0]
            y_range = self.y_limits[1] - self.y_limits[0]

            mesh_size_x = x_range / self.n_test_points_x
            mesh_size_y = y_range / self.n_test_points_y

            # generate a gmsh with the given parameters
            Xmin = self.x_limits[0]
            Xmax = self.x_limits[1]
            Ymin = self.y_limits[0]
            Ymax = self.y_limits[1]

            point1 = gmsh.model.geo.add_point(Xmin, Ymin, 0, mesh_size_x)
            point2 = gmsh.model.geo.add_point(Xmax, Ymin, 0, mesh_size_x)
            point3 = gmsh.model.geo.add_point(Xmax, Ymax, 0, mesh_size_y)
            point4 = gmsh.model.geo.add_point(Xmin, Ymax, 0, mesh_size_y)

            line1 = gmsh.model.geo.add_line(point1, point2, 1000)  ## Bottom
            line2 = gmsh.model.geo.add_line(point2, point3, 1001)  ## Right
            line3 = gmsh.model.geo.add_line(point3, point4, 1002)  ## Top
            line4 = gmsh.model.geo.add_line(point4, point1, 1003)  ## Left

            face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])

            gmsh.model.geo.add_plane_surface([face1])

            # Create the relevant Gmsh data structures
            # from Gmsh model.
            gmsh.model.geo.synchronize()

            # Generate mesh:
            gmsh.model.mesh.generate()

            mesh_file_name = Path(self.output_folder) / "internal.msh"
            vtk_file_name = Path(self.output_folder) / "internal.vtk"

            gmsh.write(str(mesh_file_name))
            print("[INFO] : Internal mesh file generated at ", str(mesh_file_name))

            # close the gmsh
            gmsh.finalize()

            # read the mesh using meshio
            mesh = meshio.gmsh.read(str(mesh_file_name))
            meshio.vtk.write(str(vtk_file_name), mesh, binary=False, fmt_version="4.2")

            print("[INFO] : VTK file for internal mesh file generated at ", str(mesh_file_name))

        elif self.mesh_generation_method == "external":

            vtk_file_name = Path(self.output_folder) / "external.vtk"

            # Use the internal mesh to generate the vtk file
            mesh = meshio.read(str(self.mesh_file_name))
            meshio.vtk.write(str(vtk_file_name), mesh, binary=False, fmt_version="4.2")

            print("[INFO] : VTK file for external mesh file generated at ", str(vtk_file_name))

    def get_test_points(self):
        """
        This function is used to extract the test points from the given mesh

        Parameters:
        None

        Returns:
        test_points (numpy.ndarray): The test points for the given domain
        """

        if self.mesh_generation_method == "internal":
            # vtk_file_name  = Path(self.output_folder) / "internal.vtk"
            # code over written to plot from np.linspace instead of vtk file
            # generate linspace of points in x and y direction based on x and y limits
            x = np.linspace(self.x_limits[0], self.x_limits[1], self.n_test_points_x)
            y = np.linspace(self.y_limits[0], self.y_limits[1], self.n_test_points_y)
            # generate meshgrid
            x_grid, y_grid = np.meshgrid(x, y)
            # stack the points
            self.test_points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T

            return self.test_points

        elif self.mesh_generation_method == "external":
            vtk_file_name = Path(self.output_folder) / "external.vtk"

        mesh = meshio.read(str(vtk_file_name))
        points = mesh.points
        return points[:, 0:2]  # return only first two columns

    def write_vtk(self, solution, output_path, filename, data_names):
        """
        Writes the data to a VTK file.

        :param solution: The solution vector.
        :type solution: numpy.ndarray
        :param output_path: The path to the output folder.
        :type output_path: str
        :param filename: The name of the output file.
        :type filename: str
        :param data_names: The list of data names in the VTK file to be written as scalars.
        :type data_names: list
        :return: None
        """
        # read the existing vtk into file
        if self.mesh_generation_method == "internal":
            vtk_file_name = Path(self.output_folder) / "internal.vtk"
        elif self.mesh_generation_method == "external":
            vtk_file_name = Path(self.output_folder) / "external.vtk"

        data = []
        with open(vtk_file_name, "r", encoding="utf-8") as File:
            for line in File:
                data.append(line)

        # get the output file name
        output_file_name = Path(output_path) / filename

        if solution.shape[1] != len(data_names):
            print("[Error] : File : geometry_2d.py, Function: write_vtk")
            print(
                "Num Columns in solution = ",
                solution.shape[1],
                " Num of data names = ",
                len(data_names),
            )
            raise ValueError("Number of data names and solution columns are not equal")

        # write the data to the output file
        with open(str(output_file_name), "w", encoding="utf-8") as FN:
            for line in data:
                FN.write(line)
                if "POINT_DATA" in line.strip():
                    break

            for i in range(solution.shape[1]):
                FN.write("SCALARS " + data_names[i] + " float\n")
                FN.write("LOOKUP_TABLE default\n")
                np.savetxt(FN, solution[:, i])
                FN.write("\n")

        # save the vtk file as image
        # self.save_vtk_as_image(str(output_file_name), data_names)

    def plot_adaptive_mesh(
        self, cells_list, area_averaged_cell_loss_list, epoch, filename="cell_residual"
    ):
        """
        Plots the residuals in each cell of the mesh.

        :param cells_list: The list of cells.
        :type cells_list: list
        :param area_averaged_cell_loss_list: The list of area averaged cell residual (or the normal residual).
        :type area_averaged_cell_loss_list: list
        :param epoch: The epoch number (for file name).
        :type epoch: int
        :param filename: The name of the output file, defaults to "cell_residual".
        :type filename: str, optional
        :return: None
        """

        plt.figure(figsize=(6.4, 4.8), dpi=300)

        # normalise colors
        norm = mcolors.Normalize(
            vmin=np.min(area_averaged_cell_loss_list), vmax=np.max(area_averaged_cell_loss_list)
        )

        # Create a colormap
        colormap = plt.cm.jet

        for index, cell in enumerate(cells_list):
            x = cell[:, 0]
            y = cell[:, 1]

            x = np.append(x, x[0])
            y = np.append(y, y[0])

            curr_cell_loss = float(area_averaged_cell_loss_list[index])

            color = colormap(norm(curr_cell_loss))

            plt.fill(x, y, color=color, alpha=0.9)

            plt.plot(x, y, "k")

            # # compute x_min, x_max, y_min, y_max
            # x_min = np.min(x)
            # x_max = np.max(x)
            # y_min = np.min(y)
            # y_max = np.max(y)

            # # compute centroid of the cells
            # centroid = np.array([np.mean(x), np.mean(y)])

            # plot the loss text within the cell
            # plt.text(centroid[0], centroid[1], f"{curr_cell_loss:.3e}", fontsize=16, horizontalalignment='center', verticalalignment='center')

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)

        # output filename
        output_filename = Path(f"{self.output_folder}/{filename}_{epoch}.png")
        plt.title(f"Cell Residual")
        plt.savefig(str(output_filename), dpi=300)
