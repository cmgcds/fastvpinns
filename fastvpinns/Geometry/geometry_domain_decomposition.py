import numpy as np
from .geometry_2d import Geometry_2D
from ..utils.print_utils import print_table, print_table_multicolumns


class GeometryDomainDecomposition(Geometry_2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.mesh_generation_method == 'domain_decomposition_uniform'
        ), 'GeometryDomainDecomposition can only be used with domain_decomposition_uniform as the mesh generation method.'
        pass

    def generate_subdomains(
        self, x_limits, y_limits, n_cells_x: int, n_cells_y: int, kernel_size_dict: dict
    ):
        """
        Generate cells and blocks using a convolutional approach.

        Args:
            x_limits: Tuple containing the lower and upper limits in the x-direction.
            y_limits: Tuple containing the lower and upper limits in the y-direction.
            n_cells_x: Number of cells in the x-direction.
            n_cells_y: Number of cells in the y-direction.
            kernel_size_row: Number of cells in the row direction of the kernel.
            kernel_size_col: Number of cells in the column direction of the kernel.
            stride_row: Number of cells to skip in the row direction.
            stride_col: Number of cells to skip in the column direction.


        Returns:
            blocks: List of blocks.
            grid_x: List of x-coordinates of the grid.
            grid_y: List of y-coordinates of the grid.
        """
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        n_cells = self.n_cells_x * self.n_cells_y
        self.cells_in_domain = np.arange(n_cells)
        self.cells_in_domain_2d = np.arange(n_cells).reshape((n_cells_x, n_cells_y))
        self.cells_in_domain_2d = np.flip(self.cells_in_domain_2d, axis=0)

        self.kernel_size_x = kernel_size_dict['kernel_size_x']
        self.kernel_size_y = kernel_size_dict['kernel_size_y']
        self.stride_x = kernel_size_dict['stride_x']
        self.stride_y = kernel_size_dict['stride_y']

        self.subdomains = []
        x, y = self.cells_in_domain_2d.shape
        for i in range(0, x - self.kernel_size_x + 1, self.stride_x):
            for j in range(0, y - self.kernel_size_y + 1, self.stride_y):
                domain = []
                for k in range(i, i + self.kernel_size_x):
                    for l in range(j, j + self.kernel_size_y):
                        domain.append(self.cells_in_domain_2d[k][l])
                self.subdomains.append(domain)
        # Generate the grid
        self.x_left = x_limits[0]
        self.x_right = x_limits[1]
        self.y_bottom = y_limits[0]
        self.y_top = y_limits[1]

        delta_x = (self.x_right - self.x_left) / n_cells_x
        delta_y = (self.y_top - self.y_bottom) / n_cells_y

        self.grid_x = np.asarray([self.x_left + i * delta_x for i in range(n_cells_x + 1)])
        self.grid_y = np.asarray([self.y_bottom + i * delta_y for i in range(n_cells_y + 1)])

        # calculate span and mean of each subdomain
        self.calculate_subdomain_mean_and_span()

        self.print_domain_info()

        return self.cells_in_domain, self.subdomains, self.grid_x, self.grid_y

    def generate_quad_mesh_with_domain_decomposition(
        self, x_limits: tuple, y_limits: tuple, n_cells_x: int, n_cells_y: int
    ):
        """
        Generate and save a quadrilateral mesh with physical curves.

        Parameters:
        x_limits (tuple): The lower and upper limits in the x-direction (x_min, x_max).
        y_limits (tuple): The lower and upper limits in the y-direction (y_min, y_max).
        n_cells_x (int): The number of cells in the x-direction.
        n_cells_y (int): The number of cells in the y-direction.
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

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # get the four points of the cell
                p1 = [x[j], y[i]]
                p2 = [x[j + 1], y[i]]
                p3 = [x[j + 1], y[i + 1]]
                p4 = [x[j], y[i + 1]]

                # append the points to the cells
                cells.append([p1, p2, p3, p4])

        # Generate the grid

        boundary_limits = []
        for block_id, cells_in_current_block in enumerate(self.subdomains):
            x_min = self.x_right
            x_max = self.x_left
            y_min = self.y_top
            y_max = self.y_bottom
            for cell in cells_in_current_block:
                ex = cell % self.n_cells_x
                ey = cell // self.n_cells_x
                x_min = min(x_min, self.grid_x[ex])
                x_max = max(x_max, self.grid_x[ex + 1])
                y_min = min(y_min, self.grid_y[ey])
                y_max = max(y_max, self.grid_y[ey + 1])

            boundary_limits.append([x_min, x_max, y_min, y_max])

        return cells, boundary_limits

    def assign_sub_domain_coords(self, cells_in_subdomain, cells_points):
        """
        Assigns coordinates to the subdomains.

        Args:
            cells_in_subdomain: List of cells in the subdomain.
            cells_points: List of points in the cells.

        Returns:
            cells_points_subdomain: List of points in the subdomains.
        """
        cells_points_subdomain = []
        for cell in cells_in_subdomain:
            cells_points_subdomain.append(cells_points[cell])
        return np.array(cells_points_subdomain, dtype=np.float64)

    def calculate_subdomain_mean_and_span(self):
        """
        Calculate the mean and span of each subdomain.
        """
        # assert that self.x_right, self.x_left, self.y_top, self.y_bottom are defined
        assert hasattr(self, 'x_right'), 'x_right is not defined'
        assert hasattr(self, 'x_left'), 'x_left is not defined'
        assert hasattr(self, 'y_top'), 'y_top is not defined'
        assert hasattr(self, 'y_bottom'), 'y_bottom is not defined'
        assert hasattr(self, 'n_cells_x'), 'n_cells_x is not defined'
        assert hasattr(self, 'n_cells_y'), 'n_cells_y is not defined'
        assert hasattr(self, 'grid_x'), 'grid_x is not defined'
        assert hasattr(self, 'grid_y'), 'grid_y is not defined'
        assert hasattr(self, 'subdomains'), 'domains is not defined'

        def calculate_subdomain_min_and_max(self):
            """
            Calculate the min and max of each subdomain.
            """
            x_min_list, x_max_list, y_min_list, y_max_list = [], [], [], []
            for subdomain in self.subdomains:
                x_min = self.x_right
                x_max = self.x_left
                y_min = self.y_top
                y_max = self.y_bottom

                for cell in subdomain:
                    ex = cell % self.n_cells_x
                    ey = cell // self.n_cells_x
                    x_min = min(x_min, self.grid_x[ex])
                    x_max = max(x_max, self.grid_x[ex + 1])
                    y_min = min(y_min, self.grid_y[ey])
                    y_max = max(y_max, self.grid_y[ey + 1])

                x_min_list.append(x_min)
                x_max_list.append(x_max)
                y_min_list.append(y_min)
                y_max_list.append(y_max)

            return x_min_list, x_max_list, y_min_list, y_max_list

        self.x_min_list, self.x_max_list, self.y_min_list, self.y_max_list = (
            calculate_subdomain_min_and_max(self)
        )

        self.x_mean_list = 0.5 * (np.array(self.x_min_list) + np.array(self.x_max_list))
        self.y_mean_list = 0.5 * (np.array(self.y_min_list) + np.array(self.y_max_list))
        self.x_span_list = np.array(self.x_max_list) - np.array(self.x_min_list)
        self.y_span_list = np.array(self.y_max_list) - np.array(self.y_min_list)

    def print_domain_info(self):
        print(f"Number of subdomains = {len(self.subdomains)}")

        title = "Subdomain Information"
        cols = ["Subdomain ID", "Cell IDs"]
        print_table(
            title,
            cols,
            [str(domain_no) for domain_no in range(len(self.subdomains))],
            self.subdomains,
        )

        title = "Subdomain Params"
        cols = [
            "Subdomain ID",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "x_mean",
            "y_mean",
            "x_span",
            "y_span",
        ]
        print_table_multicolumns(
            title,
            cols,
            [
                list(range(len(self.subdomains))),
                self.x_min_list,
                self.x_max_list,
                self.y_min_list,
                self.y_max_list,
                self.x_mean_list,
                self.y_mean_list,
                self.x_span_list,
                self.y_span_list,
            ],
        )

        return None
