import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .base import DomainDecomposition
from ...utils.print_utils import print_table
from ...utils.plot_utils import *
import tensorflow as tf


class UniformDomainDecomposition(DomainDecomposition):
    """
    This class will be responsible for:
    1. Decomposing the domain into blocks
    2. Identifying the halo cells
    3. Assigning the blocks to the workers.
    """

    def __init__(self, domain):

        super().__init__(domain)
        assert domain is not None, "Domain object is not set"
        assert domain.subdomains is not None, "Subdomains are not set for the domain object"

        self.domain = domain

        # assert that subdomains have been set for the domain object

        self.subdomains_in_domain = self.domain.subdomains

        self.subdomains_shared_by_cells = self.id_subdomains_shared_by_cells()

        self.halo_cells_in_subdomains = self.id_halo_cells()

        self.window_function_values = {}
        self.params_dict = {}
        self.bilinear_params_dict = {}
        self.subdomain_boundary_limits = {}
        self.non_overlapping_extents = self.find_non_overlapping_extents()
        print("Non overlapping extents: ", self.non_overlapping_extents)


        self.unnormalizing_factor = 0.0
        self.hard_constraints_factor = 0.0

        self.print_details()

        return

    def print_details(self):
        """
        Function which prints the summary of the domain decomposition
        """

        # print the num cells, kernel and stride dimensions in a table using print_table method
        rows = ["N cells x", "N cells y", "Kernel size x", "Kernel size y", "Stride x", "Stride y"]
        values = [
            self.domain.n_cells_x,
            self.domain.n_cells_y,
            self.domain.kernel_size_x,
            self.domain.kernel_size_y,
            self.domain.stride_x,
            self.domain.stride_y,
        ]
        print_table("Domain Decomposition", ["Item", "Values"], rows, values)

        # Print the blocks in the entire domain, this is a dict  of cells within each block
        rows = np.arange(0, len(self.subdomains_in_domain))
        values = self.subdomains_in_domain
        print_table(
            "Blocks in entire domain",
            ["Block ID", "Cells"],
            [str(row) for row in rows],
            [str(value) for value in values],
        )

        # print blocks shared by cells
        rows = self.blocks_shared_by_cells.keys()
        values = self.blocks_shared_by_cells.values()
        print_table(
            "Blocks shared by cells",
            ["Cell ID", "Blocks"],
            [str(row) for row in rows],
            [str(value) for value in values],
        )

        # print halo cells in blocks
        rows = self.halo_cells_in_blocks.keys()
        values = self.halo_cells_in_blocks.values()
        print_table(
            "Halo cells in blocks",
            ["Block ID", "Halo Cells"],
            [str(row) for row in rows],
            [str(value) for value in values],
        )

        return

    def id_subdomains_shared_by_cells(self):
        """
        This function will identify the blocks that are shared by the cells.
        """
        subdomains_shared_by_cells = {}
        for cell in self.domain.cells_in_domain:
            for block_id, block in enumerate(self.subdomains_in_domain):
                if cell in block:
                    if cell in subdomains_shared_by_cells:
                        subdomains_shared_by_cells[cell].append(block_id)
                    else:
                        subdomains_shared_by_cells[cell] = [block_id]

        return subdomains_shared_by_cells

    def id_halo_cells(self):
        """
        This function will identify the halo cells in all blocks.
        """
        halo_cells_in_subdomains = {}
        for block_id, block in enumerate(self.subdomains_in_domain):
            halo_cells = []
            for cell in block:
                if len(self.blocks_shared_by_cells[cell]) > 1:
                    halo_cells.append(cell)
            halo_cells_in_subdomains[block_id] = halo_cells

        return halo_cells_in_subdomains

    def block_normalize(self, block_id, x, y):
        """
        This function will normalize the coordinates of the points in the block.
        """

        x_mean = self.x_mean[block_id]
        x_span = self.x_span[block_id]
        y_mean = self.y_mean[block_id]
        y_span = self.y_span[block_id]

        x_normalized = (x - x_mean) / (x_span / 2.0)
        y_normalized = (y - y_mean) / (y_span / 2.0)

        return x_normalized, y_normalized

    def create_process_pool(self):
        """
        This function will create a process pool.
        """
        num_gpu = len(tf.config.list_physical_devices('GPU'))

        return Pool(num_gpu)

    def initialize_halo_update_dicts(self):
        """
        This function will initialize the dictionaries containing the updates for the halo cells.
        """
        dict_halo_val_update = {}
        dict_halo_gradx_update = {}
        dict_halo_grady_update = {}

        for block_id, block in enumerate(self.subdomains_in_domain):
            dict_halo_val_update[block_id] = {}
            dict_halo_gradx_update[block_id] = {}
            dict_halo_grady_update[block_id] = {}

            # for halo_cell in self.halo_cells_in_blocks[block_id]:
            #     dict_halo_val_update[block_id][halo_cell] =
            #     dict_halo_gradx_update[block_id][halo_cell] =
            #     dict_halo_grady_update[block_id][halo_cell] =

        return

    def find_non_overlapping_extents(self):
        """
        This function will find the extents of the non-overlapping
        sections of the subdomains
        """

        non_overlapping_extents = {}
        halo_cells_in_subdomain = self.halo_cells_in_subdomains
        for block_id, block in enumerate(self.subdomains_in_domain):
            x_min_non_overlap = self.domain.x_right
            x_max_non_overlap = self.domain.x_left
            y_min_non_overlap = self.domain.y_top
            y_max_non_overlap = self.domain.y_bottom
            for cell in block:
                if cell not in halo_cells_in_subdomain[block_id]:
                    print("Cell: ", cell)
                    ex = cell % self.domain.n_cells_x
                    ey = cell // self.domain.n_cells_x
                    x_min_non_overlap = min(x_min_non_overlap, self.domain.grid_x[ex])
                    x_max_non_overlap = max(x_max_non_overlap, self.domain.grid_x[ex + 1])
                    y_min_non_overlap = min(y_min_non_overlap, self.domain.grid_y[ey])
                    y_max_non_overlap = max(y_max_non_overlap, self.domain.grid_y[ey + 1])
            non_overlapping_extents[block_id] = [
                x_min_non_overlap,
                x_max_non_overlap,
                y_min_non_overlap,
                y_max_non_overlap,
            ]
        return non_overlapping_extents

    def update_overlap_values(self, datahandler, fespace):
        """
        Update the overlap values for the given domain decomposition.

        :param domain_decomposition: The domain decomposition.
        :type domain_decomposition: DomainDecomposition
        :param fespace: The finite element space.
        :type fespace: FiniteElementSpace
        """

        total_number_of_cells = len(self.subdomains_shared_by_cells)
        number_of_quad_points = int(fespace[0].fe_cell[0].quad_xi.shape[0])

        global_overlap_values = tf.zeros(
            (total_number_of_cells, number_of_quad_points), dtype=datahandler[0].dtype
        )
        global_overlap_gradients_x = tf.zeros(
            (total_number_of_cells, number_of_quad_points), dtype=datahandler[0].dtype
        )
        global_overlap_gradients_y = tf.zeros(
            (total_number_of_cells, number_of_quad_points), dtype=datahandler[0].dtype
        )

        # print("Global overlap values (before): ", global_overlap_values)

        # loop over subdomains
        for block_id in range(len(self.subdomains_in_domain)):

            cells_in_current_subdomain = self.subdomains_in_domain[block_id]
            # print("Subdomain ID: ", block_id)
            # print("Cells in current subdomain: ", cells_in_current_subdomain)
            cells_in_current_subdomain = tf.reshape(
                tf.constant(cells_in_current_subdomain, dtype=tf.int32), [-1, 1]
            )

            # get the solution values and gradients for the current subdomain
            subdomain_solution_values = datahandler[block_id].subdomain_solution_values
            subdomain_solution_gradients_x = datahandler[block_id].subdomain_solution_gradient_x
            subdomain_solution_gradients_y = datahandler[block_id].subdomain_solution_gradient_y

            # print("Domain number: ", block_id)
            # print("Subdomain solution values: ", subdomain_solution_values)

            global_overlap_values = tf.tensor_scatter_nd_add(
                global_overlap_values,
                indices=cells_in_current_subdomain,
                updates=subdomain_solution_values,
            )
            global_overlap_gradients_x = tf.tensor_scatter_nd_add(
                global_overlap_gradients_x,
                indices=cells_in_current_subdomain,
                updates=subdomain_solution_gradients_x,
            )
            global_overlap_gradients_y = tf.tensor_scatter_nd_add(
                global_overlap_gradients_y,
                indices=cells_in_current_subdomain,
                updates=subdomain_solution_gradients_y,
            )

        

        for block_id in range(len(self.subdomains_in_domain)):
            cells_in_current_subdomain = self.subdomains_in_domain[block_id]
            final_overlap_values = tf.gather(global_overlap_values, cells_in_current_subdomain)
            final_overlap_gradients_x = tf.gather(
                global_overlap_gradients_x, cells_in_current_subdomain
            )
            final_overlap_gradients_y = tf.gather(
                global_overlap_gradients_y, cells_in_current_subdomain
            )

            datahandler[block_id].reset_overlap_values()
            overlap_solution_values = tf.subtract(
                final_overlap_values, datahandler[block_id].subdomain_solution_values
            )
            overlap_solution_gradients_x = tf.subtract(
                final_overlap_gradients_x, datahandler[block_id].subdomain_solution_gradient_x
            )
            overlap_solution_gradients_y = tf.subtract(
                final_overlap_gradients_y, datahandler[block_id].subdomain_solution_gradient_y
            )
            datahandler[block_id].update_overlap_solution(
                overlap_solution_values, overlap_solution_gradients_x, overlap_solution_gradients_y
            )

        return
