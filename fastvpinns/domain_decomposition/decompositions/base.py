import tensorflow as tf
from ...FE.fespace_domain_decomposition import *
from ...Geometry.geometry_domain_decomposition import *
import numpy as np
from ...utils.print_utils import print_table
import matplotlib.pyplot as plt
from ...utils.plot_utils import *
from pathlib import Path
from multiprocessing import Pool


class DomainDecomposition:
    """
    This class will be responsible for:
    1. Decomposing the domain into blocks
    2. Identifying the halo cells
    3. Assigning the blocks to the workers.
    """

    def __init__(self, domain):

        self.domain = domain

        self.subdomains_in_domain = self.domain.subdomains

        self.blocks_shared_by_cells = (
            self.id_blocks_shared_by_cells()
        )  # This is a dictionary with cell_id as key and list of blocks as value. A block is a collection of cells.

        self.halo_cells_in_blocks = (
            self.id_halo_cells()
        )  # This is a dictionary with block_id as key and list of halo cells as value. A halo cell is a cell that is shared by more than one block.

        # self.plot_domains()

        # print the details of the domain decomposition
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
            self.domain.kernel_size_row,
            self.domain.kernel_size_col,
            self.domain.stride_row,
            self.domain.stride_col,
        ]
        print_table("Domain Decomposition", ["Item", "Values"], rows, values)

        # Print the blocks in the entire domain, this is a dict  of cells within each block
        rows = np.arange(0, len(self.subdomains_in_domain))
        values = self.subdomains_in_domain
        print_table("Blocks in entire domain", ["Block ID", "Cells"], rows, values)

        # print blocks shared by cells
        rows = self.blocks_shared_by_cells.keys()
        values = self.blocks_shared_by_cells.values()
        print_table("Blocks shared by cells", ["Cell ID", "Blocks"], rows, values)

        # print halo cells in blocks
        rows = self.halo_cells_in_blocks.keys()
        values = self.halo_cells_in_blocks.values()
        print_table("Halo cells in blocks", ["Block ID", "Halo Cells"], rows, values)

    def id_blocks_shared_by_cells(self):
        """
        This function will identify the blocks that are shared by the cells.
        """
        blocks_shared_by_cells = {}
        for cell in self.domain.cells_in_domain:
            for block_id, block in enumerate(self.subdomains_in_domain):
                if cell in block:
                    if cell in blocks_shared_by_cells:
                        blocks_shared_by_cells[cell].append(block_id)
                    else:
                        blocks_shared_by_cells[cell] = [block_id]

        return blocks_shared_by_cells

    def id_halo_cells(self):
        """
        This function will identify the halo cells in all blocks.
        """
        halo_cells_in_blocks = {}
        for block_id, block in enumerate(self.subdomains_in_domain):
            halo_cells = []
            for cell in block:
                if len(self.blocks_shared_by_cells[cell]) > 1:
                    halo_cells.append(cell)
            halo_cells_in_blocks[block_id] = halo_cells

        return halo_cells_in_blocks

    def plot_domains(self):
        """
        This function will plot the domains.
        """
        # loop over all cells and obtain the x_min, x_max, y_min, y_max
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for cell in self.global_fespace.cells:
            x_min = min(x_min, np.min(cell[:, 0]))
            x_max = max(x_max, np.max(cell[:, 0]))
            y_min = min(y_min, np.min(cell[:, 1]))
            y_max = max(y_max, np.max(cell[:, 1]))

        # loop over all the blocks and plot the domains
        for block_id, block in enumerate(self.blocks_in_entire_domain):

            # plot the entire domain
            plt.figure(figsize=(6.4, 4.8))
            # Set axis properties
            plt.gca().set_xlim([x_min, x_max])
            plt.gca().set_ylim([y_min, y_max])
            plt.gca().set_xlabel(r"$x$")
            plt.gca().set_ylabel(r"$y$")

            # plot the cells
            for i, cell in enumerate(self.global_fespace.cells):
                # get the coordinates of the cell
                x = cell[:, 0]
                y = cell[:, 1]
                # add the first point to the end of the array
                x = np.append(x, x[0])
                y = np.append(y, y[0])

                plt.plot(x, y, 'k-', linewidth=0.9)
                # print the cell number at the centroid
                centroid = np.mean(cell, axis=0)
                plt.text(centroid[0], centroid[1], f"{i}", fontsize=20)

            # obtain the x_min, x_max, y_min, y_max of the block
            x_min_block = np.inf
            x_max_block = -np.inf
            y_min_block = np.inf
            y_max_block = -np.inf

            for cell_no in block:
                cell = self.global_fespace.cells[cell_no]
                x_min_block = min(x_min_block, np.min(cell[:, 0]))
                x_max_block = max(x_max_block, np.max(cell[:, 0]))
                y_min_block = min(y_min_block, np.min(cell[:, 1]))
                y_max_block = max(y_max_block, np.max(cell[:, 1]))

            # draw a rectangle around the block boundary based on the x_min, x_max, y_min, y_max
            plt.plot([x_min_block, x_max_block], [y_min_block, y_min_block], 'r-', linewidth=2)
            plt.plot([x_min_block, x_max_block], [y_max_block, y_max_block], 'r-', linewidth=2)
            plt.plot([x_min_block, x_min_block], [y_min_block, y_max_block], 'r-', linewidth=2)
            plt.plot([x_max_block, x_max_block], [y_min_block, y_max_block], 'r-', linewidth=2)

            # Color the cells in the block with a different color
            for cell_no in block:
                cell = self.global_fespace.cells[cell_no]
                x = cell[:, 0]
                y = cell[:, 1]
                # add the first point to the end of the array
                x = np.append(x, x[0])
                y = np.append(y, y[0])

                plt.fill(x, y, 'r', alpha=0.5)

            # print the block number at the centroid, calculate centroid of the block based on the x_min_block, x_max_block, y_min_block, y_max_block
            centroid = np.array([(x_min_block + x_max_block) / 2, (y_min_block + y_max_block) / 2])

            # print the text f"Block_id: {block_id}" at the centroid with a white background
            plt.text(
                centroid[0] - 0.1,
                centroid[1],
                f"Block_id: {block_id}",
                fontsize=20,
                bbox=dict(facecolor='white', alpha=0.9),
            )

            # Add a title
            plt.title(f"Block {block_id}")

            # create a foler called "iter_plots" in the output_path
            Path(self.output_path + "/iter_plots").mkdir(parents=True, exist_ok=True)

            output_file_name = Path(
                self.output_path + "/iter_plots/" + "block_" + str(block_id) + ".png"
            )

            # save the figure
            plt.savefig(output_file_name, dpi=300)

        print("[INFO] - Plotted all the domains in the folder iter directory")

        return

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
