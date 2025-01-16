import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_subdomains(global_cell_list, subdomains, output_path):
    """
    This function will plot the domains.
    """
    # loop over all cells and obtain the x_min, x_max, y_min, y_max
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for cell in global_cell_list:
        x_min = min(x_min, np.min(cell[:, 0]))
        x_max = max(x_max, np.max(cell[:, 0]))
        y_min = min(y_min, np.min(cell[:, 1]))
        y_max = max(y_max, np.max(cell[:, 1]))

    # loop over all the blocks and plot the domains
    for block_id, block in enumerate(subdomains):

        # plot the entire domain
        plt.figure(figsize=(6.4, 4.8))
        # Set axis properties
        plt.gca().set_xlim([x_min, x_max])
        plt.gca().set_ylim([y_min, y_max])
        plt.gca().set_xlabel(r"$x$")
        plt.gca().set_ylabel(r"$y$")

        # plot the cells
        for i, cell in enumerate(global_cell_list):
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
            cell = global_cell_list.cells[cell_no]
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
            cell = global_cell_list.cells[cell_no]
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
        Path(output_path + "/iter_plots").mkdir(parents=True, exist_ok=True)

        output_file_name = Path(output_path + "/iter_plots/" + "block_" + str(block_id) + ".png")

        # save the figure
        plt.savefig(output_file_name, dpi=300)

    print("[INFO] - Plotted all the domains in the folder iter directory")

    return
