# Main File for running the Python code
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  02/Nov/2023


import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
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


# plot the loss function
def plot_loss_function(loss_function, output_path):
    """
    This function will plot the loss function
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_path + "/loss_function.png", dpi=300)

    plt.close()


def plot_array(array, output_path, filename, title, x_label="Epochs", y_label="Loss"):
    """
    This function will plot the loss function
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(array)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_path + f"/{filename}.png", dpi=300)
    plt.close()


# general utility to plot multiple parameters
def plot_multiple_loss_function(
    loss_function_list,
    output_path,
    filename,
    legend_labels,
    y_label,
    title,
    x_label="Epochs",
):
    """
    This function will plot the loss function in log scale for multiple parameters

    -loss_function_list: list of loss functions
    - output_path: output path to save the plot
    - legend_labels: list of legend labels
    - y_label: y axis label
    - x_label: x axis label
    """

    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    for loss_function, label in zip(loss_function_list, legend_labels):
        plt.plot(loss_function, label=label)

    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(output_path + f"/{filename}.png", dpi=300)
    plt.close()


# plot the loss function
def plot_inverse_test_loss_function(loss_function, output_path):
    """
    This function will plot the test loss function of the inverse parameter
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.savefig(output_path + "/test_inverse_loss_function.png", dpi=300)
    plt.close()


def plot_test_loss_function(loss_function, output_path, fileprefix=""):
    """
    This function will plot the loss function
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    if fileprefix == "":
        plt.savefig(output_path + "/test_loss_function.png", dpi=300)
    else:
        plt.savefig(output_path + "/" + fileprefix + "_test_loss_function.png", dpi=300)
    plt.close()


def plot_test_time_loss_function(time_array, loss_function, output_path):
    """
    This function will plot the loss function
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(time_array, loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Time [s]")
    plt.ylabel("MAE Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.savefig(output_path + "/test_time_loss_function.png", dpi=300)
    plt.close()


def plot_contour(x, y, z, output_path, filename, title):
    """
    This function will plot the contour plot
    """

    plt.figure(figsize=(6.4, 4.8))
    plt.contourf(x, y, z, levels=100, cmap="jet")
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path + "/" + filename + ".png", dpi=300)

    plt.close()


# plot the Inverse parameter prediction
def plot_inverse_param_function(
    inverse_predicted, inverse_param_name, actual_value, output_path, file_prefix
):
    """
    This function will plot the loss function
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    plt.plot(inverse_predicted, label=f"Predicted " + inverse_param_name)

    # draw a horizontal dotted line at the actual value
    plt.hlines(
        actual_value,
        0,
        len(inverse_predicted),
        colors="k",
        linestyles="dashed",
        label=f"Actual " + inverse_param_name,
    )

    # generate a box and print the absolute difference between the actual and predicted value
    # generate the text on a empty region of the plot with fontsize 20
    # plt.text(0.85, 0.25, f'Absolute Error = {abs(actual_value - inverse_predicted[-1])}', fontsize=20,
    #      horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)

    # plot y axis in log scale
    # plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel(inverse_param_name)

    # plt.title("Loss Function")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(output_path + f"/{file_prefix}.png", dpi=300)
    plt.close()

    # plot the loss of inverse parameter
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    actual_val_array = np.ones_like(inverse_predicted) * actual_value
    plt.plot(abs(actual_val_array - inverse_predicted))
    plt.xlabel("Epochs")
    plt.ylabel("Absolute Error")
    plt.yscale("log")
    plt.title("Absolute Error of " + inverse_param_name)
    plt.tight_layout()
    plt.savefig(output_path + f"/{file_prefix}_absolute_error.png", dpi=300)
    plt.close()
