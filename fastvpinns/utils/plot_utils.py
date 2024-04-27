"""
filename: plot_utils.py
description: This file contains the utility functions for
              plotting the loss functions and the predicted inverse parameters

author: Thivin Anandh D
date: 02/11/2023
changelog: 02/11/2023 - file created
           02/11/2023 - added functions to plot the loss functions and the predicted 
                        inverse parameters

known_issues: None
"""

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np


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
    """This function will plot the loss function.
    :param loss_function: list of loss values
    :type loss_function: list
    :param output_path: path to save the plot
    :type output_path: str
    :return: None
    :rtype: None
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
    """This function will plot the loss function.
    :param array: list of loss values
    :type array: list
    :param output_path: path to save the plot
    :type output_path: str
    :param filename: filename to save the plot
    :type filename: str
    :param title: title of the plot
    :type title: str
    :param x_label: x-axis label, defaults to "Epochs"
    :type x_label: str, optional
    :param y_label: y-axis label, defaults to "Loss"
    :type y_label: str, optional
    :return: None
    :rtype: None
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
    loss_function_list, output_path, filename, legend_labels, y_label, title, x_label="Epochs"
):
    """This function will plot the loss function in log scale for multiple parameters.
    :param loss_function_list: list of loss values for multiple parameters
    :type loss_function_list: list
    :param output_path: path to save the plot
    :type output_path: str
    :param filename: filename to save the plot
    :type filename: str
    :param legend_labels: list of legend labels
    :type legend_labels: list
    :param y_label: y-axis label
    :type y_label: str
    :param title: title of the plot
    :type title: str
    :param x_label: x-axis label, defaults to "Epochs"
    :type x_label: str, optional
    :return: None
    :rtype: None
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
    """This function will plot the test loss function of the inverse parameter.
    :param loss_function: list of loss values
    :type loss_function: list
    :param output_path: path to save the plot
    :type output_path: str
    :return: None
    :rtype: None
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
    """This function will plot the test loss function.
    :param loss_function: list of loss values
    :type loss_function: list
    :param output_path: path to save the plot
    :type output_path: str
    :param fileprefix: prefix for the filename, defaults to ""
    :type fileprefix: str, optional
    :return: None
    :rtype: None
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
    """This function will plot the test loss as a function of time in seconds.
    :param time_array: array of time values
    :type time_array: numpy.ndarray
    :param loss_function: list of loss values
    :type loss_function: list
    :param output_path: path to save the plot
    :type output_path: str
    :return: None
    :rtype: None
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
    """This function will plot the contour plot.
    :param x: x values
    :type x: numpy.ndarray
    :param y: y values
    :type y: numpy.ndarray
    :param z: z values
    :type z: numpy.ndarray
    :param output_path: path to save the plot
    :type output_path: str
    :param filename: filename to save the plot
    :type filename: str
    :param title: title of the plot
    :type title: str
    :return: None
    :rtype: None
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
    """This function will plot the predicted inverse parameter.
    :param inverse_predicted: list of predicted inverse parameter values
    :type inverse_predicted: list
    :param inverse_param_name: name of the inverse parameter
    :type inverse_param_name: str
    :param actual_value: actual value of the inverse parameter
    :type actual_value: float
    :param output_path: path to save the plot
    :type output_path: str
    :param file_prefix: prefix for the filename
    :type file_prefix: str
    :return: None
    :rtype: None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    plt.plot(inverse_predicted, label="Predicted " + inverse_param_name)

    # draw a horizontal dotted line at the actual value
    plt.hlines(
        actual_value,
        0,
        len(inverse_predicted),
        colors="k",
        linestyles="dashed",
        label="Actual " + inverse_param_name,
    )

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
