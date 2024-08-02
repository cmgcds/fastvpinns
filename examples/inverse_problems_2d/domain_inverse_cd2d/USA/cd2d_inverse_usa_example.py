# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# Actual Solution : sin(X) * cos(Y) * exp(-1.0 * eps * (X**2 + Y**2))
import numpy as np
import tensorflow as tf


def usa_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    # val = 20 * np.exp(-0.1 * y) * np.cos(x)
    val = np.exp(-0.1 * y) * np.cos(x)

    return val


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    # return -y * np.sin(x * y) * np.tanh(8 * x * y) + 8 * y * np.cos(x * y) / np.cosh(8 * x * y)**2 + 10 * (x**2 + y**2) * \
    #       (16 * np.sin(x * y) / np.cosh(8 * x * y)**2 + np.cos(x * y) * np.tanh(8 * x * y) + 128 * np.cos(x * y) * np.tanh(8 * x * y) / np.cosh(8 * x * y)**2) * np.sin(x) * np.cos(y)

    # return (196.0 - 396.0 * np.sin(x) ** 2) * np.exp(-0.2 * y)

    return (
        (
            -0.1 * x * np.sin(x * y) * np.cos(x)
            - y * np.sin(x) * np.sin(x * y)
            + 1.99 * np.cos(x) * np.cos(x * y)
        )
        * np.exp(-0.1 * y)
        * np.sin(x)
    )


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    val = np.exp(-0.1 * y) * np.cos(x)

    return val


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: usa_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet"}


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """

    eps = 20  # will not be used in the loss function, as it will be replaced by the predicted value of NN
    b1 = 0.0
    b2 = 0
    c = 0.0

    return {"eps": eps, "b_x": b1, "b_y": b2, "c": c}


def get_inverse_params_actual_dict(x, y):
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = np.sin(x) * np.cos(x * y)
    return {"eps": eps}
