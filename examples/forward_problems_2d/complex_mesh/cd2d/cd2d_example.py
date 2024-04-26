# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf


def circle_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return np.cos(x**2 + y**2)


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    f_temp = (
        4.0 * x**2 * np.cos(x**2 + y**2)
        - 2.0 * x * np.sin(x**2 + y**2)
        + 4.0 * y**2 * np.cos(x**2 + y**2)
        + 4.0 * np.sin(x**2 + y**2)
    )

    return f_temp


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    val = np.cos(x**2 + y**2)
    return val


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: circle_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet"}


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1.0
    b_x = 1.0
    b_y = 0.0
    c = 0.0

    return {"eps": eps, "b_x": b_x, "b_y": b_y, "c": c}
