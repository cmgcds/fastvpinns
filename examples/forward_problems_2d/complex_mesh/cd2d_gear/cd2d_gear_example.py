# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# for gear geometrt
import numpy as np
import tensorflow as tf


def inner_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0


def outer_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """

    return 0.0


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp =
    return 50 * np.sin(x) + np.cos(x)


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    r = np.sqrt(x**2 + y**2)

    return np.ones_like(x) * 0


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: outer_boundary, 1001: inner_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet"}


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1
    b_x = 0.1
    b_y = 0.0
    c = 0.0

    return {"eps": eps, "b_x": b_x, "b_y": b_y, "c": c}


def get_inverse_params_actual_dict(x, y):
    """
    This function will return a dictionary of inverse parameters
    """
    eps = np.cos(x) + np.cos(y)

    return {"eps": eps}
