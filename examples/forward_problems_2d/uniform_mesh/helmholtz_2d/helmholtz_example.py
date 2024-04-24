# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf


def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp = 1

    term1 = 2 * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)
    term2 = 2 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    term3 = (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)
    term4 = -2 * (np.pi**2) * (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)

    result = term1 + term2 + term3 + term4
    return result


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    k = 1.0
    eps = 1.0

    return {"k": k, "eps": eps}
