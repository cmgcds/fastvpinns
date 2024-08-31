# Example file for the poisson problem
# Path: examples/nse2d.py
# file contains the exact solution, rhs and boundary conditions for the Burgers equation.
import numpy as np
import tensorflow as tf


def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """

    u = np.sin(x**2 + y**2)
    v = np.cos(x**2) * np.tanh(8 * y**2)

    return [u, v]


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = np.sin(x**2 + y**2)
    v = np.cos(x**2) * np.tanh(8 * y**2)

    return [u, v]


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = np.sin(x**2 + y**2)
    v = np.cos(x**2) * np.tanh(8 * y**2)

    return [u, v]


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = np.sin(x**2 + y**2)
    v = np.cos(x**2) * np.tanh(8 * y**2)

    return [u, v]


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp = 1

    u = (
        4 * x**2 * np.sin(x**2 + y**2)
        + x * np.sin(2 * x**2 + 2 * y**2)
        + 4 * y**2 * np.sin(x**2 + y**2)
        + 2 * y * np.cos(x**2) * np.cos(x**2 + y**2) * np.tanh(8 * y**2)
        - 4 * np.cos(x**2 + y**2)
    )
    v = (
        4 * x**2 * np.cos(x**2) * np.tanh(8 * y**2)
        - 2 * x * np.sin(x**2) * np.sin(x**2 + y**2) * np.tanh(8 * y**2)
        + 512 * y**2 * np.cos(x**2) * np.tanh(8 * y**2) / np.cosh(8 * y**2) ** 2
        + 16 * y * np.cos(x**2) ** 2 * np.tanh(8 * y**2) / np.cosh(8 * y**2) ** 2
        + 2 * np.sin(x**2) * np.tanh(8 * y**2)
        - 16 * np.cos(x**2) / np.cosh(8 * y**2) ** 2
    )

    return [u, v]


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    # val = 16 * x * (1 - x) * y * (1 - y)
    u = np.sin(x**2 + y**2)
    v = np.cos(x**2) * np.tanh(8 * y**2)

    return [u, v]


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    If no boundary condition needs to be specified, provide "none" for the component
    Accepted Values : "dirichlet", "neumann"
    Currently, it works on Homogenous Boundary Conditions only - Either Dirichlet on both components and Neumann on both components
    """
    return {
        1000: ["dirichlet", "dirichlet"],
        1001: ["dirichlet", "dirichlet"],
        1002: ["dirichlet", "dirichlet"],
        1003: ["dirichlet", "dirichlet"],
    }


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    re_nr = 1.0

    return {"re_nr": re_nr}
