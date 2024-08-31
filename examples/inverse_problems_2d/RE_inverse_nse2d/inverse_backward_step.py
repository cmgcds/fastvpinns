# Example file for the poisson problem
# Path: examples/nse2d.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

ACTUAL_RE_NR = 200.0
INITIAL_GUESS = 120.0


def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    if y >= 0.0:
        u = 24 * y * (0.5 - y)
    else:
        u = 0.0
    v = 0.0
    p = 0.0
    return [np.ones_like(x) * u, np.ones_like(x) * v, np.ones_like(x) * p]


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = 0.0
    v = 0.0
    p = 0.0
    return [np.ones_like(x) * u, np.ones_like(x) * v, np.ones_like(x) * p]


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = 0.0
    v = 0.0
    p = 0.0

    return [u, np.ones_like(x) * v, np.ones_like(x) * p]


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    u = 0.0
    v = 0.0
    p = 0.0

    return [np.ones_like(x) * u, np.ones_like(x) * v, np.ones_like(x) * p]


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp = 1

    u = 0
    v = 0
    p = 0

    return [np.ones_like(x) * u, np.ones_like(x) * v, np.ones_like(x) * p]


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    # val = 16 * x * (1 - x) * y * (1 - y)
    u = y * (1 - y) * 4
    v = 0.0
    p = 0.0

    return [np.ones_like(x) * u, np.ones_like(x) * v, np.ones_like(x) * p]


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
        1000: ["dirichlet", "dirichlet", "none"],
        1001: ["neumann", "neumann", "none"],
        1002: ["dirichlet", "dirichlet", "none"],
        1003: ["dirichlet", "dirichlet", "none"],
    }


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    re_nr = ACTUAL_RE_NR

    return {"re_nr": re_nr}


def get_penalty_coefficients_dict():
    """
    This function will return a dictionary of penalty coefficients
    """
    return {"residual_u": 1e1, "residual_v": 1e1, "divergence": 1e1}


def get_inverse_params_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    re_nr = INITIAL_GUESS

    return {"re_nr": re_nr}


def get_inverse_params_actual_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    re_nr = ACTUAL_RE_NR

    return {"re_nr": re_nr}
