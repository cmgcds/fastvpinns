# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

GLOBAL_EPS_VALUE = 1e-8

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)

    # return x * ( y**2) - y**2 * np.exp((2*(x-1))/eps) - x * np.exp((3*(y-1))/eps) + np.exp((2*(x-1) + 3*(y-1))/eps)
    return np.zeros_like(x)

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)

    # return x * ( y**2) - y**2 * np.exp((2*(x-1))/eps) - x * np.exp((3*(y-1))/eps) + np.exp((2*(x-1) + 3*(y-1))/eps)
    return np.zeros_like(x)


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)

    # return x * ( y**2) - y**2 * np.exp((2*(x-1))/eps) - x * np.exp((3*(y-1))/eps) + np.exp((2*(x-1) + 3*(y-1))/eps)
    return np.zeros_like(x)


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)

    # return x * ( y**2) - y**2 * np.exp((2*(x-1))/eps) - x * np.exp((3*(y-1))/eps) + np.exp((2*(x-1) + 3*(y-1))/eps)
    return np.zeros_like(x)


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    eps = GLOBAL_EPS_VALUE
    # return 2*eps*(-x + np.exp(2*(x - 1)/eps)) + x*y**2 + 6*x*y - x*np.exp(3*(y - 1)/eps) - y**2*np.exp(2*(x - 1)/eps) + 2*y**2 - 6*y*np.exp(2*(x - 1)/eps) - 2*np.exp(3*(y - 1)/eps) + np.exp((2*x + 3*y - 5)/eps)
    return np.ones_like(x)

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    eps = GLOBAL_EPS_VALUE
    # exact_sol = np.where((x == 0) | (y == 0) | (x==1) | (y==1), 0, x)
    # a = 1
    # b = 1/eps
    # L = 0
    # R = 0
    # exact_sol = (a*x) + (((R-L-a)*((np.exp(-b*(1-x))) - np.exp(-b)))/(1-np.exp(-b))) + L
    a = 1
    b = 1/eps
    L = 0
    R = 0
    b1 = (a*x) + (((R-L-a)*((np.exp(-b*(1-x))) - np.exp(-b)))/(1-np.exp(-b))) + L
    exact_sol = np.where((x == 0) | (y == 0) | (x==1) | (y==1), 0, b1)
    return exact_sol

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
    eps = GLOBAL_EPS_VALUE
    b_x = 1.0
    b_y = 0.0
    c = 0.0
    return {"eps": eps, "b_x": b_x, "b_y": b_y, "c": c}