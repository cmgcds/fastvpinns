# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# Actual Solution : sin(X) * cos(Y) * exp(-1.0 * eps * (X**2 + Y**2))
import numpy as np
import tensorflow as tf


EPS = 0.3

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10
    return val

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10
    return val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10
    return val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10
    return val

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    X =x
    Y =y
    eps = EPS

    return -EPS * (
        40.0 * X * eps * (np.tanh(X)**2 - 1) * np.sin(X) 
        - 40.0 * X * eps * np.cos(X) * np.tanh(X) 
        + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X) 
        + 20 * (np.tanh(X)**2 - 1) * np.sin(X) * np.tanh(X) 
        - 20 * (np.tanh(X)**2 - 1) * np.cos(X) 
        - 10 * np.sin(X) * np.tanh(X)
    ) * np.exp(-1.0 * X**2 * eps)


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10

    return val

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
    # Initial Guess
    eps = EPS

    return {"eps": eps}


def get_inverse_params_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = 2

    return {"eps": eps}


def get_inverse_params_actual_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}