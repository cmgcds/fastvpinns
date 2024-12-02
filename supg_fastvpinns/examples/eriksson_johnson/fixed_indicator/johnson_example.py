# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

GLOBAL_EPS_VALUE = 0.1

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)
    r1 = (1+np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    r2 = (1-np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    u = ((np.exp(r1*(x-1))-np.exp(r2*(x-1)))/(np.exp(-r1)-np.exp(-r2)))*np.sin(np.pi*y)

    return u

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)
    r1 = (1+np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    r2 = (1-np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    u = ((np.exp(r1*(x-1))-np.exp(r2*(x-1)))/(np.exp(-r1)-np.exp(-r2)))*np.sin(np.pi*y)

    return u

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)
    r1 = (1+np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    r2 = (1-np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    u = ((np.exp(r1*(x-1))-np.exp(r2*(x-1)))/(np.exp(-r1)-np.exp(-r2)))*np.sin(np.pi*y)

    return u

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)
    r1 = (1+np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    r2 = (1-np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    u = ((np.exp(r1*(x-1))-np.exp(r2*(x-1)))/(np.exp(-r1)-np.exp(-r2)))*np.sin(np.pi*y)

    return u

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # eps = GLOBAL_EPS_VALUE
    # return (1/2)*(0.2*np.pi**2*eps**2*(1 - np.exp((x - 1)*np.sqrt(4*np.pi**2*eps**2 + 1)/eps)) - eps*(np.sqrt(4*np.pi**2*eps**2 + 1) + (np.sqrt(4*np.pi**2*eps**2 + 1) + 1)*np.exp((x - 1)*np.sqrt(4*np.pi**2*eps**2 + 1)/eps) - 1) - 0.05*(np.sqrt(4*np.pi**2*eps**2 + 1) - 1)**2 + 0.05*(np.sqrt(4*np.pi**2*eps**2 + 1) + 1)**2*np.exp((x - 1)*np.sqrt(4*np.pi**2*eps**2 + 1)/eps))*np.exp(-1/2*(x - 1)*(np.sqrt(4*np.pi**2*eps**2 + 1) - 1)/eps)*np.sin(np.pi*y)/(eps**2*(-np.exp((1/2)*(-np.sqrt(4*np.pi**2*eps**2 + 1) - 1)/eps) + np.exp((1/2)*(np.sqrt(4*np.pi**2*eps**2 + 1) - 1)/eps)))

    return x*0

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    eps = GLOBAL_EPS_VALUE
    # val = 16 * x * (1 - x) * y * (1 - y)
    r1 = (1+np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    r2 = (1-np.sqrt(1+4*(eps**2)*(np.pi**2)))/(2*eps)
    u = ((np.exp(r1*(x-1))-np.exp(r2*(x-1)))/(np.exp(-r1)-np.exp(-r2)))*np.sin(np.pi*y)
    # u = x*0

    return u

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