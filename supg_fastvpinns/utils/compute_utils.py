
# File for computing utilities
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  02/Nov/2023

import numpy as np

def compute_l2_error(u_exact, u_approx):
    """
    This function will compute the L2 error between the exact solution and the approximate solution
    """
    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = np.sqrt(np.mean(np.square(u_exact - u_approx)))
    return l2_error

def compute_l1_error(u_exact, u_approx):
    """
    This function will compute the L1 error between the exact solution and the approximate solution
    """
    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()
    # compute the L1 error
    l1_error = np.mean(np.abs(u_exact - u_approx))
    return l1_error

def compute_linf_error(u_exact, u_approx):
    """
    This function will compute the L_inf error between the exact solution and the approximate solution
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L_inf error
    linf_error = np.max(np.abs(u_exact - u_approx))
    return linf_error

def compute_l2_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L2 error between the exact solution and the approximate solution
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = compute_l2_error(u_exact, u_approx)
    # compute the relative L2 error
    l2_error_relative = l2_error / np.sqrt(np.mean(np.square(u_exact)))
    return l2_error_relative

def compute_linf_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L_inf error between the exact solution and the approximate solution
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L_inf error
    linf_error = compute_linf_error(u_exact, u_approx)
    # compute the relative L_inf error
    linf_error_relative = linf_error / np.max(np.abs(u_exact))
    return linf_error_relative

def compute_l1_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L1 error between the exact solution and the approximate solution
    """
    #flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l1_error = compute_l1_error(u_exact, u_approx)
    # compute the relative l1 error
    l1_error_relative = l1_error / np.mean(np.abs(u_exact))
    return l1_error_relative

def compute_errors_combined(u_exact, u_approx):
    """
    This function will compute the L2 and L_inf absolute and relative errors
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()
    
    # compute the L2 error
    l2_error = compute_l2_error(u_exact, u_approx)
    # compute the L_inf error
    linf_error = compute_linf_error(u_exact, u_approx)
    # compute the relative L2 error
    l2_error_relative = compute_l2_error_relative(u_exact, u_approx)
    # compute the relative L_inf error
    linf_error_relative = compute_linf_error_relative(u_exact, u_approx)

    
    # compute L1 Error 
    l1_error = compute_l1_error(u_exact, u_approx)

    # compute the relative L1 error
    l1_error_relative = compute_l1_error_relative(u_exact, u_approx)

    return l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative