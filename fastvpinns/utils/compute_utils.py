"""
filename: compute_utils.py
description: This file contains the utility functions for 
             computing the errors between the exact and 
             predicted solutions
author: Thivin Anandh D
date: 02/11/2023
changelog: 02/11/2023 - file created
           02/11/2023 - added functions to compute L1, L2, L_inf errors

known_issues: None
"""

# Importing the required libraries
import numpy as np


def compute_l2_error(u_exact, u_approx):
    """This function will compute the L2 error between the exact solution and the approximate solution.
    The L2 error is defined as:

    ..math::
        \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (u_{exact} - u_{approx})^2}

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: L2 error between the exact and approximate solutions
    :rtype: float
    """
    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = np.sqrt(np.mean(np.square(u_exact - u_approx)))
    return l2_error


def compute_l1_error(u_exact, u_approx):
    """This function will compute the L1 error between the exact solution and the approximate solution.
    The L1 error is defined as:
    ..math::
        \\frac{1}{N} \\sum_{i=1}^{N} |u_{exact} - u_{approx}|
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: L1 error between the exact and approximate solutions
    :rtype: float
    """
    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()
    # compute the L1 error
    l1_error = np.mean(np.abs(u_exact - u_approx))
    return l1_error


def compute_linf_error(u_exact, u_approx):
    """This function will compute the L_inf error between the exact solution and the approximate solution.
    The L_inf error is defined as
        ..math::
            \\max_{i=1}^{N} |u_{exact} - u_{approx}|
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: L_inf error between the exact and approximate solutions
    :rtype: float
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L_inf error
    linf_error = np.max(np.abs(u_exact - u_approx))
    return linf_error


def compute_l2_error_relative(u_exact, u_approx):
    """This function will compute the relative L2 error between the exact solution and the approximate solution.
    The relative L2 error is defined as:
        ..math::
            \\frac{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (u_{exact} - u_{approx})^2}}{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} u_{exact}^2}}
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: relative L2 error between the exact and approximate solutions
    :rtype: float
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
    """This function will compute the relative L_inf error between the exact solution and the approximate solution.
    The relative L_inf error is defined as:
        ..math::
            \\frac{\\max_{i=1}^{N} |u_{exact} - u_{approx}|}{\\max_{i=1}^{N} |u_{exact}|}
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: relative L_inf error between the exact and approximate solutions
    :rtype: float
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
    """This function will compute the relative L1 error between the exact solution and the approximate solution.
    The relative L1 error is defined as:
        ..math::
            \\frac{\\frac{1}{N} \\sum_{i=1}^{N} |u_{exact} - u_{approx}|}{\\frac{1}{N} \\sum_{i=1}^{N} |u_{exact}|}
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: relative L1 error between the exact and approximate solutions
    :rtype: float
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l1_error = compute_l1_error(u_exact, u_approx)
    # compute the relative l1 error
    l1_error_relative = l1_error / np.mean(np.abs(u_exact))
    return l1_error_relative


def compute_errors_combined(u_exact, u_approx):
    """This function will compute the L1, L2 and L_inf absolute and relative errors.
    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray
    :return: L1, L2 and L_inf absolute and relative errors
    :rtype: tuple

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

    return (
        l2_error,
        linf_error,
        l2_error_relative,
        linf_error_relative,
        l1_error,
        l1_error_relative,
    )
