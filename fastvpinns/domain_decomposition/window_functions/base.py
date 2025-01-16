import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC, abstractmethod


class WindowFunction:
    """
    Base class for window functions.
    """

    def __init__(self):
        """
        Initialize the window function.

        Parameters
        ----------
        subdomain_mean_list : list
            List of the means of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        subdomain_span_list : list
            List of the spans of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        """

    @abstractmethod
    def get_kernel(self):
        """
        Evaluate the kernel function at given coordinates.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.

        Returns
        -------
        float
            Value of the kernel function at (x, y).
        """
        pass

    @abstractmethod
    def evaluate_window_function(self, x, y, block_id):
        """
        Evaluate the window function at given coordinates.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        block_id : int
            Subdomain ID.

        Returns
        -------
        float
            Value of the window function at (x, y).
        """
        pass

    @abstractmethod
    def get_partition_of_unity(self):
        """
        Get the partition of unity for the window function.
        """
        pass

    @abstractmethod
    def plot_window_function(self):
        """
        Plot the window function.
        """
        pass

    @abstractmethod
    def check_partition_of_unity(self):
        """
        Check if the partition of unity is satisfied.
        """
        pass
