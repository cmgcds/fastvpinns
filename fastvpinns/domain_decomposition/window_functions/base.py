import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC, abstractmethod


class WindowFunction():
    """
    Base class for window functions.
    """

    def __init__(self, subdomain_mean_list, subdomain_span_list, kernel_type):
        """
        Initialize the window function.

        Parameters
        ----------
        subdomain_mean_list : list
            List of the means of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        subdomain_span_list : list
            List of the spans of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        """
        self.subdomain_mean_list = subdomain_mean_list
        self.subdomain_span_list = subdomain_span_list

        self.x_mean_list = subdomain_mean_list[0]
        self.y_mean_list = subdomain_mean_list[1]

        self.x_span_list = subdomain_span_list[0]
        self.y_span_list = subdomain_span_list[1]

        self.kernel_type = kernel_type

        if self.kernel_type not in ['cosine']:
            raise ValueError('Kernel type not recognized.')
        
    def get_min_max(self):
        """
        Get the minimum and maximum values of the subdomains.
        """
        
        self.x_min = [mean - span/2 for mean, span in zip(self.x_mean_list, self.x_span_list)]
        self.x_max = [mean + span/2 for mean, span in zip(self.x_mean_list, self.x_span_list)]
        self.y_min = [mean - span/2 for mean, span in zip(self.y_mean_list, self.y_span_list)]
        self.y_max = [mean + span/2 for mean, span in zip(self.y_mean_list, self.y_span_list)]

    

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




