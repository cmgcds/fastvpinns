from .base import WindowFunction
import tensorflow as tf
import numpy as np
from ...utils.plot_utils import plot_contour

pi = tf.constant(np.pi, dtype=tf.float32)


class CosineWindowFunction(WindowFunction):
    """
    Cosine window function.
    """

    def __init__(self, decomposed_domain):
        """
        Initialize the cosine window function.

        Parameters
        ----------
        subdomain_mean_list : list
            List of the means of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        subdomain_span_list : list
            List of the spans of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        scaling_factor : float
            Scaling factor for the subdomains.
        overlap_factor : float
            Overlap factor for the subdomains.
        """
        super().__init__()
        self.subdomain_boundary_limits = decomposed_domain.subdomain_boundary_limits
        self.subdomain_non_overlap_limits = decomposed_domain.non_overlapping_extents

        self.plot_window_function()

    def get_kernel(self, x1, x2, x3, x4):
        """
        Evaluate the cosine kernel function at given coordinates.

        Parameters
        ----------
        x1 : float
            Coordinate of the lower bound of subdomain.
        x2 : float
            Coordinate of the upper bound of subdomain.
        x3 : float
            Coordinate of the lower bound of non-overlapping part of subdomain.
        x4 : float
            Coordinate of the upper bound of non-overlapping part of subdomain.


        Returns
        -------
        kernel : function
            Cosine kernel function

        """

        clamp = lambda f: tf.clip_by_value(f, 0.0, 1.0)

        kernel = lambda x: clamp(
            tf.where(
                x <= x1,
                0.0,
                tf.where(
                    x1 < x,
                    tf.where(
                        x < x2,
                        0.5 * (1 - tf.cos(pi * (x - x1) / (x2 - x1))),
                        tf.where(
                            x2 <= x,
                            tf.where(
                                x <= x3,
                                1.0,
                                tf.where(
                                    x3 < x,
                                    tf.where(
                                        x < x4, 0.5 * (1 + tf.cos(pi * (x - x3) / (x4 - x3))), 0.0
                                    ),
                                    0.0,
                                ),
                            ),
                            0.0,
                        ),
                    ),
                    0.0,
                ),
            )
        )

        return kernel

    def evaluate_window_function(self, x, y, block_id):
        """
        Evaluate the cosine window function at given coordinates.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        block_id : int
            Block id.

        Returns
        -------
        float
            Value of the window function at (x, y).
        """

        x1 = self.subdomain_boundary_limits[block_id][0]
        x2 = self.subdomain_non_overlap_limits[block_id][0]
        x3 = self.subdomain_non_overlap_limits[block_id][1]
        x4 = self.subdomain_boundary_limits[block_id][1]

        y1 = self.subdomain_boundary_limits[block_id][2]
        y2 = self.subdomain_non_overlap_limits[block_id][2]
        y3 = self.subdomain_non_overlap_limits[block_id][3]
        y4 = self.subdomain_boundary_limits[block_id][3]

        return self.get_kernel(x1, x2, x3, x4)(x) * self.get_kernel(y1, y2, y3, y4)(y)

    def get_partition_of_unity(self):
        return super().get_partition_of_unity()

    def check_partition_of_unity(self):
        return super().check_partition_of_unity()

    def plot_window_function(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        for block_id in range(len(self.subdomain_boundary_limits)):
            z = self.evaluate_window_function(X, Y, block_id)
            plot_contour(x=X, y=Y, z=z, title=f"Window Function subdomain {block_id}", output_path=".", filename=f"window_function_{block_id}")