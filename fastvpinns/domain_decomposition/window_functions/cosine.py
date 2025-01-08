from .base import WindowFunction
import tensorflow as tf
import numpy as np

pi = tf.constant(np.pi, dtype=tf.float32)


class CosineWindowFunction(WindowFunction):
    """
    Cosine window function.
    """

    def __init__(self, subdomain_mean_list, subdomain_span_list, scaling_factor, overlap_factor):
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
        super().__init__(subdomain_mean_list, subdomain_span_list, 'cosine')

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
            0
            if x <= x1
            else (
                0.5 * (1 - tf.cos(pi * (x - x1) / (x2 - x1)))
                if x1 < x < x2
                else (
                    1
                    if x2 <= x <= x3
                    else 0.5 * (1 + tf.cos(pi * (x - x3) / (x4 - x3))) if x3 < x < x4 else 0
                )
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

        x1 = self.x_min[block_id]
        x2 = self.x_non_overlap_min[block_id]
        x3 = self.x_non_overlap_max[block_id]
        x4 = self.x_max[block_id]

        y1 = self.y_min[block_id]
        y2 = self.y_non_overlap_min[block_id]
        y3 = self.y_non_overlap_max[block_id]
        y4 = self.y_max[block_id]

        return self.get_kernel(x1, x2, x3, x4)(x) * self.get_kernel(y1, y2, y3, y4)(y)

    def get_partition_of_unity(self):
        return super().get_partition_of_unity()

    def check_partition_of_unity(self):
        return super().check_partition_of_unity()

    def get_non_overlap(self):
        """
        Get the non-overlapping part of the subdomains.
        """

        self.x_non_overlap_min = [
            mean - span / 2 * (1 - self.overlap_factor)
            for mean, span in zip(self.x_mean_list, self.x_span_list)
        ]
        self.x_non_overlap_max = [
            mean + span / 2 * (1 - self.overlap_factor)
            for mean, span in zip(self.x_mean_list, self.x_span_list)
        ]
        self.y_non_overlap_min = [
            mean - span / 2 * (1 - self.overlap_factor)
            for mean, span in zip(self.y_mean_list, self.y_span_list)
        ]
        self.y_non_overlap_max = [
            mean + span / 2 * (1 - self.overlap_factor)
            for mean, span in zip(self.y_mean_list, self.y_span_list)
        ]
