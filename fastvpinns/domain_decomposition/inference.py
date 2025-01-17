import tensorflow as tf
import numpy as np
from ..utils.plot_utils import plot_contour


class InferenceDomainDecomposition:
    """
    Class for inference in domain decomposition
    """

    def __init__(
        self,
        decomposed_domain,
        model,
        datahandler,
        window_function,
        test_points,
        exact_solution,
        num_test_points,
        output_path,
    ):
        """
        Constructor for the InferenceDomainDecomposition class

        :param decomposed_domain: The decomposed domain
        :type decomposed_domain: DecomposedDomain
        :param model: The model
        :type model: tf.keras.Model
        :param datahandler: The datahandler
        :type datahandler: DataHandlerDomainDecomposition
        :param test_points: The test points
        :type test_points: tf.Tensor
        :param exact_solution: The exact solution
        :type exact_solution: function
        """
        self.decomposed_domain = decomposed_domain
        self.model = model
        self.datahandler = datahandler
        self.window_function = window_function
        self.test_points = test_points
        self.exact_solution = exact_solution
        self.num_test_points_x = num_test_points[0]
        self.num_test_points_y = num_test_points[1]

        self.dtype = self.datahandler[0].dtype
        self.output_path = output_path

        self.calculate_means_nd_stds()
        self.test_points = tf.cast(self.test_points, self.dtype)

    def calculate_means_nd_stds(self):
        self.x_means = []
        self.x_stds = []
        self.y_means = []
        self.y_stds = []
        for i in range(len(self.model)):
            self.x_min_limit = self.decomposed_domain.subdomain_boundary_limits[i][0]
            self.x_max_limit = self.decomposed_domain.subdomain_boundary_limits[i][1]
            self.y_min_limits = self.decomposed_domain.subdomain_boundary_limits[i][2]
            self.y_max_limits = self.decomposed_domain.subdomain_boundary_limits[i][3]
            x_mean = (self.x_min_limit + self.x_max_limit) / 2.0
            x_std = (self.x_max_limit - self.x_min_limit) / 2.0
            y_mean = (self.y_min_limits + self.y_max_limits) / 2.0
            y_std = (self.y_max_limits - self.y_min_limits) / 2.0
            self.x_means.append(x_mean)
            self.x_stds.append(x_std)
            self.y_means.append(y_mean)
            self.y_stds.append(y_std)

    def compute_solution_nd_error(self):
        """
        Compute the solution and the loss
        """

        y_predicted = tf.zeros((self.test_points.shape[0], 1), self.dtype)
        for i in range(len(self.model)):
            x_test = (self.test_points[:, 0:1] - self.x_means[i]) / self.x_stds[i]
            y_test = (self.test_points[:, 1:2] - self.y_means[i]) / self.y_stds[i]
            # x_test = self.test_points[:, 0:1]
            # y_test = self.test_points[:, 1:2]

            normalized_test_points = tf.concat([x_test, y_test], axis=1)

            subdomain_y_predicted = self.model[i](normalized_test_points)
            subdomain_y_predicted = (
                self.decomposed_domain.unnormalizing_factor * subdomain_y_predicted
            )
            window_function_values = tf.cast(
                self.window_function.evaluate_window_function(
                    self.test_points[:, 0:1], self.test_points[:, 1:2], i
                ),
                self.dtype,
            )
            subdomain_y_predicted = subdomain_y_predicted * window_function_values
            

            y_predicted = y_predicted + subdomain_y_predicted

        y_predicted = (
                tf.tanh(
                    (self.decomposed_domain.hard_constraints_factor) * self.test_points[:, 0:1]
                )
                * tf.tanh(
                    (self.decomposed_domain.hard_constraints_factor) * self.test_points[:, 1:2]
                )
                * tf.tanh(
                    (self.decomposed_domain.hard_constraints_factor)
                    * (self.test_points[:, 0:1] - 1.0)
                )
                * tf.tanh(
                    (self.decomposed_domain.hard_constraints_factor)
                    * (self.test_points[:, 1:2] - 1.0)
                )
                * y_predicted
            )

        self.plot_solution_nd_error(
            y_predicted,
            self.exact_solution(self.test_points[:, 0:1], self.test_points[:, 1:2]),
            self.test_points,
        )
        y_predicted = y_predicted.numpy().reshape(-1,)
        y_exact = self.exact_solution(self.test_points[:, 0:1], self.test_points[:, 1:2]).reshape(-1,)
        error = y_exact - y_predicted
        linf_error = np.max(np.abs(error))
        l2_error = np.sqrt(np.mean(error ** 2))
        return linf_error, l2_error


    def plot_solution_nd_error(self, y_predicted, y_exact, test_points):
        """
        Plot the solution and the error
        """
        y_predicted = y_predicted.numpy().reshape(self.num_test_points_x, self.num_test_points_y)
        y_exact = y_exact.reshape(self.num_test_points_x, self.num_test_points_y)
        X = test_points[:, 0].numpy().reshape(self.num_test_points_x, self.num_test_points_y)
        Y = test_points[:, 1].numpy().reshape(self.num_test_points_x, self.num_test_points_y)

        plot_contour(
            x=X,
            y=Y,
            z=y_predicted,
            title="Predicted Solution",
            output_path=self.output_path,
            filename="predicted_solution",
        )
        plot_contour(
            x=X,
            y=Y,
            z=y_exact,
            title="Exact Solution",
            output_path=self.output_path,
            filename="exact_solution",
        )
        plot_contour(
            x=X,
            y=Y,
            z=np.abs(y_exact - y_predicted),
            title="Error",
            output_path=self.output_path,
            filename="error",
        )

        pass
