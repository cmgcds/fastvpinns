import sys
import yaml
import numpy as np

from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.compute_utils import compute_errors_combined


def get_errors(
    model,
    console,
    y_pred,
    y_exact,
    Y_Exact_Matrix,
    i_n_test_points_x,
    i_n_test_points_y,
    i_output_path,
    epoch,
    loss,
    num_epochs,
):
    """
    Calculate and return various error metrics and loss values.

    Args:
        model (object): The trained model.
        console (object): The console object for printing messages.
        y_exact (array): The exact solution.
        Y_Exact_Matrix (array): The matrix of exact solutions.
        i_n_test_points_x (int): The number of test points in the x-direction.
        i_n_test_points_y (int): The number of test points in the y-direction.
        i_output_path (str): The output path for saving plots.
        epoch (int): The current epoch number.
        loss (dict): The dictionary containing different loss values.
        num_epochs (int): The total number of epochs.

    Returns:
        dict: A dictionary containing various error metrics and loss values.
    """

    # Compute error metrics
    l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative = (
        compute_errors_combined(y_exact, y_pred)
    )

    # Print epoch information
    console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
    console.print("[bold]--------------------[/bold]")
    console.print(
        f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
    )
    console.print(f"Test Losses        || L1 Error : {l1_error:.3e}", end=" ")
    console.print(f"L2 Error : {l2_error:.3e}", end=" ")
    console.print(f"Linf Error : {linf_error:.3e}", end="\n")

    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'l2_error_relative': l2_error_relative,
        'linf_error_relative': linf_error_relative,
        'l1_error': l1_error,
        'l1_error_relative': l1_error_relative,
        'loss_pde': loss_pde,
        'loss_dirichlet': loss_dirichlet,
        'total_loss': total_loss,
    }


def plot_results(
    loss_array,
    test_loss_array,
    y_pred,
    X,
    Y,
    Y_Exact_Matrix,
    i_output_path,
    epoch,
    i_n_test_points_x,
    i_n_test_points_y,
):
    """
    Plot the loss function, test loss function, solution, and error.

    Args:
        loss_array (array): Array of loss values during training.
        test_loss_array (array): Array of test loss values during training.
        y_pred (array): Predicted solution.
        X (array): X-coordinates of the grid.
        Y (array): Y-coordinates of the grid.
        Y_Exact_Matrix (array): Matrix of exact solutions.
        i_output_path (str): Output path for saving plots.
        epoch (int): Current epoch number.
        i_n_test_points_x (int): Number of test points in the x-direction.
        i_n_test_points_y (int): Number of test points in the y-direction.
    """
    # plot loss
    plot_loss_function(loss_array, i_output_path)  # plots NN loss
    plot_test_loss_function(test_loss_array, i_output_path)  # plots test loss

    # plot solution
    y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
    error = np.abs(Y_Exact_Matrix - y_pred)
    plot_contour(
        x=X,
        y=Y,
        z=y_pred,
        output_path=i_output_path,
        filename=f"prediction_{epoch+1}",
        title="Prediction",
    )
    plot_contour(
        x=X, y=Y, z=error, output_path=i_output_path, filename=f"error_{epoch+1}", title="Error"
    )
