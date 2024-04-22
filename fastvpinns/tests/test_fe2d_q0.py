from src.FE_2D.basis_function_2d import *
import numpy as np
import pandas as pd
import pytest


def helper_function(xi, eta, function):
    """
    Helper function to compute the value of the basis functions or gradients at the given (xi, eta) coordinates.
    """
    value_array = []
    for _, xi_val in enumerate(xi):
        for _, eta_val in enumerate(eta):
            val = function(
                xi=xi_val, eta=eta_val
            )  # In python, the values will be returned as a
            if np.isscalar(val):  # 0-dimensional array, so we need to convert it
                value_array.append(val)
            else:
                for _, val_i in enumerate(val):
                    value_array.append(val_i)

    return np.array(value_array)


def test_bq0_value():
    """
    Funciton is used to test the value method of Basis2DQ0
    """
    # Call the value method of Basis2DQ0
    xi = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)

    # Create an instance of Basis2DQ0
    bq0 = Basis2DQ0()

    n_shape_functions = bq0.num_shape_functions

    validation_data = pd.read_csv("tests/Q0_result.csv", dtype=float, header=None)

    # For Q0
    validation_values = validation_data.iloc[0, :][:-1].copy()
    bq0_value_array = helper_function(xi, eta, bq0.value)
    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_value_array, validation_values, atol=1e-6)

    # For Q0 gradx
    validation_values = validation_data.iloc[1, :][:-1]
    bq0_gradx_array = helper_function(xi, eta, bq0.gradx)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_gradx_array, validation_values, atol=1e-6)

    # For Q0 grady
    validation_values = validation_data.iloc[2, :][:-1]
    bq0_grady_array = helper_function(xi, eta, bq0.grady)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_grady_array, validation_values, atol=1e-6)

    # For Q0 gradxx
    validation_values = validation_data.iloc[3, :][:-1]
    bq0_gradxx_array = helper_function(xi, eta, bq0.gradxx)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_gradxx_array, validation_values, atol=1e-6)

    # For Q0 gradxy
    validation_values = validation_data.iloc[4, :][:-1]
    bq0_gradxy_array = helper_function(xi, eta, bq0.gradxy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_gradxy_array, validation_values, atol=1e-6)

    # For Q0 gradyy
    validation_values = validation_data.iloc[5, :][:-1]
    bq0_gradyy_array = helper_function(xi, eta, bq0.gradyy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq0_gradyy_array, validation_values, atol=1e-6)


def test_bq1_value():
    """
    Function is used to test the value method of Basis2DQ1
    """
    # Call the value method of Basis2DQ1
    xi = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)

    # Create an instance of Basis2DQ1
    bq1 = Basis2DQ1()

    n_shape_functions = bq1.num_shape_functions

    validation_data = pd.read_csv("tests/Q1_result.csv", dtype=float, header=None)

    # For Q1
    validation_values = validation_data.iloc[0, :][:-1].copy()
    bq1_value_array = helper_function(xi, eta, bq1.value)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_value_array, validation_values, atol=1e-6)
    # For Q1 gradx
    validation_values = validation_data.iloc[1, :][:-1]
    bq1_gradx_array = helper_function(xi, eta, bq1.gradx)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_gradx_array, validation_values, atol=1e-6)

    # For Q1 grady
    validation_values = validation_data.iloc[2, :][:-1]
    bq1_grady_array = helper_function(xi, eta, bq1.grady)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_grady_array, validation_values, atol=1e-6)

    # For Q1 gradxx
    validation_values = validation_data.iloc[3, :][:-1]
    bq1_gradxx_array = helper_function(xi, eta, bq1.gradxx)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_gradxx_array, validation_values, atol=1e-6)

    # For Q1 gradxy
    validation_values = validation_data.iloc[4, :][:-1]
    bq1_gradxy_array = helper_function(xi, eta, bq1.gradxy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_gradxy_array, validation_values, atol=1e-6)

    # For Q1 gradyy
    validation_values = validation_data.iloc[5, :][:-1]
    bq1_gradyy_array = helper_function(xi, eta, bq1.gradyy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq1_gradyy_array, validation_values, atol=1e-6)


def test_bq2_value():
    """
    Function is used to test the value method of Basis2DQ2
    """
    # Call the value method of Basis2DQ2
    xi = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)

    # Create an instance of Basis2DQ2
    bq2 = Basis2DQ2()

    n_shape_functions = bq2.num_shape_functions

    validation_data = pd.read_csv("tests/Q2_result.csv", dtype=float, header=None)

    # For Q2
    validation_values = validation_data.iloc[0, :][:-1].copy()
    bq2_value_array = helper_function(xi, eta, bq2.value)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_value_array, validation_values, atol=1e-6)

    # For Q2 gradx
    validation_values = validation_data.iloc[1, :][:-1]
    bq2_gradx_array = helper_function(xi, eta, bq2.gradx)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradx_array, validation_values, atol=1e-6)

    # For Q2 grady
    validation_values = validation_data.iloc[2, :][:-1]
    bq2_grady_array = helper_function(xi, eta, bq2.grady)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_grady_array, validation_values, atol=1e-6)

    # For Q2 gradxx
    validation_values = validation_data.iloc[3, :][:-1]
    bq2_gradxx_array = helper_function(xi, eta, bq2.gradxx)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradxx_array, validation_values, atol=1e-5)

    # For Q2 gradxy
    validation_values = validation_data.iloc[4, :][:-1]
    bq2_gradxy_array = helper_function(xi, eta, bq2.gradxy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradxy_array, validation_values, atol=1e-5)

    # For Q2 gradyy
    validation_values = validation_data.iloc[5, :][:-1]
    bq2_gradyy_array = helper_function(xi, eta, bq2.gradyy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradyy_array, validation_values, atol=1e-6)


def test_bq2_value():
    """
    Function is used to test the value method of Basis2DQ2
    """
    # Call the value method of Basis2DQ2
    xi = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)

    # Create an instance of Basis2DQ2
    bq2 = Basis2DQ2()

    n_shape_functions = bq2.num_shape_functions

    validation_data = pd.read_csv("tests/Q2_result.csv", dtype=float, header=None)

    # For Q2
    validation_values = validation_data.iloc[0, :][:-1].copy()
    bq2_value_array = helper_function(xi, eta, bq2.value)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_value_array, validation_values, atol=1e-6)

    # For Q2 gradx
    validation_values = validation_data.iloc[1, :][:-1]
    bq2_gradx_array = helper_function(xi, eta, bq2.gradx)

    # Compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradx_array, validation_values, atol=1e-6)

    # For Q2 grady
    validation_values = validation_data.iloc[2, :][:-1]
    bq2_grady_array = helper_function(xi, eta, bq2.grady)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_grady_array, validation_values, atol=1e-6)

    # For Q2 gradxx
    validation_values = validation_data.iloc[3, :][:-1]
    bq2_gradxx_array = helper_function(xi, eta, bq2.gradxx)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradxx_array, validation_values, atol=1e-5)

    # For Q2 gradxy
    validation_values = validation_data.iloc[4, :][:-1]
    bq2_gradxy_array = helper_function(xi, eta, bq2.gradxy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradxy_array, validation_values, atol=1e-5)

    # For Q2 gradyy
    validation_values = validation_data.iloc[5, :][:-1]
    bq2_gradyy_array = helper_function(xi, eta, bq2.gradyy)

    # compare the two arrays element-wise, if the absolute difference is less than 1e-6, then the test passes
    assert np.allclose(bq2_gradyy_array, validation_values, atol=1e-6)


if __name__ == "__main__":
    """
    Run All the test cases for the basis functions
    """
    # Run the test cases for Q0
    test_bq0_value()

    # Run the test cases for Q1
    test_bq1_value()

    # Run the test cases for Q2
    test_bq2_value()

    print("Everything passed")
