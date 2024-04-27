import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fastvpinns.utils.plot_utils import (
    plot_loss_function,
    plot_array,
    plot_multiple_loss_function,
    plot_inverse_test_loss_function,
    plot_test_loss_function,
    plot_test_time_loss_function,
    plot_contour,
    plot_inverse_param_function,
)


# Create a fixture for the output path
@pytest.fixture
def output_path():
    path = "tests/dump"
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


# Create a fixture for the dummy data
@pytest.fixture
def dummy_data():
    return np.arange(100)


# Test each function
def test_plot_loss_function(output_path, dummy_data):
    plot_loss_function(dummy_data, output_path)
    assert os.path.exists(f"{output_path}/loss_function.png")


def test_plot_array(output_path, dummy_data):
    plot_array(dummy_data, output_path, "dummy", "Dummy Plot")
    assert os.path.exists(f"{output_path}/dummy.png")


def test_plot_multiple_loss_function(output_path, dummy_data):
    plot_multiple_loss_function(
        [dummy_data, dummy_data], output_path, "dummy", ["Dummy 1", "Dummy 2"], "Loss", "Dummy Plot"
    )
    assert os.path.exists(f"{output_path}/dummy.png")


def test_plot_inverse_test_loss_function(output_path, dummy_data):
    plot_inverse_test_loss_function(dummy_data, output_path)
    assert os.path.exists(f"{output_path}/test_inverse_loss_function.png")


def test_plot_test_loss_function(output_path, dummy_data):
    plot_test_loss_function(dummy_data, output_path)
    assert os.path.exists(f"{output_path}/test_loss_function.png")


def test_plot_test_time_loss_function(output_path, dummy_data):
    plot_test_time_loss_function(dummy_data, dummy_data, output_path)
    assert os.path.exists(f"{output_path}/test_time_loss_function.png")


def test_plot_contour(output_path):
    x, y = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    z = np.sin(np.sqrt(x**2 + y**2))
    plot_contour(x, y, z, output_path, "dummy", "Dummy Plot")
    assert os.path.exists(f"{output_path}/dummy.png")


def test_plot_inverse_param_function(output_path, dummy_data):
    plot_inverse_param_function(dummy_data, "Dummy Param", 50, output_path, "dummy")
    assert os.path.exists(f"{output_path}/dummy.png")
    assert os.path.exists(f"{output_path}/dummy_absolute_error.png")
