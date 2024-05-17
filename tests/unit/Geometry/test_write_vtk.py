# Author: Thivin Anandh. D
# Routines to check the VTK routines

import pytest
import numpy as np
import shutil
from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D

import os
import numpy as np
from pathlib import Path


def test_write_vtk_internal():
    """
    Test the write_vtk method for internal geometry.
    """
    # use pathlib to create a directory "tests/test_dump"
    Path("tests/test_dump").mkdir(parents=True, exist_ok=True)

    # Define the geometry
    domain = Geometry_2D("quadrilateral", "internal", 10, 10, "tests/test_dump")

    # read internal mesh
    cells, boundary_points = domain.generate_quad_mesh_internal(
        x_limits=[0, 1], y_limits=[0, 1], n_cells_x=4, n_cells_y=4, num_boundary_points=100
    )

    # Define test inputs
    output_path = "tests/test_dump"
    filename = "internal.vtk"

    # Check if the output file exists
    assert os.path.exists(os.path.join(output_path, filename))

    # delete the assets
    shutil.rmtree("tests/test_dump")


@pytest.mark.parametrize("mesh_generation_method", ["external", "internal"])
def test_write_vtk_solution_mismatch(mesh_generation_method):
    """
    Test the write_vtk method when the number of solution columns and data names do not match.
    """
    Path("tests/dump").mkdir(parents=True, exist_ok=True)

    # Define test inputs
    solution = np.array([[1, 2], [3, 4]])
    output_path = "tests/dump"
    filename = "output.vtk"
    data_names = ["data1", "data2", "data3"]

    # Define the geometry
    domain = Geometry_2D("quadrilateral", mesh_generation_method, 10, 10, "tests/dump")

    if mesh_generation_method == "internal":
        # read internal mesh
        cells, boundary_points = domain.generate_quad_mesh_internal(
            x_limits=[0, 1], y_limits=[0, 1], n_cells_x=4, n_cells_y=4, num_boundary_points=100
        )
    elif mesh_generation_method == "external":
        # read external mesh
        cells, boundary_points = domain.read_mesh(
            mesh_file="tests/support_files/circle_quad.mesh",
            boundary_point_refinement_level=2,
            bd_sampling_method="uniform",
            refinement_level=0,
        )
    else:
        pass

    with pytest.raises(ValueError):
        domain.write_vtk(solution, output_path, filename, data_names)

    shutil.rmtree("tests/dump")
