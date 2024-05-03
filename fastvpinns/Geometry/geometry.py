"""
This module, `geometry.py`, contains the `Geometry` Abstract class which defines functions to read mesh from Gmsh and 
generate internal mesh for 2D and 3D geometries. 

Author: Thivin Anandh D
Date: 03-May-2024
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import meshio
from pyDOE import lhs

import gmsh

from abc import abstractmethod


class Geometry:
    """
    Abstract class which defines functions to read mesh from Gmsh and internal mesh for 2D problems.

    :param mesh_type: The type of mesh to be used.
    :type mesh_type: str
    :param mesh_generation_method: The method used to generate the mesh.
    :type mesh_generation_method: str
    """

    def __init__(self, mesh_type, mesh_generation_method):
        self.mesh_type = mesh_type
        self.mesh_generation_method = mesh_generation_method

    @abstractmethod
    def read_mesh(
        self,
        mesh_file: str,
        boundary_point_refinement_level: int,
        bd_sampling_method: str,
        refinement_level: int,
    ):
        """
        Abstract method to read mesh from Gmsh.

        :param mesh_file: The path to the mesh file.
        :type mesh_file: str
        :param boundary_point_refinement_level: The refinement level of the boundary points.
        :type boundary_point_refinement_level: int
        :param bd_sampling_method: The method used to sample the boundary points.
        :type bd_sampling_method: str
        :param refinement_level: The refinement level of the mesh.
        :type refinement_level: int
        """

    @abstractmethod
    def generate_vtk_for_test(self):
        """
        Generates a VTK from Mesh file (External) or using gmsh (for Internal).

        :return: None
        """

    @abstractmethod
    def get_test_points(self):
        """
        This function is used to extract the test points from the given mesh

        Parameters:
        None

        Returns:
        test_points (numpy.ndarray): The test points for the given domain
        """
