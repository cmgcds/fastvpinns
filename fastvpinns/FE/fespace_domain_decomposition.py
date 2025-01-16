import numpy as np
from .fespace2d import Fespace2D


class FESpaceDomainDecomposition(Fespace2D):
    """
    Finite element spaces for domain decomposition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            mesh=None,
            bound_function_dict=None,
            bound_condition_dict=None,
            boundary_points=None,
            **kwargs,
        )
        assert (
            self.mesh is None
        ), 'FESpaceDomainDecomposition does not support external meshes. domain.mesh should be None.'
        assert (
            self.bound_function_dict is None
        ), 'FESpaceDomainDecomposition does not support soft boundary constraints. domain.bound_function_dict should be None.'
        assert (
            self.bound_condition_dict is None
        ), 'FESpaceDomainDecomposition does not support hard boundary constraints. domain.bound_condition_dict should be None.'
        assert (
            self.boundary_points is None
        ), 'FESpaceDomainDecomposition does not support boundary point sampling, hard constraints are used instead. domain.boundary_points should be None.'
        pass

    def generate_dirichlet_boundary_data(self):
        pass

    def generate_dirichlet_boundary_data_vector(self, component):
        pass
