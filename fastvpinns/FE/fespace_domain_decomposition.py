import numpy as np
from .fespace2d import Fespace2D


class FESpaceDomainDecomposition(Fespace2D):
    """
    Finite element spaces for domain decomposition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mesh=None, **kwargs)
        assert (
            self.mesh is None
        ), 'FESpaceDomainDecomposition does not support external meshes. domain.mesh should be None.'
        pass
