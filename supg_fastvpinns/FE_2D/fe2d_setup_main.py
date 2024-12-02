# This program will be used to setup the FE2D and quadrature rule for a given cell based on the 
# given mesh and the degree of the basis functions

# Author: Thivin Anandh D
# Date:  30/Aug/2023

# Importing the required libraries
from .basis_function_2d import *

# import Quadrature rules
from .quadratureformulas_quad2d import *
from .quadratureformulas_tri2d import *

# import base class for FE transformation
from .fe_transformation_2d import *


class FE2DSetupMain:
    """
    This class is used to setup the FE2D and quadrature rule for a given cell based on the given mesh and the degree of the basis functions.
    """
    def __init__(self, cell_type: str, fe_order: int, fe_type: str, quad_order: int, quad_type: str):
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type

        self.assign_basis_function()
    
    def assign_basis_for_fem(self, fe_order):
        """
        This method assigns the basis function based on the cell type and the fe_order for FEM (PARMOON) basis functions.
        """
        if fe_order == 0:
            return Basis2DQ0()
        elif fe_order == 1:
            return Basis2DQ1()
        elif fe_order == 2:
            return Basis2DQ2()
        elif fe_order == 3:
            return Basis2DQ3()
        elif fe_order == 4:
            return Basis2DQ4()
        elif fe_order == 5:
            return Basis2DQ5()
        elif fe_order == 6:
            return Basis2DQ6()
        elif fe_order == 7:
            return Basis2DQ7()
        elif fe_order == 8:
            return Basis2DQ8()
        elif fe_order == 9:
            return Basis2DQ9()
        else:
            print(f'Invalid FE order {self.fe_order} in {self.__class__.__name__} from {__name__}.')
            raise ValueError('FE order should be between 0 and 9.')
        

    def assign_basis_function(self) -> BasisFunction2D:
        """
        This method assigns the basis function based on the cell type and the fe_order.
        """
        if self.cell_type == 'quadrilateral':
            self.n_nodes = 4
            
            if self.fe_type == 'parmoon':
                return self.assign_basis_for_fem(self.fe_order)
            
            elif self.fe_type == 'legendre':
                return Basis2DQN(self.fe_order**2)
            
            elif self.fe_type == 'jacobi': 
                return Basis2DQNJacobi(self.fe_order**2)
            else:
                print(f'Invalid FE order {self.fe_order} in {self.__class__.__name__} from {__name__}.')
                raise ValueError('FE order should be one of the : "parmoon", "legendre" and "jacobi".')
        
        if self.cell_type == 'triangle':
            self.n_nodes = 3
            
            if self.fe_order == 0:
                return Basis2DP0()
            elif self.fe_order == 1:
                return Basis2DP1()
            elif self.fe_order == 2:
                return Basis2DP2()
            elif self.fe_order == 3:
                return Basis2DP3()
            elif self.fe_order == 4:
                return Basis2DP4()
            elif self.fe_order == 5:
                return Basis2DP5()
            elif self.fe_order == 6:
                return Basis2DP6()
            
            else:
                print(f'Invalid FE order {self.fe_order} in {self.__class__.__name__} from {__name__}.')
                raise ValueError('FE order should be between 0 and 6.')
        
        print(f'Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}.')
    
    def assign_quadrature_rules(self):
        """
        This method assigns the quadrature rule based on the quad_order.
        """
        if self.cell_type == 'quadrilateral':
            if self.quad_order < 2:
                raise ValueError('Quad order should be greater than 1.')
            elif self.quad_order >= 2 and self.quad_order <= 9999:
                weights, xi, eta = Quadratureformulas_Quad2D(self.quad_order, self.quad_type).get_quad_values()
                return weights, xi, eta
            
            else:
                print(f'Invalid quad order {self.quad_order} in {self.__class__.__name__} from {__name__}.')
                raise ValueError('Quad order should be between 1 and 9999.')
        
        if self.cell_type == 'triangle':
            weights, xi, eta = Quadratureformulas_Tri2D(self.quad_order).get_quad_values()
            return weights, xi, eta
        
        raise ValueError(f'Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}.')
    
    def assign_fe_transformation(self, fe_transformation_type, cell_coordinates) -> FETransforamtion2D:
        """
        This method assigns the FE transformation based on the cell type.
        """
        if self.cell_type == 'quadrilateral':
            if fe_transformation_type == 'affine':
                return QuadAffin(cell_coordinates)
            elif fe_transformation_type == 'bilinear':
                return QuadBilinear(cell_coordinates)
            else:
                raise ValueError(f'Invalid FE transformation type {fe_transformation_type} in {self.__class__.__name__} from {__name__}.')
            
        elif self.cell_type == 'triangle':
            if fe_transformation_type == 'affine':
                return TriAffin(cell_coordinates)
            else:
                raise ValueError(f'Invalid FE transformation type {fe_transformation_type} in {self.__class__.__name__} from {__name__}.')
            
        else:
            raise ValueError(f'Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}.')