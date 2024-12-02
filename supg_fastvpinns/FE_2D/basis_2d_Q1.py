
# Purpose: Defines the basis functions for a 2D Q1 element.
# Reference: ParMooN -  File: BF_C_Q_Q1_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ1(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q1 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=4)



    #  ***********************************************************************
    #  Q1 element, conforming, 2D
    #  ***********************************************************************

    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0.25*(1-xi-eta+xi*eta)
        values[1, :] = 0.25*(1+xi-eta-xi*eta)
        values[2, :] = 0.25*(1-xi+eta-xi*eta)
        values[3, :] = 0.25*(1+xi+eta+xi*eta)

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0.25*(-1+eta)
        values[1, :] = 0.25*( 1-eta)
        values[2, :] = 0.25*(-1-eta)
        values[3, :] = 0.25*( 1+eta)

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0.25*(-1+xi)
        values[1, :] = 0.25*(-1-xi)
        values[2, :] = 0.25*( 1-xi)
        values[3, :] = 0.25*( 1+xi)

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0
        values[1, :] = 0
        values[2, :] = 0
        values[3, :] = 0

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0.25
        values[1, :] = -0.25
        values[2, :] = -0.25 
        values[3, :] = 0.25

        return values

    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = 0
        values[1, :] = 0
        values[2, :] = 0
        values[3, :] = 0  

        return values

