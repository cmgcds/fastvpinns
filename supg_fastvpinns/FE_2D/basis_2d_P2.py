
# Purpose: Defines the basis functions for a 2D P2 element.
# Reference: ParMooN -  File: BF_C_T_P2_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DP2(BasisFunction2D):
    """
    This class defines the basis functions for a 2D P2 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=6)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-1

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  t1*(2*xi+2*eta-1)
        values[1, :] =  -4*t1*xi
        values[2, :] =  xi*(2*xi-1)
        values[3, :] =  -4*t1*eta
        values[4, :] =  4*xi*eta
        values[5, :] =  eta*(2*eta-1)

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  4*xi+4*eta-3
        values[1, :] =  -8*xi-4*eta+4
        values[2, :] =  4*xi-1
        values[3, :] =  -4*eta
        values[4, :] =  4*eta
        values[5, :] =  0

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  4*xi+4*eta-3
        values[1, :] =  -4*xi
        values[2, :] =  0
        values[3, :] =  -4*xi-8*eta+4
        values[4, :] =  4*xi
        values[5, :] =  4*eta-1

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  4
        values[1, :] =  -8
        values[2, :] =  4
        values[3, :] =  0
        values[4, :] =  0
        values[5, :] =  0

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  4
        values[1, :] =  -4
        values[2, :] =  0
        values[3, :] =  -4
        values[4, :] =  4
        values[5, :] =  0

        return values

    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  4
        values[1, :] =  0
        values[2, :] =  0
        values[3, :] =  -8
        values[4, :] =  0
        values[5, :] =  4  

        return values



