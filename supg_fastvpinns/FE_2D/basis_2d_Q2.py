
# Purpose: Defines the basis functions for a 2D Q2 element.
# Reference: ParMooN -  File: BF_C_Q_Q2_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ2(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q2 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=9)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0=0.5*xi*(xi-1)
        xi1=1-xi*xi
        xi2=0.5*xi*(xi+1)
        eta0=0.5*eta*(eta-1)
        eta1=1-eta*eta
        eta2=0.5*eta*(eta+1)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi0*eta1
        values[4, :] = xi1*eta1
        values[5, :] = xi2*eta1
        values[6, :] = xi0*eta2
        values[7, :] = xi1*eta2
        values[8, :] = xi2*eta2

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0=xi-0.5
        xi1=-2*xi
        xi2=xi+0.5
        eta0=0.5*eta*(eta-1)
        eta1=1-eta*eta
        eta2=0.5*eta*(eta+1)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi0*eta1
        values[4, :] = xi1*eta1
        values[5, :] = xi2*eta1
        values[6, :] = xi0*eta2
        values[7, :] = xi1*eta2
        values[8, :] = xi2*eta2

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0=0.5*xi*(xi-1)
        xi1=1-xi*xi
        xi2=0.5*xi*(xi+1)
        eta0=eta-0.5
        eta1=-2*eta
        eta2=eta+0.5

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi0*eta1
        values[4, :] = xi1*eta1
        values[5, :] = xi2*eta1
        values[6, :] = xi0*eta2
        values[7, :] = xi1*eta2
        values[8, :] = xi2*eta2

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        eta0=0.5*eta*(eta-1)
        eta1=1-eta*eta
        eta2=0.5*eta*(eta+1)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = eta0
        values[1, :] = -2*eta0
        values[2, :] = eta0
        values[3, :] = eta1
        values[4, :] = -2*eta1
        values[5, :] = eta1
        values[6, :] = eta2
        values[7, :] = -2*eta2
        values[8, :] = eta2

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0=xi-0.5
        xi1=-2*xi
        xi2=xi+0.5
        eta0=eta-0.5
        eta1=-2*eta
        eta2=eta+0.5

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi0*eta1
        values[4, :] = xi1*eta1
        values[5, :] = xi2*eta1
        values[6, :] = xi0*eta2
        values[7, :] = xi1*eta2
        values[8, :] = xi2*eta2

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0=0.5*xi*(xi-1)
        xi1=1-xi*xi
        xi2=0.5*xi*(xi+1)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0
        values[1, :] = xi1
        values[2, :] = xi2
        values[3, :] = -2*xi0
        values[4, :] = -2*xi1
        values[5, :] = -2*xi2
        values[6, :] = xi0
        values[7, :] = xi1
        values[8, :] = xi2 

        return values



