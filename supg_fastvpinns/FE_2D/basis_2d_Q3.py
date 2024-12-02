
# Purpose: Defines the basis functions for a 2D Q3 element.
# Reference: ParMooN -  File: BF_C_Q_Q3_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ3(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q3 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=16)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0 = -0.625E-1*(3.0*xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi1 =  0.5625*(xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi2 = -0.5625*(xi+1.0)*(3.0*xi+1.0)*(xi-1.0)
        xi3 = 0.625E-1*(xi+1.0)*(3.0*xi+1.0)*(3.0*xi-1.0)
        eta0 = -0.625E-1*(3.0*eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta1 =  0.5625*(eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta2 = -0.5625*(eta+1.0)*(3.0*eta+1.0)*(eta-1.0)
        eta3 = 0.625E-1*(eta+1.0)*(3.0*eta+1.0)*(3.0*eta-1.0)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi*xi
        xi0 = -0.16875E1*t1+0.1125E1*xi+0.625E-1
        xi1 = 0.50625E1*t1-0.1125E1*xi-0.16875E1
        xi2 = -0.50625E1*t1-0.1125E1*xi+0.16875E1
        xi3 = 0.16875E1*t1-0.625E-1+0.1125E1*xi
        eta0 = -0.625E-1*(3.0*eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta1 =  0.5625*(eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta2 = -0.5625*(eta+1.0)*(3.0*eta+1.0)*(eta-1.0)
        eta3 = 0.625E-1*(eta+1.0)*(3.0*eta+1.0)*(3.0*eta-1.0)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1=eta*eta
        xi0 = -0.625E-1*(3.0*xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi1 =  0.5625*(xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi2 = -0.5625*(xi+1.0)*(3.0*xi+1.0)*(xi-1.0)
        xi3 = 0.625E-1*(xi+1.0)*(3.0*xi+1.0)*(3.0*xi-1.0)
        eta0 = -0.16875E1*t1+0.1125E1*eta+0.625E-1
        eta1 = 0.50625E1*t1-0.1125E1*eta-0.16875E1
        eta2 = -0.50625E1*t1-0.1125E1*eta+0.16875E1
        eta3 = 0.16875E1*t1-0.625E-1+0.1125E1*eta

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0 = -0.3375E1*xi+0.1125E1
        xi1 = 1.0125E1*xi-0.1125E1
        xi2 = -1.0125E1*xi-0.1125E1
        xi3 = 0.3375E1*xi+0.1125E1
        eta0 = -0.625E-1*(3.0*eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta1 =  0.5625*(eta+1.0)*(3.0*eta-1.0)*(eta-1.0)
        eta2 = -0.5625*(eta+1.0)*(3.0*eta+1.0)*(eta-1.0)
        eta3 = 0.625E-1*(eta+1.0)*(3.0*eta+1.0)*(3.0*eta-1.0)

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi*xi
        xi0 = -0.16875E1*t1+0.1125E1*xi+0.625E-1
        xi1 = 0.50625E1*t1-0.1125E1*xi-0.16875E1
        xi2 = -0.50625E1*t1-0.1125E1*xi+0.16875E1
        xi3 = 0.16875E1*t1-0.625E-1+0.1125E1*xi
        t2=eta*eta
        eta0 = -0.16875E1*t2+0.1125E1*eta+0.625E-1
        eta1 = 0.50625E1*t2-0.1125E1*eta-0.16875E1
        eta2 = -0.50625E1*t2-0.1125E1*eta+0.16875E1
        eta3 = 0.16875E1*t2-0.625E-1+0.1125E1*eta

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3

        return values

    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = -0.625E-1*(3.0*xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi1 =  0.5625*(xi+1.0)*(3.0*xi-1.0)*(xi-1.0)
        xi2 = -0.5625*(xi+1.0)*(3.0*xi+1.0)*(xi-1.0)
        xi3 = 0.625E-1*(xi+1.0)*(3.0*xi+1.0)*(3.0*xi-1.0)
        eta0 = -0.3375E1*eta+0.1125E1
        eta1 = 1.0125E1*eta-0.1125E1
        eta2 = -1.0125E1*eta-0.1125E1
        eta3 = 0.3375E1*eta+0.1125E1

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] = xi0*eta0
        values[1, :] = xi1*eta0
        values[2, :] = xi2*eta0
        values[3, :] = xi3*eta0
        values[4, :] = xi0*eta1
        values[5, :] = xi1*eta1
        values[6, :] = xi2*eta1
        values[7, :] = xi3*eta1
        values[8, :] = xi0*eta2
        values[9, :] = xi1*eta2
        values[10, :] = xi2*eta2
        values[11, :] = xi3*eta2
        values[12, :] = xi0*eta3
        values[13, :] = xi1*eta3
        values[14, :] = xi2*eta3
        values[15, :] = xi3*eta3  

        return values


