
# Purpose: Defines the basis functions for a 2D P3 element.
# Reference: ParMooN -  File: BF_C_T_P3_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DP3(BasisFunction2D):
    """
    This class defines the basis functions for a 2D P3 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=10)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-2.0/3.0
        t3 = xi+eta-1.0
        t6 = xi*(xi-1.0/3.0)
        t16 = eta*(eta-1.0/3.0)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0/2.0*(xi+eta-1.0/3.0)*t1*t3
        values[1, :] =  27.0/2.0*xi*t1*t3
        values[2, :] =  -27.0/2.0*t6*t3
        values[3, :] =  9.0/2.0*t6*(xi-2.0/3.0)
        values[4, :] =  27.0/2.0*t1*t3*eta
        values[5, :] =  -27.0*xi*eta*t3
        values[6, :] =  27.0/2.0*t6*eta
        values[7, :] =  -27.0/2.0*t16*t3
        values[8, :] =  27.0/2.0*t16*xi
        values[9, :] =  9.0/2.0*t16*(eta-2.0/3.0)

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-2.0/3.0
        t2 = xi+eta-1.0
        t3 = t1*t2
        t4 = xi*t2
        t7 = xi-1.0/3.0
        t9 = xi*t7
        t11 = xi-2.0/3.0
        t15 = t2*eta
        t18 = xi*eta
        t23 = eta*(eta-1.0/3.0)
        t24 = xi+eta-1.0/3.0
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0/2.0*t3-9.0/2.0*t24*t2-9.0/2.0*t24*t1
        values[1, :] =  27.0/2.0*t3+27.0/2.0*t4+27.0/2.0*xi*t1
        values[2, :] =  -27.0/2.0*t7*t2-27.0/2.0*t4-27.0/2.0*t9
        values[3, :] =  9.0/2.0*t7*t11+9.0/2.0*xi*t11+9.0/2.0*t9
        values[4, :] =  27.0/2.0*t15+27.0/2.0*t1*eta
        values[5, :] =  -27.0*t15-27.0*t18
        values[6, :] =  27.0/2.0*t18+27.0/2.0*t7*eta
        values[7, :] =  -27.0/2.0*t23
        values[8, :] =  27.0/2.0*t23
        values[9, :] =  0.0

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-1.0
        t2 = xi*t1
        t3 = xi+eta-2.0/3.0
        t7 = xi*(xi-1.0/3.0)
        t8 = t1*eta
        t10 = t1*t3
        t12 = xi*eta
        t14 = eta-1.0/3.0
        t16 = eta*t14
        t20 = eta-2.0/3.0
        t24 = xi+eta-1.0/3.0
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0/2.0*t10-9.0/2.0*t24*t1-9.0/2.0*t24*t3
        values[1, :] =  27.0/2.0*t2+27.0/2.0*xi*t3
        values[2, :] =  -27.0/2.0*t7
        values[3, :] =  0.0
        values[4, :] =  27.0/2.0*t8+27.0/2.0*t3*eta+27.0/2.0*t10
        values[5, :] =  -27.0*t2-27.0*t12
        values[6, :] =  27.0/2.0*t7
        values[7, :] =  -27.0/2.0*t14*t1-27.0/2.0*t8-27.0/2.0*t16
        values[8, :] =  27.0/2.0*t14*xi+27.0/2.0*t12
        values[9, :] =  9.0/2.0*t14*t20+9.0/2.0*eta*t20+9.0/2.0*t16

        return values

    #  values of the derivatives in xi-xi direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-2.0/3.0
        t2 = xi+eta-1.0
        t7 = xi-1.0/3.0
        t11 = xi-2.0/3.0
        t24 = xi+eta-1.0/3.0

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0*(t1+t2+t24)
        values[1, :] =  27.0*(t1+t2+xi)
        values[2, :] =  -27.0*(t7+t2+xi)
        values[3, :] =  9.0*(t7+t11+xi)
        values[4, :] =  27.0*eta
        values[5, :] =  -54.0*eta
        values[6, :] =  27.0*eta
        values[7, :] =  0.0
        values[8, :] =  0.0
        values[9, :] =  0.0

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-2.0/3.0
        t2 = xi+eta-1.0
        t24 = xi+eta-1.0/3.0

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0*(t1+t2+t24)
        values[1, :] =  27.0/2.0*(t1+t2)+27.0*xi
        values[2, :] =  -27.0*xi+9.0/2.0
        values[3, :] =  0.0
        values[4, :] =  27.0/2.0*(t1+t2)+27.0*eta
        values[5, :] =  -54.0*(xi+eta-0.5)
        values[6, :] =  27.0*xi-9.0/2.0
        values[7, :] =  -27.0*eta+9.0/2.0
        values[8, :] =  27.0*eta-9.0/2.0
        values[9, :] =  0.0

        return values

    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t1 = xi+eta-1.0
        t3 = xi+eta-2.0/3.0
        t14 = eta-1.0/3.0
        t20 = eta-2.0/3.0
        t24 = xi+eta-1.0/3.0

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -9.0*(t3+t1+t24)
        values[1, :] =  27.0*xi
        values[2, :] =  0.0
        values[3, :] =  0.0
        values[4, :] =  27.0*(t1+eta+t3)
        values[5, :] =  -54.0*xi
        values[6, :] =  0.0
        values[7, :] =  -27.0*(t14+t1+eta)
        values[8, :] =  27.0*xi
        values[9, :] =  9.0*(t14+t20+eta)

        return values


