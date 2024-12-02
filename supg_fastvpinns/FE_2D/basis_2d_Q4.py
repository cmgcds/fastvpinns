
# Purpose: Defines the basis functions for a 2D Q4 element.
# Reference: ParMooN -  File: BF_C_Q_Q4_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ4(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q4 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=25)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    
     
      
        xi0 = 0.666666666666666666667*xi*xi*xi*xi-0.666666666666666666667*xi*xi*xi - 0.166666666666666666667*xi*xi+0.166666666666666666667*xi
        xi1 = -2.66666666666666666667*xi*xi*xi*xi+1.33333333333333333333*xi*xi*xi+2.66666666666666666667*xi*xi-1.33333333333333333333*xi
        xi2 = 4.0*xi*xi*xi*xi-5.0*xi*xi+1.0
        xi3 = -2.66666666666666666667*xi*xi*xi*xi-1.33333333333333333333*xi*xi*xi+2.666666666666667*xi*xi+1.33333333333333333333*xi
        xi4 = 0.666666666666666666667*xi*xi*xi*xi+0.666666666666666666667*xi*xi*xi-0.166666666666666666667*xi*xi-0.166666666666666666667*xi
        eta0 = 0.666666666666666666667*eta*eta*eta*eta-0.666666666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta+0.166666666666666666667*eta
        eta1 = -2.66666666666666666667*eta*eta*eta*eta+1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta-1.33333333333333333333*eta
        eta2 = 4.0*eta*eta*eta*eta-5.0*eta*eta+1.0
        eta3 = -2.66666666666666666667*eta*eta*eta*eta-1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta+1.33333333333333333333*eta
        eta4 = 0.666666666666666666667*eta*eta*eta*eta+0.666666666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta-0.166666666666666666667*eta

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values



    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = 2.666666666666666666667*xi*xi*xi-2*xi*xi-0.333333333333333333333*xi+0.1666666666666666666667
        xi1 = -10.66666666666666666667*xi*xi*xi+4*xi*xi+5.33333333333333333333*xi-1.33333333333333333333
        xi2 = 16.0*xi*xi*xi-10.0*xi
        xi3 = -10.66666666666666666667*xi*xi*xi-4*xi*xi+5.33333333333333333333*xi+1.33333333333333333333
        xi4  = 2.666666666666666666667*xi*xi*xi+2*xi*xi-0.333333333333333333333*xi-0.1666666666666666666667
        eta0 = 0.666666666666666666667*eta*eta*eta*eta-0.6666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta+0.166666666666666666667*eta
        eta1 = -2.66666666666666666667*eta*eta*eta*eta+1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta-1.33333333333333333333*eta
        eta2 = 4.0*eta*eta*eta*eta-5.0*eta*eta+1.0
        eta3 = -2.66666666666666666667*eta*eta*eta*eta-1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta+1.33333333333333333333*eta
        eta4 = 0.666666666666666666667*eta*eta*eta*eta+0.666666666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta-0.166666666666666666667*eta

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = 0.666666666666666666667*xi*xi*xi*xi-0.666666666666666666667*xi*xi*xi-0.166666666666666666667*xi*xi+0.166666666666666666667*xi
        xi1 = -2.66666666666666666667*xi*xi*xi*xi+1.33333333333333333333*xi*xi*xi+2.666666666666667*xi*xi-1.33333333333333333333*xi
        xi2 = 4.0*xi*xi*xi*xi-5.0*xi*xi+1.0
        xi3 = -2.66666666666666666667*xi*xi*xi*xi-1.33333333333333333333*xi*xi*xi+2.66666666666666666667*xi*xi+1.33333333333333333333*xi
        xi4 = 0.666666666666666666667*xi*xi*xi*xi+0.6666666666666666666667*xi*xi*xi-0.166666666666666666667*xi*xi-0.1666666666666666666667*xi
        eta0 = 2.666666666666666666667*eta*eta*eta-2*eta*eta-0.333333333333333333333*eta+0.1666666666666666666667
        eta1 = -10.6666666666666666667*eta*eta*eta+4*eta*eta+5.33333333333333333333*eta-1.33333333333333333333
        eta2 = 16.0*eta*eta*eta-10.0*eta
        eta3 = -10.66666666666666666667*eta*eta*eta-4*eta*eta+5.33333333333333333333*eta+1.33333333333333333333
        eta4 = 2.66666666666666666667*eta*eta*eta+2*eta*eta-0.333333333333333333333*eta-0.166666666666666666667

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = 8*xi*xi-4*xi-0.333333333333333333333
        xi1 = -32*xi*xi+8*xi+5.33333333333333333333
        xi2 = 48.0*xi*xi-10.0
        xi3 = -32*xi*xi-8*xi+5.33333333333333333333
        xi4 = 8*xi*xi+4*xi-0.333333333333333333333
        eta0 = 0.6666666666666666666667*eta*eta*eta*eta-0.666666666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta+0.166666666666666666667*eta
        eta1 = -2.66666666666666666667*eta*eta*eta*eta+1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta-1.33333333333333333333*eta
        eta2 = 4.0*eta*eta*eta*eta-5.0*eta*eta+1.0
        eta3 = -2.66666666666666666667*eta*eta*eta*eta-1.33333333333333333333*eta*eta*eta+2.66666666666666666667*eta*eta+1.33333333333333333333*eta
        eta4 = 0.666666666666666666667*eta*eta*eta*eta+0.666666666666666666667*eta*eta*eta-0.166666666666666666667*eta*eta-0.166666666666666666667*eta

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = 2.666666666666666666667*xi*xi*xi-2*xi*xi-0.333333333333333333333*xi+0.166666666666666666667
        xi1 = -10.6666666666666666667*xi*xi*xi+4*xi*xi+5.33333333333333333333*xi-1.33333333333333333333
        xi2 = 16.0*xi*xi*xi-10.0*xi
        xi3 = -10.6666666666666666667*xi*xi*xi-4*xi*xi+5.33333333333333333333*xi+1.33333333333333333333
        xi4 = 2.66666666666666666667*xi*xi*xi+2*xi*xi-0.333333333333333333333*xi-0.166666666666666666667
        eta0 = 2.66666666666666666667*eta*eta*eta-2*eta*eta-0.333333333333333333333*eta+0.166666666666666666667
        eta1 = -10.6666666666666666667*eta*eta*eta+4*eta*eta+5.33333333333333333333*eta-1.33333333333333333333
        eta2 = 16.0*eta*eta*eta-10.0*eta
        eta3 = -10.6666666666666666667*eta*eta*eta-4*eta*eta+5.33333333333333333333*eta+1.33333333333333333333
        eta4 = 2.66666666666666666667*eta*eta*eta+2*eta*eta-0.333333333333333333333*eta-0.166666666666666666667

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0 = 0.666666666666666666667*xi*xi*xi*xi-0.666666666666666666667*xi*xi*xi-0.166666666666666666667*xi*xi+0.166666666666666666667*xi
        xi1 = -2.66666666666666666667*xi*xi*xi*xi+1.33333333333333333333*xi*xi*xi+2.66666666666666666667*xi*xi-1.33333333333333333333*xi
        xi2 = 4.0*xi*xi*xi*xi-5.0*xi*xi+1.0
        xi3 = -2.666666666666666666667*xi*xi*xi*xi-1.33333333333333333333*xi*xi*xi+2.66666666666666666667*xi*xi+1.33333333333333333333*xi
        xi4 = 0.6666666666666667*xi*xi*xi*xi+0.6666666666666666666667*xi*xi*xi-0.1666666666666666666667*xi*xi-0.166666666666666666667*xi
        eta0 = 8*eta*eta-4*eta-0.333333333333333333333
        eta1 = -32*eta*eta+8*eta+5.33333333333333333333
        eta2 = 48.0*eta*eta-10.0
        eta3 = -32*eta*eta-8*eta+5.33333333333333333333
        eta4 = 8*eta*eta+4*eta-0.333333333333333333333

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi0*eta1
        values[6, :] =  xi1*eta1
        values[7, :] =  xi2*eta1
        values[8, :] =  xi3*eta1
        values[9, :] =  xi4*eta1
        values[10, :] =  xi0*eta2
        values[11, :] =  xi1*eta2
        values[12, :] =  xi2*eta2
        values[13, :] =  xi3*eta2
        values[14, :] =  xi4*eta2
        values[15, :] =  xi0*eta3
        values[16, :] =  xi1*eta3
        values[17, :] =  xi2*eta3
        values[18, :] =  xi3*eta3
        values[19, :] =  xi4*eta3
        values[20, :] =  xi0*eta4
        values[21, :] =  xi1*eta4
        values[22, :] =  xi2*eta4
        values[23, :] =  xi3*eta4
        values[24, :] =  xi4*eta4

        return values



