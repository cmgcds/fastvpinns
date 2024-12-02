
# Purpose: Defines the basis functions for a 2D Q5 element.
# Reference: ParMooN -  File: BF_C_Q_Q5_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ5(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q5 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=36)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    
  

        xi0= -0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi+0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi-0.1171875E-1*xi+0.1171875E-1
        xi1= 0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi-0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi+0.1627604166666667*xi-0.9765625E-1
        xi2= -0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi+0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi-0.29296875E1*xi+0.5859375
        xi3= 0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi-0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi+0.29296875E1*xi+0.5859375
        xi4= -0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi+0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi-0.1627604166666667*xi-0.9765625E-1
        xi5= 0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi-0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi+0.1171875E-1*xi+0.1171875E-1

        eta0= -0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta+0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta-0.1171875E-1*eta+0.1171875E-1
        eta1= 0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta-0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta+0.1627604166666667*eta-0.9765625E-1
        eta2= -0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta+0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta-0.29296875E1*eta+0.5859375
        eta3= 0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta-0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta+0.29296875E1*eta+0.5859375
        eta4= -0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta+0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta-0.1627604166666667*eta-0.9765625E-1
        eta5= 0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta-0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta+0.1171875E-1*eta+0.1171875E-1


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values



    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.4069010416666667E1*xi*xi*xi*xi+0.3255208333333333E1*xi*xi*xi+0.9765625*xi*xi-0.6510416666666667*xi-0.1171875E-1
        xi1= 0.2034505208333333E2*xi*xi*xi*xi-0.9765625E1*xi*xi*xi-0.126953125E2*xi*xi+0.5078125E1*xi+0.1627604166666667
        xi2= -0.4069010416666667E2*xi*xi*xi*xi+0.6510416666666667E1*xi*xi*xi+0.33203125E2*xi*xi-0.4427083333333333E1*xi-0.29296875E1
        xi3= 0.4069010416666667E2*xi*xi*xi*xi+0.6510416666666667E1*xi*xi*xi-0.33203125E2*xi*xi-0.4427083333333333E1*xi+0.29296875E1
        xi4= -0.2034505208333333E2*xi*xi*xi*xi-0.9765625E1*xi*xi*xi+0.126953125E2*xi*xi+0.5078125E1*xi-0.1627604166666667
        xi5= 0.4069010416666667E1*xi*xi*xi*xi+0.3255208333333333E1*xi*xi*xi-0.9765625*xi*xi-0.6510416666666667*xi+0.1171875E-1

        eta0= -0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta+0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta-0.1171875E-1*eta+0.1171875E-1
        eta1= 0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta-0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta+0.1627604166666667*eta-0.9765625E-1
        eta2= -0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta+0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta-0.29296875E1*eta+0.5859375
        eta3= 0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta-0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta+0.29296875E1*eta+0.5859375
        eta4= -0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta+0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta-0.1627604166666667*eta-0.9765625E-1
        eta5= 0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta-0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta+0.1171875E-1*eta+0.1171875E-1


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

  
        xi0= -0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi+0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi-0.1171875E-1*xi+0.1171875E-1
        xi1= 0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi-0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi+0.1627604166666667*xi-0.9765625E-1
        xi2= -0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi+0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi-0.29296875E1*xi+0.5859375
        xi3= 0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi-0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi+0.29296875E1*xi+0.5859375
        xi4= -0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi+0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi-0.1627604166666667*xi-0.9765625E-1
        xi5= 0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi-0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi+0.1171875E-1*xi+0.1171875E-1

        eta0= -0.4069010416666667E1*eta*eta*eta*eta+0.3255208333333333E1*eta*eta*eta+0.9765625*eta*eta-0.6510416666666667*eta-0.1171875E-1
        eta1= 0.2034505208333333E2*eta*eta*eta*eta-0.9765625E1*eta*eta*eta-0.126953125E2*eta*eta+0.5078125E1*eta+0.1627604166666667
        eta2= -0.4069010416666667E2*eta*eta*eta*eta+0.6510416666666667E1*eta*eta*eta+0.33203125E2*eta*eta-0.4427083333333333E1*eta-0.29296875E1
        eta3= 0.4069010416666667E2*eta*eta*eta*eta+0.6510416666666667E1*eta*eta*eta-0.33203125E2*eta*eta-0.4427083333333333E1*eta+0.29296875E1
        eta4= -0.2034505208333333E2*eta*eta*eta*eta-0.9765625E1*eta*eta*eta+0.126953125E2*eta*eta+0.5078125E1*eta-0.1627604166666667
        eta5= 0.4069010416666667E1*eta*eta*eta*eta+0.3255208333333333E1*eta*eta*eta-0.9765625*eta*eta-0.6510416666666667*eta+0.1171875E-1
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values

    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

     
        xi0= -0.1627604166666667E2*xi*xi*xi+0.9765625E1*xi*xi+0.1953125E1*xi-0.6510416666666667
        xi1= 0.8138020833333333E2*xi*xi*xi-0.29296875E2*xi*xi-0.25390625E2*xi+0.5078125E1
        xi2= -0.1627604166666667E3*xi*xi*xi+0.1953125E2*xi*xi+0.6640625E2*xi-0.4427083333333333E1
        xi3= 0.1627604166666667E3*xi*xi*xi+0.1953125E2*xi*xi-0.6640625E2*xi-0.4427083333333333E1
        xi4= -0.8138020833333333E2*xi*xi*xi-0.29296875E2*xi*xi+0.25390625E2*xi+0.5078125E1
        xi5= 0.1627604166666667E2*xi*xi*xi+0.9765625E1*xi*xi-0.1953125E1*xi-0.6510416666666667

        eta0= -0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta+0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta-0.1171875E-1*eta+0.1171875E-1
        eta1= 0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta-0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta+0.1627604166666667*eta-0.9765625E-1
        eta2= -0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta+0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta-0.29296875E1*eta+0.5859375
        eta3= 0.8138020833333333E1*eta*eta*eta*eta*eta+0.1627604166666667E1*eta*eta*eta*eta-0.1106770833333333E2*eta*eta*eta-0.2213541666666667E1*eta*eta+0.29296875E1*eta+0.5859375
        eta4= -0.4069010416666667E1*eta*eta*eta*eta*eta-0.244140625E1*eta*eta*eta*eta+0.4231770833333333E1*eta*eta*eta+0.25390625E1*eta*eta-0.1627604166666667*eta-0.9765625E-1
        eta5= 0.8138020833333333*eta*eta*eta*eta*eta+0.8138020833333333*eta*eta*eta*eta-0.3255208333333333*eta*eta*eta-0.3255208333333333*eta*eta+0.1171875E-1*eta+0.1171875E-1
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values

    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.4069010416666667E1*xi*xi*xi*xi+0.3255208333333333E1*xi*xi*xi+0.9765625*xi*xi-0.6510416666666667*xi-0.1171875E-1
        xi1= 0.2034505208333333E2*xi*xi*xi*xi-0.9765625E1*xi*xi*xi-0.126953125E2*xi*xi+0.5078125E1*xi+0.1627604166666667
        xi2= -0.4069010416666667E2*xi*xi*xi*xi+0.6510416666666667E1*xi*xi*xi+0.33203125E2*xi*xi-0.4427083333333333E1*xi-0.29296875E1
        xi3= 0.4069010416666667E2*xi*xi*xi*xi+0.6510416666666667E1*xi*xi*xi-0.33203125E2*xi*xi-0.4427083333333333E1*xi+0.29296875E1
        xi4= -0.2034505208333333E2*xi*xi*xi*xi-0.9765625E1*xi*xi*xi+0.126953125E2*xi*xi+0.5078125E1*xi-0.1627604166666667
        xi5= 0.4069010416666667E1*xi*xi*xi*xi+0.3255208333333333E1*xi*xi*xi-0.9765625*xi*xi-0.6510416666666667*xi+0.1171875E-1

        eta0= -0.4069010416666667E1*eta*eta*eta*eta+0.3255208333333333E1*eta*eta*eta+0.9765625*eta*eta-0.6510416666666667*eta-0.1171875E-1
        eta1= 0.2034505208333333E2*eta*eta*eta*eta-0.9765625E1*eta*eta*eta-0.126953125E2*eta*eta+0.5078125E1*eta+0.1627604166666667
        eta2= -0.4069010416666667E2*eta*eta*eta*eta+0.6510416666666667E1*eta*eta*eta+0.33203125E2*eta*eta-0.4427083333333333E1*eta-0.29296875E1
        eta3= 0.4069010416666667E2*eta*eta*eta*eta+0.6510416666666667E1*eta*eta*eta-0.33203125E2*eta*eta-0.4427083333333333E1*eta+0.29296875E1
        eta4= -0.2034505208333333E2*eta*eta*eta*eta-0.9765625E1*eta*eta*eta+0.126953125E2*eta*eta+0.5078125E1*eta-0.1627604166666667
        eta5= 0.4069010416666667E1*eta*eta*eta*eta+0.3255208333333333E1*eta*eta*eta-0.9765625*eta*eta-0.6510416666666667*eta+0.1171875E-1
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0= -0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi+0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi-0.1171875E-1*xi+0.1171875E-1
        xi1= 0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi-0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi+0.1627604166666667*xi-0.9765625E-1
        xi2= -0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi+0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi-0.29296875E1*xi+0.5859375
        xi3= 0.8138020833333333E1*xi*xi*xi*xi*xi+0.1627604166666667E1*xi*xi*xi*xi-0.1106770833333333E2*xi*xi*xi-0.2213541666666667E1*xi*xi+0.29296875E1*xi+0.5859375
        xi4= -0.4069010416666667E1*xi*xi*xi*xi*xi-0.244140625E1*xi*xi*xi*xi+0.4231770833333333E1*xi*xi*xi+0.25390625E1*xi*xi-0.1627604166666667*xi-0.9765625E-1
        xi5= 0.8138020833333333*xi*xi*xi*xi*xi+0.8138020833333333*xi*xi*xi*xi-0.3255208333333333*xi*xi*xi-0.3255208333333333*xi*xi+0.1171875E-1*xi+0.1171875E-1

        eta0= -0.1627604166666667E2*eta*eta*eta+0.9765625E1*eta*eta+0.1953125E1*eta-0.6510416666666667
        eta1= 0.8138020833333333E2*eta*eta*eta-0.29296875E2*eta*eta-0.25390625E2*eta+0.5078125E1
        eta2= -0.1627604166666667E3*eta*eta*eta+0.1953125E2*eta*eta+0.6640625E2*eta-0.4427083333333333E1
        eta3= 0.1627604166666667E3*eta*eta*eta+0.1953125E2*eta*eta-0.6640625E2*eta-0.4427083333333333E1
        eta4= -0.8138020833333333E2*eta*eta*eta-0.29296875E2*eta*eta+0.25390625E2*eta+0.5078125E1
        eta5= 0.1627604166666667E2*eta*eta*eta+0.9765625E1*eta*eta-0.1953125E1*eta-0.6510416666666667
    

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi0*eta1
        values[7, :] =  xi1*eta1
        values[8, :] =  xi2*eta1
        values[9, :] =  xi3*eta1
        values[10, :] =  xi4*eta1
        values[11, :] =  xi5*eta1
        values[12, :] =  xi0*eta2
        values[13, :] =  xi1*eta2
        values[14, :] =  xi2*eta2
        values[15, :] =  xi3*eta2
        values[16, :] =  xi4*eta2
        values[17, :] =  xi5*eta2
        values[18, :] =  xi0*eta3
        values[19, :] =  xi1*eta3
        values[20, :] =  xi2*eta3
        values[21, :] =  xi3*eta3
        values[22, :] =  xi4*eta3
        values[23, :] =  xi5*eta3
        values[24, :] =  xi0*eta4
        values[25, :] =  xi1*eta4
        values[26, :] =  xi2*eta4
        values[27, :] =  xi3*eta4
        values[28, :] =  xi4*eta4
        values[29, :] =  xi5*eta4
        values[30, :] =  xi0*eta5
        values[31, :] =  xi1*eta5
        values[32, :] =  xi2*eta5
        values[33, :] =  xi3*eta5
        values[34, :] =  xi4*eta5
        values[35, :] =  xi5*eta5

        return values




