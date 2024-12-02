
# Purpose: Defines the basis functions for a 2D Q6 element.
# Reference: ParMooN -  File: BF_C_Q_Q6_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ6(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q6 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=49)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    
  

        xi0= 0.10125E1*xi*xi*xi*xi*xi*xi-0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi+0.5625*xi*xi*xi+0.5E-1*xi*xi-0.5E-1*xi
        xi1= -0.6075E1*xi*xi*xi*xi*xi*xi+0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi-0.45E1*xi*xi*xi-0.675*xi*xi+0.45*xi
        xi2= 0.151875E2*xi*xi*xi*xi*xi*xi-0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi+0.73125E1*xi*xi*xi+0.675E1*xi*xi-0.225E1*xi
        xi3= -0.2025E2*xi*xi*xi*xi*xi*xi+0.315E2*xi*xi*xi*xi-0.1225E2*xi*xi+1.0
        xi4= 0.151875E2*xi*xi*xi*xi*xi*xi+0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi-0.73125E1*xi*xi*xi+0.675E1*xi*xi+0.225E1*xi
        xi5= -0.6075E1*xi*xi*xi*xi*xi*xi-0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi+0.45E1*xi*xi*xi-0.675*xi*xi-0.45*xi
        xi6= 0.10125E1*xi*xi*xi*xi*xi*xi+0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi-0.5625*xi*xi*xi+0.5E-1*xi*xi+0.5E-1*xi

        eta0= 0.10125E1*eta*eta*eta*eta*eta*eta-0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta+0.5625*eta*eta*eta+0.5E-1*eta*eta-0.5E-1*eta
        eta1= -0.6075E1*eta*eta*eta*eta*eta*eta+0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta-0.45E1*eta*eta*eta-0.675*eta*eta+0.45*eta
        eta2= 0.151875E2*eta*eta*eta*eta*eta*eta-0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta+0.73125E1*eta*eta*eta+0.675E1*eta*eta-0.225E1*eta
        eta3= -0.2025E2*eta*eta*eta*eta*eta*eta+0.315E2*eta*eta*eta*eta-0.1225E2*eta*eta+1.0
        eta4= 0.151875E2*eta*eta*eta*eta*eta*eta+0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta-0.73125E1*eta*eta*eta+0.675E1*eta*eta+0.225E1*eta
        eta5= -0.6075E1*eta*eta*eta*eta*eta*eta-0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta+0.45E1*eta*eta*eta-0.675*eta*eta-0.45*eta
        eta6= 0.10125E1*eta*eta*eta*eta*eta*eta+0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta-0.5625*eta*eta*eta+0.5E-1*eta*eta+0.5E-1*eta


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values



    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= 0.6075E1*xi*xi*xi*xi*xi-0.50625E1*xi*xi*xi*xi-0.225E1*xi*xi*xi+0.16875E1*xi*xi+0.1*xi-0.5E-1
        xi1= -0.3645E2*xi*xi*xi*xi*xi+0.2025E2*xi*xi*xi*xi+0.27E2*xi*xi*xi-0.135E2*xi*xi-0.135E1*xi+0.45
        xi2= 0.91125E2*xi*xi*xi*xi*xi-0.253125E2*xi*xi*xi*xi-0.8775E2*xi*xi*xi+0.219375E2*xi*xi+0.135E2*xi-0.225E1
        xi3= -0.1215E3*xi*xi*xi*xi*xi+0.126E3*xi*xi*xi-0.245E2*xi
        xi4= 0.91125E2*xi*xi*xi*xi*xi+0.253125E2*xi*xi*xi*xi-0.8775E2*xi*xi*xi-0.219375E2*xi*xi+0.135E2*xi+0.225E1
        xi5= -0.3645E2*xi*xi*xi*xi*xi-0.2025E2*xi*xi*xi*xi+0.27E2*xi*xi*xi+0.135E2*xi*xi-0.135E1*xi-0.45
        xi6= 0.6075E1*xi*xi*xi*xi*xi+0.50625E1*xi*xi*xi*xi-0.225E1*xi*xi*xi-0.16875E1*xi*xi+0.1*xi+0.5E-1

        eta0= 0.10125E1*eta*eta*eta*eta*eta*eta-0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta+0.5625*eta*eta*eta+0.5E-1*eta*eta-0.5E-1*eta
        eta1= -0.6075E1*eta*eta*eta*eta*eta*eta+0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta-0.45E1*eta*eta*eta-0.675*eta*eta+0.45*eta
        eta2= 0.151875E2*eta*eta*eta*eta*eta*eta-0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta+0.73125E1*eta*eta*eta+0.675E1*eta*eta-0.225E1*eta
        eta3= -0.2025E2*eta*eta*eta*eta*eta*eta+0.315E2*eta*eta*eta*eta-0.1225E2*eta*eta+1.0
        eta4= 0.151875E2*eta*eta*eta*eta*eta*eta+0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta-0.73125E1*eta*eta*eta+0.675E1*eta*eta+0.225E1*eta
        eta5= -0.6075E1*eta*eta*eta*eta*eta*eta-0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta+0.45E1*eta*eta*eta-0.675*eta*eta-0.45*eta
        eta6= 0.10125E1*eta*eta*eta*eta*eta*eta+0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta-0.5625*eta*eta*eta+0.5E-1*eta*eta+0.5E-1*eta


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= 0.10125E1*xi*xi*xi*xi*xi*xi-0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi+0.5625*xi*xi*xi+0.5E-1*xi*xi-0.5E-1*xi
        xi1= -0.6075E1*xi*xi*xi*xi*xi*xi+0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi-0.45E1*xi*xi*xi-0.675*xi*xi+0.45*xi
        xi2= 0.151875E2*xi*xi*xi*xi*xi*xi-0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi+0.73125E1*xi*xi*xi+0.675E1*xi*xi-0.225E1*xi
        xi3= -0.2025E2*xi*xi*xi*xi*xi*xi+0.315E2*xi*xi*xi*xi-0.1225E2*xi*xi+1.0
        xi4= 0.151875E2*xi*xi*xi*xi*xi*xi+0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi-0.73125E1*xi*xi*xi+0.675E1*xi*xi+0.225E1*xi
        xi5= -0.6075E1*xi*xi*xi*xi*xi*xi-0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi+0.45E1*xi*xi*xi-0.675*xi*xi-0.45*xi
        xi6= 0.10125E1*xi*xi*xi*xi*xi*xi+0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi-0.5625*xi*xi*xi+0.5E-1*xi*xi+0.5E-1*xi

        eta0= 0.6075E1*eta*eta*eta*eta*eta-0.50625E1*eta*eta*eta*eta-0.225E1*eta*eta*eta+0.16875E1*eta*eta+0.1*eta-0.5E-1
        eta1= -0.3645E2*eta*eta*eta*eta*eta+0.2025E2*eta*eta*eta*eta+0.27E2*eta*eta*eta-0.135E2*eta*eta-0.135E1*eta+0.45
        eta2= 0.91125E2*eta*eta*eta*eta*eta-0.253125E2*eta*eta*eta*eta-0.8775E2*eta*eta*eta+0.219375E2*eta*eta+0.135E2*eta-0.225E1
        eta3= -0.1215E3*eta*eta*eta*eta*eta+0.126E3*eta*eta*eta-0.245E2*eta
        eta4= 0.91125E2*eta*eta*eta*eta*eta+0.253125E2*eta*eta*eta*eta-0.8775E2*eta*eta*eta-0.219375E2*eta*eta+0.135E2*eta+0.225E1
        eta5= -0.3645E2*eta*eta*eta*eta*eta-0.2025E2*eta*eta*eta*eta+0.27E2*eta*eta*eta+0.135E2*eta*eta-0.135E1*eta-0.45
        eta6= 0.6075E1*eta*eta*eta*eta*eta+0.50625E1*eta*eta*eta*eta-0.225E1*eta*eta*eta-0.16875E1*eta*eta+0.1*eta+0.5E-1


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values


    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= 0.30375E2*xi*xi*xi*xi-0.2025E2*xi*xi*xi-0.675E1*xi*xi+0.3375E1*xi+0.1
        xi1= -0.18225E3*xi*xi*xi*xi+0.81E2*xi*xi*xi+0.81E2*xi*xi-0.27E2*xi-0.135E1
        xi2= 0.455625E3*xi*xi*xi*xi-0.10125E3*xi*xi*xi-0.26325E3*xi*xi+0.43875E2*xi+0.135E2
        xi3= -0.6075E3*xi*xi*xi*xi+0.378E3*xi*xi-0.245E2
        xi4= 0.455625E3*xi*xi*xi*xi+0.10125E3*xi*xi*xi-0.26325E3*xi*xi-0.43875E2*xi+0.135E2
        xi5= -0.18225E3*xi*xi*xi*xi-0.81E2*xi*xi*xi+0.81E2*xi*xi+0.27E2*xi-0.135E1
        xi6= 0.30375E2*xi*xi*xi*xi+0.2025E2*xi*xi*xi-0.675E1*xi*xi-0.3375E1*xi+0.1

        eta0= 0.10125E1*eta*eta*eta*eta*eta*eta-0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta+0.5625*eta*eta*eta+0.5E-1*eta*eta-0.5E-1*eta
        eta1= -0.6075E1*eta*eta*eta*eta*eta*eta+0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta-0.45E1*eta*eta*eta-0.675*eta*eta+0.45*eta
        eta2= 0.151875E2*eta*eta*eta*eta*eta*eta-0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta+0.73125E1*eta*eta*eta+0.675E1*eta*eta-0.225E1*eta
        eta3= -0.2025E2*eta*eta*eta*eta*eta*eta+0.315E2*eta*eta*eta*eta-0.1225E2*eta*eta+1.0
        eta4= 0.151875E2*eta*eta*eta*eta*eta*eta+0.50625E1*eta*eta*eta*eta*eta-0.219375E2*eta*eta*eta*eta-0.73125E1*eta*eta*eta+0.675E1*eta*eta+0.225E1*eta
        eta5= -0.6075E1*eta*eta*eta*eta*eta*eta-0.405E1*eta*eta*eta*eta*eta+0.675E1*eta*eta*eta*eta+0.45E1*eta*eta*eta-0.675*eta*eta-0.45*eta
        eta6= 0.10125E1*eta*eta*eta*eta*eta*eta+0.10125E1*eta*eta*eta*eta*eta-0.5625*eta*eta*eta*eta-0.5625*eta*eta*eta+0.5E-1*eta*eta+0.5E-1*eta


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values


    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= 0.6075E1*xi*xi*xi*xi*xi-0.50625E1*xi*xi*xi*xi-0.225E1*xi*xi*xi+0.16875E1*xi*xi+0.1*xi-0.5E-1
        xi1= -0.3645E2*xi*xi*xi*xi*xi+0.2025E2*xi*xi*xi*xi+0.27E2*xi*xi*xi-0.135E2*xi*xi-0.135E1*xi+0.45
        xi2= 0.91125E2*xi*xi*xi*xi*xi-0.253125E2*xi*xi*xi*xi-0.8775E2*xi*xi*xi+0.219375E2*xi*xi+0.135E2*xi-0.225E1
        xi3= -0.1215E3*xi*xi*xi*xi*xi+0.126E3*xi*xi*xi-0.245E2*xi
        xi4= 0.91125E2*xi*xi*xi*xi*xi+0.253125E2*xi*xi*xi*xi-0.8775E2*xi*xi*xi-0.219375E2*xi*xi+0.135E2*xi+0.225E1
        xi5= -0.3645E2*xi*xi*xi*xi*xi-0.2025E2*xi*xi*xi*xi+0.27E2*xi*xi*xi+0.135E2*xi*xi-0.135E1*xi-0.45
        xi6= 0.6075E1*xi*xi*xi*xi*xi+0.50625E1*xi*xi*xi*xi-0.225E1*xi*xi*xi-0.16875E1*xi*xi+0.1*xi+0.5E-1

        eta0= 0.6075E1*eta*eta*eta*eta*eta-0.50625E1*eta*eta*eta*eta-0.225E1*eta*eta*eta+0.16875E1*eta*eta+0.1*eta-0.5E-1
        eta1= -0.3645E2*eta*eta*eta*eta*eta+0.2025E2*eta*eta*eta*eta+0.27E2*eta*eta*eta-0.135E2*eta*eta-0.135E1*eta+0.45
        eta2= 0.91125E2*eta*eta*eta*eta*eta-0.253125E2*eta*eta*eta*eta-0.8775E2*eta*eta*eta+0.219375E2*eta*eta+0.135E2*eta-0.225E1
        eta3= -0.1215E3*eta*eta*eta*eta*eta+0.126E3*eta*eta*eta-0.245E2*eta
        eta4= 0.91125E2*eta*eta*eta*eta*eta+0.253125E2*eta*eta*eta*eta-0.8775E2*eta*eta*eta-0.219375E2*eta*eta+0.135E2*eta+0.225E1
        eta5= -0.3645E2*eta*eta*eta*eta*eta-0.2025E2*eta*eta*eta*eta+0.27E2*eta*eta*eta+0.135E2*eta*eta-0.135E1*eta-0.45
        eta6= 0.6075E1*eta*eta*eta*eta*eta+0.50625E1*eta*eta*eta*eta-0.225E1*eta*eta*eta-0.16875E1*eta*eta+0.1*eta+0.5E-1


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= 0.10125E1*xi*xi*xi*xi*xi*xi-0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi+0.5625*xi*xi*xi+0.5E-1*xi*xi-0.5E-1*xi
        xi1= -0.6075E1*xi*xi*xi*xi*xi*xi+0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi-0.45E1*xi*xi*xi-0.675*xi*xi+0.45*xi
        xi2= 0.151875E2*xi*xi*xi*xi*xi*xi-0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi+0.73125E1*xi*xi*xi+0.675E1*xi*xi-0.225E1*xi
        xi3= -0.2025E2*xi*xi*xi*xi*xi*xi+0.315E2*xi*xi*xi*xi-0.1225E2*xi*xi+1.0
        xi4= 0.151875E2*xi*xi*xi*xi*xi*xi+0.50625E1*xi*xi*xi*xi*xi-0.219375E2*xi*xi*xi*xi-0.73125E1*xi*xi*xi+0.675E1*xi*xi+0.225E1*xi
        xi5= -0.6075E1*xi*xi*xi*xi*xi*xi-0.405E1*xi*xi*xi*xi*xi+0.675E1*xi*xi*xi*xi+0.45E1*xi*xi*xi-0.675*xi*xi-0.45*xi
        xi6= 0.10125E1*xi*xi*xi*xi*xi*xi+0.10125E1*xi*xi*xi*xi*xi-0.5625*xi*xi*xi*xi-0.5625*xi*xi*xi+0.5E-1*xi*xi+0.5E-1*xi

        eta0= 0.30375E2*eta*eta*eta*eta-0.2025E2*eta*eta*eta-0.675E1*eta*eta+0.3375E1*eta+0.1
        eta1= -0.18225E3*eta*eta*eta*eta+0.81E2*eta*eta*eta+0.81E2*eta*eta-0.27E2*eta-0.135E1
        eta2= 0.455625E3*eta*eta*eta*eta-0.10125E3*eta*eta*eta-0.26325E3*eta*eta+0.43875E2*eta+0.135E2
        eta3= -0.6075E3*eta*eta*eta*eta+0.378E3*eta*eta-0.245E2
        eta4= 0.455625E3*eta*eta*eta*eta+0.10125E3*eta*eta*eta-0.26325E3*eta*eta-0.43875E2*eta+0.135E2
        eta5= -0.18225E3*eta*eta*eta*eta-0.81E2*eta*eta*eta+0.81E2*eta*eta+0.27E2*eta-0.135E1
        eta6= 0.30375E2*eta*eta*eta*eta+0.2025E2*eta*eta*eta-0.675E1*eta*eta-0.3375E1*eta+0.1


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi0*eta1
        values[8, :] =  xi1*eta1
        values[9, :] =  xi2*eta1
        values[10, :] =  xi3*eta1
        values[11, :] =  xi4*eta1
        values[12, :] =  xi5*eta1
        values[13, :] =  xi6*eta1
        values[14, :] =  xi0*eta2
        values[15, :] =  xi1*eta2
        values[16, :] =  xi2*eta2
        values[17, :] =  xi3*eta2
        values[18, :] =  xi4*eta2
        values[19, :] =  xi5*eta2
        values[20, :] =  xi6*eta2
        values[21, :] =  xi0*eta3
        values[22, :] =  xi1*eta3
        values[23, :] =  xi2*eta3
        values[24, :] =  xi3*eta3
        values[25, :] =  xi4*eta3
        values[26, :] =  xi5*eta3
        values[27, :] =  xi6*eta3
        values[28, :] =  xi0*eta4
        values[29, :] =  xi1*eta4
        values[30, :] =  xi2*eta4
        values[31, :] =  xi3*eta4
        values[32, :] =  xi4*eta4
        values[33, :] =  xi5*eta4
        values[34, :] =  xi6*eta4
        values[35, :] =  xi0*eta5
        values[36, :] =  xi1*eta5
        values[37, :] =  xi2*eta5
        values[38, :] =  xi3*eta5
        values[39, :] =  xi4*eta5
        values[40, :] =  xi5*eta5
        values[41, :] =  xi6*eta5
        values[42, :] =  xi0*eta6
        values[43, :] =  xi1*eta6
        values[44, :] =  xi2*eta6
        values[45, :] =  xi3*eta6
        values[46, :] =  xi4*eta6
        values[47, :] =  xi5*eta6
        values[48, :] =  xi6*eta6

        return values


