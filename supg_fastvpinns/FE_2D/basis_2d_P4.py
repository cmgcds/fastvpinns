
# Purpose: Defines the basis functions for a 2D P4 element.
# Reference: ParMooN -  File: BF_C_T_P4_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DP4(BasisFunction2D):
    """
    This class defines the basis functions for a 2D P4 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=15)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t5, t7, t9, t11, t13, t15, t17, t18, t19, t20, t21, t22, t23
        t24, t25, t26, t27, t30, t34, t35, t36, t40, t43, t48, t50, t60
        t61, t68, t70, t75

        t3 = xi*xi
        t5 = xi*eta
        t7 = eta*eta
        t9 = t3*xi
        t11 = t3*eta
        t13 = xi*t7
        t15 = t7*eta
        t17 = t3*t3
        t18 = 32.0/3.0*t17
        t19 = t9*eta
        t20 = 128.0/3.0*t19
        t21 = t3*t7
        t22 = 64.0*t21
        t23 = xi*t15
        t24 = 128.0/3.0*t23
        t25 = t7*t7
        t26 = 32.0/3.0*t25
        t27 = 1.0-25.0/3.0*xi-25.0/3.0*eta+70.0/3.0*t3+140.0/3.0*t5+70.0/3.0*t7-80.0/3.0*t9-80.0*t11-80.0*t13-80.0/3.0*t15+t18+t20+t22+t24+t26
        t30 = 208.0/3.0*t5
        t34 = 128.0/3.0*t17
        t35 = 128.0*t19
        t36 = 128.0*t21
        t40 = 28.0*t5
        t43 = 16.0*t13
        t48 = 16.0/3.0*t5
        t50 = 32.0*t11
        t60 = 128.0*t23
        t61 = 128.0/3.0*t25
        t68 = 32.0*t5
        t70 = 32.0*t13
        t75 = 16.0*t11

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  t27
        values[1, :] =  16.0*xi-208.0/3.0*t3-t30+96.0*t9+192.0*t11+96.0*t13-t34-t35-t36-t24
        values[2, :] =  -12.0*xi+76.0*t3+t40-128.0*t9-144.0*t11-t43+64.0*t17+t35+t22
        values[3, :] =  16.0/3.0*xi-112.0/3.0*t3-t48+224.0/3.0*t9+t50-t34-t20
        values[4, :] =  -xi+22.0/3.0*t3-16.0*t9+t18
        values[5, :] =  16.0*eta-t30-208.0/3.0*t7+96.0*t11+192.0*t13+96.0*t15-t20-t36-t60-t61
        values[6, :] =  96.0*t5-224.0*t11-224.0*t13+t35+256.0*t21+t60
        values[7, :] =  -t68+160.0*t11+t70-t35-t36
        values[8, :] =  t48-t50+t20
        values[9, :] =  -12.0*eta+t40+76.0*t7-t75-144.0*t13-128.0*t15+t22+t60+64.0*t25
        values[10, :] =  -t68+t50+160.0*t13-t36-t60
        values[11, :] =  4.0*t5-t75-t43+t22
        values[12, :] =  16.0/3.0*eta-t48-112.0/3.0*t7+t70+224.0/3.0*t15-t24-t61
        values[13, :] =  t48-t70+t24
        values[14, :] =  -eta+22.0/3.0*t7-16.0*t15+t26

        return values


    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t5, t7, t9, t10, t11, t12, t13, t14, t15, t16, t19, t23, t24
        t25, t28, t31, t35, t37, t44, t51, t53, t56, t63

        t3 = xi*xi
        t5 = xi*eta
        t7 = eta*eta
        t9 = t3*xi
        t10 = 128.0/3.0*t9
        t11 = t3*eta
        t12 = 128.0*t11
        t13 = xi*t7
        t14 = 128.0*t13
        t15 = t7*eta
        t16 = 128.0/3.0*t15
        t19 = 208.0/3.0*eta
        t23 = 512.0/3.0*t9
        t24 = 384.0*t11
        t25 = 256.0*t13
        t28 = 28.0*eta
        t31 = 16.0*t7
        t35 = 16.0/3.0*eta
        t37 = 64.0*t5
        t44 = 128.0*t15
        t51 = 32.0*eta
        t53 = 32.0*t7
        t56 = 32.0*t5
        t63 = -t35+t53-t16

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -25.0/3.0+140.0/3.0*xi+140.0/3.0*eta-80.0*t3-160.0*t5-80.0*t7+t10+t12+t14+t16
        values[1, :] =  16.0-416.0/3.0*xi-t19+288.0*t3+384.0*t5+96.0*t7-t23-t24-t25-t16
        values[2, :] =  -12.0+152.0*xi+t28-384.0*t3-288.0*t5-t31+256.0*t9+t24+t14
        values[3, :] =  16.0/3.0-224.0/3.0*xi-t35+224.0*t3+t37-t23-t12
        values[4, :] =  -1.0+44.0/3.0*xi-48.0*t3+t10
        values[5, :] =  -t19+192.0*t5+192.0*t7-t12-t25-t44
        values[6, :] =  96.0*eta-448.0*t5-224.0*t7+t24+512.0*t13+t44
        values[7, :] =  -t51+320.0*t5+t53-t24-t25
        values[8, :] =  t35-t37+t12
        values[9, :] =  t28-t56-144.0*t7+t14+t44
        values[10, :] =  -t51+t37+160.0*t7-t25-t44
        values[11, :] =  4.0*eta-t56-t31+t14
        values[12, :] =  t63
        values[13, :] =  -t63
        values[14, :] =  0.0

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t5, t7, t9, t10, t11, t12, t13, t14, t15, t16, t18, t21, t22
        t24, t26, t28, t29, t30, t35, t36, t43, t45, t48

        t3 = xi*xi
        t5 = xi*eta
        t7 = eta*eta
        t9 = t3*xi
        t10 = 128.0/3.0*t9
        t11 = t3*eta
        t12 = 128.0*t11
        t13 = xi*t7
        t14 = 128.0*t13
        t15 = t7*eta
        t16 = 128.0/3.0*t15
        t18 = 208.0/3.0*xi
        t21 = 128.0*t9
        t22 = 256.0*t11
        t24 = 28.0*xi
        t26 = 32.0*t5
        t28 = 16.0/3.0*xi
        t29 = 32.0*t3
        t30 = -t28+t29-t10
        t35 = 384.0*t13
        t36 = 512.0/3.0*t15
        t43 = 32.0*xi
        t45 = 64.0*t5
        t48 = 16.0*t3

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  -25.0/3.0+140.0/3.0*xi+140.0/3.0*eta-80.0*t3-160.0*t5-80.0*t7+t10+t12+t14+t16
        values[1, :] =  -t18+192.0*t3+192.0*t5-t21-t22-t14
        values[2, :] =  t24-144.0*t3-t26+t21+t12
        values[3, :] =  t30
        values[4, :] =  0.0
        values[5, :] =  16.0-t18-416.0/3.0*eta+96.0*t3+384.0*t5+288.0*t7-t10-t22-t35-t36
        values[6, :] =  96.0*xi-224.0*t3-448.0*t5+t21+512.0*t11+t35
        values[7, :] =  -t43+160.0*t3+t45-t21-t22
        values[8, :] =  -t30
        values[9, :] =  -12.0+t24+152.0*eta-t48-288.0*t5-384.0*t7+t12+t35+256.0*t15
        values[10, :] =  -t43+t29+320.0*t5-t22-t35
        values[11, :] =  4.0*xi-t48-t26+t12
        values[12, :] =  16.0/3.0-t28-224.0/3.0*eta+t45+224.0*t7-t14-t36
        values[13, :] =  t28-t45+t14
        values[14, :] =  -1.0+44.0/3.0*eta-48.0*t7+t16

        return values


    #  values of the derivatives in xi-xi direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t4, t5, t6, t7, t8, t12, t13, t14, t21, t34

        t3 = xi*xi
        t4 = 128.0*t3
        t5 = xi*eta
        t6 = 256.0*t5
        t7 = eta*eta
        t8 = 128.0*t7
        t12 = 512.0*t3
        t13 = 768.0*t5
        t14 = 256.0*t7
        t21 = 64.0*eta
        t34 = -32.0*eta+t8

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  140.0/3.0-160.0*xi-160.0*eta+t4+t6+t8
        values[1, :] =  -416.0/3.0+576.0*xi+384.0*eta-t12-t13-t14
        values[2, :] =  152.0-768.0*xi-288.0*eta+768.0*t3+t13+t8
        values[3, :] =  -224.0/3.0+448.0*xi+t21-t12-t6
        values[4, :] =  44.0/3.0-96.0*xi+t4
        values[5, :] =  192.0*eta-t6-t14
        values[6, :] =  -448.0*eta+t13+512.0*t7
        values[7, :] =  320.0*eta-t13-t14
        values[8, :] =  -t21+t6
        values[9, :] =  t34
        values[10, :] =  t21-t14
        values[11, :] =  t34
        values[12, :] =  0.0
        values[13, :] =  0.0
        values[14, :] =  0.0

        return values


    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t4, t5, t6, t7, t8, t12, t13, t16, t18, t19, t22, t29, t31
        t37

        t3 = xi*xi
        t4 = 128.0*t3
        t5 = xi*eta
        t6 = 256.0*t5
        t7 = eta*eta
        t8 = 128.0*t7
        t12 = 384.0*t3
        t13 = 512.0*t5
        t16 = 32.0*eta
        t18 = 64.0*xi
        t19 = -16.0/3.0+t18-t4
        t22 = 384.0*t7
        t29 = 64.0*eta
        t31 = 32.0*xi
        t37 = -16.0/3.0+t29-t8

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  140.0/3.0-160.0*xi-160.0*eta+t4+t6+t8
        values[1, :] =  -208.0/3.0+384.0*xi+192.0*eta-t12-t13-t8
        values[2, :] =  28.0-288.0*xi-t16+t12+t6
        values[3, :] =  t19
        values[4, :] =  0.0
        values[5, :] =  -208.0/3.0+192.0*xi+384.0*eta-t4-t13-t22
        values[6, :] =  96.0-448.0*xi-448.0*eta+t12+1024.0*t5+t22
        values[7, :] =  -32.0+320.0*xi+t29-t12-t13
        values[8, :] =  -t19
        values[9, :] =  28.0-t31-288.0*eta+t6+t22
        values[10, :] =  -32.0+t18+320.0*eta-t13-t22
        values[11, :] =  4.0-t31-t16+t6
        values[12, :] =  t37
        values[13, :] =  -t37
        values[14, :] =  0.0

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        t3, t4, t5, t6, t7, t8, t11, t14, t17, t18, t23

        t3 = xi*xi
        t4 = 128.0*t3
        t5 = xi*eta
        t6 = 256.0*t5
        t7 = eta*eta
        t8 = 128.0*t7
        t11 = 256.0*t3
        t14 = -32.0*xi+t4
        t17 = 768.0*t5
        t18 = 512.0*t7
        t23 = 64.0*xi

        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  140.0/3.0-160.0*xi-160.0*eta+t4+t6+t8
        values[1, :] =  192.0*xi-t11-t6
        values[2, :] =  t14
        values[3, :] =  0.0
        values[4, :] =  0.0
        values[5, :] =  -416.0/3.0+384.0*xi+576.0*eta-t11-t17-t18
        values[6, :] =  -448.0*xi+512.0*t3+t17
        values[7, :] =  t23-t11
        values[8, :] =  0.0
        values[9, :] =  152.0-288.0*xi-768.0*eta+t4+t17+768.0*t7
        values[10, :] =  320.0*xi-t11-t17
        values[11, :] =  t14
        values[12, :] =  -224.0/3.0+t23+448.0*eta-t6-t18
        values[13, :] =  -t23+t6
        values[14, :] =  44.0/3.0-96.0*eta+t8

        return values


