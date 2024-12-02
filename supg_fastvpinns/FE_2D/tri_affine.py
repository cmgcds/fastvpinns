# Purpose   : Defines the Tri Affine transformation of the reference element.
# Reference : ParMooN -  File: TriAffine.C
# Author    : Thivin Anandh D
# Date      : 30/Aug/2023

import numpy as np
from .fe_transformation_2d import FETransforamtion2D

class TriAffin(FETransforamtion2D):
    """
    The TriAffin class defines the Tri Affine transformation of the reference element.
    """

    # constructor
    def __init__(self, co_ordinates) -> None:
        """
        The constructor of the TriAffin class.
        """
        self.co_ordinates = co_ordinates
        self.set_cell()
        self.get_jacobian(0,0) # 0,0 is just a dummy value # this sets the jacobian and the inverse of the jacobian
    
    def set_cell(self):
        """
        Set the cell co-ordinates, which will be used to calculate the Jacobian and actual values.
        """
        
        self.x0 = self.co_ordinates[0][0]
        self.x1 = self.co_ordinates[1][0]
        self.x2 = self.co_ordinates[2][0]
        
        # get the y-co-ordinates of the cell
        self.y0 = self.co_ordinates[0][1]
        self.y1 = self.co_ordinates[1][1]
        self.y2 = self.co_ordinates[2][1]
        
        self.xc0 = self.x0
        self.xc1 = self.x1 - self.x0
        self.xc2 = self.x2 - self.x0

        self.yc0 = self.y0
        self.yc1 = self.y1 - self.y0
        self.yc2 = self.y2 - self.y0
        
        self.detjk = self.xc1 * self.yc2 - self.xc2 * self.yc1
        

    def get_original_from_ref(self, xi, eta):
        """
        This method returns the original co-ordinates from the reference co-ordinates.
        """
        x = self.xc0 + self.xc1 * xi + self.xc2 * eta
        y = self.yc0 + self.yc1 * xi + self.yc2 * eta
    
        return np.array([x, y])
    
    def get_jacobian(self, xi, eta):
        """
        This method returns the Jacobian of the transformation.
        """
        self.detjk = self.xc1 * self.yc2 - self.xc2 * self.yc1
        self.rec_detjk = 1 / self.detjk
        
        return abs(self.detjk)


    def get_orig_from_ref_derivative(self, ref_gradx, ref_grady, xi, eta):
        """
        This method returns the derivatives of the original co-ordinates with respect to the reference co-ordinates.
        """

        n_test = ref_gradx.shape[0]

        gradx_orig = np.zeros(ref_gradx.shape, dtype=np.float64)
        grady_orig = np.zeros(ref_grady.shape, dtype=np.float64)

        for i in range(n_test):
            # (yc2*uxiref[i]-yc1*uetaref[i]) * rec_detjk;
            gradx_orig[i] = (self.yc2 * ref_gradx[i] - self.yc1 * ref_grady[i]) * self.rec_detjk
            # (-xc2*uxiref[i]+xc1*uetaref[i]) * rec_detjk;
            grady_orig[i] = (-self.xc2 * ref_gradx[i] + self.xc1 * ref_grady[i]) * self.rec_detjk

        return gradx_orig, grady_orig


    def get_orig_from_ref_second_derivative(self, grad_xx_ref, grad_xy_ref, grad_yy_ref, xi, eta):
        """
        This method returns the second derivatives of the original co-ordinates with respect to the reference co-ordinates.
        """
        # print("Not implemented yet")
        return grad_xx_ref, grad_xy_ref, grad_yy_ref