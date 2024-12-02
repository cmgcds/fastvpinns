# Purpose   : Defines the Quad Affine transformation of the reference element.
# Reference : ParMooN -  File: QuadBilineare.C
# Author    : Thivin Anandh D
# Date      : 30/Aug/2023

import numpy as np
from .fe_transformation_2d import FETransforamtion2D

class QuadBilinear(FETransforamtion2D):
    """
    The QuadBilinear class defines the Quad Bilinear transformation of the reference element.

    Attributes:
    - co_ordinates (numpy.ndarray): The coordinates of the reference element.
    
    Methods:
    - set_cell(): Set the cell co-ordinates, which will be used to calculate the Jacobian and actual values.
    - get_original_from_ref(xi, eta): Returns the original co-ordinates from the reference co-ordinates.
    - get_jacobian(xi, eta): Returns the Jacobian of the transformation.
    - get_orig_from_ref_derivative(ref_gradx, ref_grady, xi, eta): Returns the derivatives of the original co-ordinates with respect to the reference co-ordinates.
    - get_orig_from_ref_second_derivative(grad_xx_ref, grad_xy_ref, grad_yy_ref, xi, eta): Returns the second derivatives of the original co-ordinates with respect to the reference co-ordinates.
    """
    def __init__(self, co_ordinates) -> None:
        """
        The constructor of the QuadAffin class.
        """
        self.co_ordinates = co_ordinates
        self.set_cell()

    def set_cell(self):
        """
        Set the cell co-ordinates, which will be used as intermediate values to calculate the Jacobian and actual values.

        Args:
            None
        
        Returns:
            None
        """
        self.x0 = self.co_ordinates[0][0]
        self.x1 = self.co_ordinates[1][0]
        self.x2 = self.co_ordinates[2][0]
        self.x3 = self.co_ordinates[3][0]
        
        # get the y-co-ordinates of the cell
        self.y0 = self.co_ordinates[0][1]
        self.y1 = self.co_ordinates[1][1]
        self.y2 = self.co_ordinates[2][1]
        self.y3 = self.co_ordinates[3][1]
        
        self.xc0 = (self.x0 + self.x1 + self.x2 + self.x3) * 0.25
        self.xc1 = (-self.x0 + self.x1 + self.x2 - self.x3) * 0.25
        self.xc2 = (-self.x0 - self.x1 + self.x2 + self.x3) * 0.25
        self.xc3 = (self.x0 - self.x1 + self.x2 - self.x3) * 0.25

        self.yc0 = (self.y0 + self.y1 + self.y2 + self.y3) * 0.25
        self.yc1 = (-self.y0 + self.y1 + self.y2 - self.y3) * 0.25
        self.yc2 = (-self.y0 - self.y1 + self.y2 + self.y3) * 0.25
        self.yc3 = (self.y0 - self.y1 + self.y2 - self.y3) * 0.25

    def get_original_from_ref(self, xi, eta):
        """
        This method returns the original co-ordinates from the reference co-ordinates.
        
        Args:
            xi (float): The xi coordinate in the reference element.
            eta (float): The eta coordinate in the reference element.
        
        Returns:
            numpy.ndarray: The original co-ordinates [x, y] corresponding to the given reference co-ordinates.
        """
        x = self.xc0 + self.xc1 * xi + self.xc2 * eta + self.xc3 * xi * eta
        y = self.yc0 + self.yc1 * xi + self.yc2 * eta + self.yc3 * xi * eta
    
        return np.array([x, y], dtype=np.float64)
    
    def get_jacobian(self, xi, eta):
        """
        This method returns the Jacobian of the transformation.
        
        Args:
            xi (float): The xi coordinate in the reference element.
            eta (float): The eta coordinate in the reference element.
        
        Returns:
            float: The Jacobian of the transformation at the given reference co-ordinates.
        """
        self.detjk = abs((self.xc1 + self.xc3 * eta) * \
                        (self.yc2 + self.yc3 * xi) - \
                        (self.xc2 + self.xc3 * xi) * (self.yc1 + self.yc3 * eta))
        return self.detjk
    
    def get_orig_from_ref_derivative(self, ref_gradx, ref_grady, xi, eta):
        """
        This method returns the derivatives of the original co-ordinates with respect to the reference co-ordinates.
        
        Args:
            ref_gradx (numpy.ndarray): The gradient of the xi coordinate in the reference element.
            ref_grady (numpy.ndarray): The gradient of the eta coordinate in the reference element.
            xi (float): The xi coordinate in the reference element.
            eta (float): The eta coordinate in the reference element.
        
        Returns:
            numpy.ndarray: The derivatives of the original co-ordinates [x, y] with respect to the reference co-ordinates.
        """
        n_test = ref_gradx.shape[0]
        gradx_orig = np.zeros(ref_gradx.shape, dtype=np.float64)
        grady_orig = np.zeros(ref_grady.shape, dtype=np.float64)

        for j in range(n_test):
            Xi = xi
            Eta = eta
            rec_detjk = 1 / ((self.xc1 + self.xc3 * Eta) * (self.yc2 + self.yc3 * Xi)
                                - (self.xc2 + self.xc3 * Xi) * (self.yc1 + self.yc3 * Eta))
            gradx_orig[j] = ((self.yc2 + self.yc3 * Xi) * ref_gradx[j]
                                - (self.yc1 + self.yc3 * Eta) * ref_grady[j]) * rec_detjk
            grady_orig[j] = (-(self.xc2 + self.xc3 * Xi) * ref_gradx[j]
                                + (self.xc1 + self.xc3 * Eta) * ref_grady[j]) * rec_detjk

        return gradx_orig, grady_orig
    
    def get_orig_from_ref_second_derivative(self, grad_xx_ref, grad_xy_ref, grad_yy_ref, xi, eta):
        """
        This method returns the second derivatives of the original co-ordinates with respect to the reference co-ordinates.
        
        Args:
            grad_xx_ref (numpy.ndarray): The second derivative of the xi coordinate in the reference element.
            grad_xy_ref (numpy.ndarray): The mixed second derivative of the xi and eta coordinates in the reference element.
            grad_yy_ref (numpy.ndarray): The second derivative of the eta coordinate in the reference element.
            xi (float): The xi coordinate in the reference element.
            eta (float): The eta coordinate in the reference element.
        
        Returns:
            numpy.ndarray: The second derivatives of the original co-ordinates [xx, xy, yy] with respect to the reference co-ordinates.
        """
        # print(" Error : Second Derivative not implemented -- Ignore this error, if second derivative is not required ")
        return grad_xx_ref, grad_xy_ref, grad_yy_ref