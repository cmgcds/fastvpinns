# Purpose   : Defines the Quad Affine transformation of the reference element.
# Reference : ParMooN -  File: QuadAffine.C
# Author    : Thivin Anandh D
# Date      : 30/Aug/2023

import numpy as np
from .fe_transformation_2d import FETransforamtion2D

class QuadAffin(FETransforamtion2D):
    """
    The QuadAffin class defines the Quad Affine transformation of the reference element.
    """

    def __init__(self, co_ordinates) -> None:
        """
        Constructor for the QuadAffin class.

        Parameters:
        - co_ordinates (numpy.ndarray): The coordinates of the reference element.

        Returns:
        None
        """
        self.co_ordinates = co_ordinates
        self.set_cell()
        self.get_jacobian(0,0) # 0,0 is just a dummy value # this sets the jacobian and the inverse of the jacobian
    
    def set_cell(self):
        """
        Set the cell co-ordinates, which will be used to calculate the Jacobian and actual values.

        Parameters:
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
        
        self.xc0 = (self.x1 + self.x3) * 0.5
        self.xc1 = (self.x1 - self.x0) * 0.5
        self.xc2 = (self.x3 - self.x0) * 0.5

        self.yc0 = (self.y1 + self.y3) * 0.5
        self.yc1 = (self.y1 - self.y0) * 0.5
        self.yc2 = (self.y3 - self.y0) * 0.5

    def get_original_from_ref(self, xi, eta):
        """
        Returns the original co-ordinates from the reference co-ordinates.

        Parameters:
        - xi (float): The xi coordinate.
        - eta (float): The eta coordinate.

        Returns:
        numpy.ndarray: The original co-ordinates.
        """
        x = self.xc0 + self.xc1 * xi + self.xc2 * eta
        y = self.yc0 + self.yc1 * xi + self.yc2 * eta
    
        return np.array([x, y])
    
    def get_jacobian(self, xi, eta):
        """
        Returns the Jacobian of the transformation.

        Parameters:
        - xi (float): The xi coordinate.
        - eta (float): The eta coordinate.

        Returns:
        float: The Jacobian of the transformation.
        """
        self.detjk = self.xc1 * self.yc2 - self.xc2 * self.yc1
        self.rec_detjk = 1 / self.detjk
        
        return abs(self.detjk)

    def get_orig_from_ref_derivative(self, ref_gradx, ref_grady, xi, eta):
        """
        Returns the derivatives of the original co-ordinates with respect to the reference co-ordinates.

        Parameters:
        - ref_gradx (numpy.ndarray): The reference gradient in the x-direction.
        - ref_grady (numpy.ndarray): The reference gradient in the y-direction.
        - xi (float): The xi coordinate.
        - eta (float): The eta coordinate.

        Returns:
        tuple: The derivatives of the original co-ordinates with respect to the reference co-ordinates.
        """
        gradx_orig = np.zeros(ref_gradx.shape)
        grady_orig = np.zeros(ref_grady.shape)

        for i in range(ref_gradx.shape[0]):
            ref_gradx[i] = (self.yc2 * ref_gradx[i]   - self.yc1 * ref_grady[i])* self.rec_detjk
            ref_grady[i] = (-self.xc2 * ref_gradx[i]  + self.xc1 * ref_grady[i])* self.rec_detjk

        return gradx_orig, grady_orig

    
    def get_orig_from_ref_second_derivative(self, grad_xx_ref, grad_xy_ref, grad_yy_ref):
        """ 
        Returns the second derivatives (xx, xy, yy) of the original co-ordinates with respect to the reference co-ordinates.

        Parameters:
        - grad_xx_ref (numpy.ndarray): The reference second derivative in the xx-direction.
        - grad_xy_ref (numpy.ndarray): The reference second derivative in the xy-direction.
        - grad_yy_ref (numpy.ndarray): The reference second derivative in the yy-direction.

        Returns:
        tuple: The second derivatives (xx, xy, yy) of the original co-ordinates with respect to the reference co-ordinates.
        """
        GeoData = np.zeros((3, 3))
        Eye = np.identity(3)

        # Populate GeoData (assuming xc1, xc2, yc1, yc2 are defined)
        GeoData[0, 0] = self.xc1 * self.xc1
        GeoData[0, 1] = 2 * self.xc1 * self.yc1
        GeoData[0, 2] = self.yc1 * self.yc1
        GeoData[1, 0] = self.xc1 * self.xc2
        GeoData[1, 1] = self.yc1 * self.xc2 + self.xc1 * self.yc2
        GeoData[1, 2] = self.yc1 * self.yc2
        GeoData[2, 0] = self.xc2 * self.xc2
        GeoData[2, 1] = 2 * self.xc2 * self.yc2
        GeoData[2, 2] = self.yc2 * self.yc2

        # solve the linear system  
        solution = np.linalg.solve(GeoData, Eye)

        # generate empty arrays for the original second derivatives
        grad_xx_orig = np.zeros(grad_xx_ref.shape)
        grad_xy_orig = np.zeros(grad_xy_ref.shape)
        grad_yy_orig = np.zeros(grad_yy_ref.shape)

        for j in range(grad_xx_ref.shape[0]):
            r20 = grad_xx_ref[j]
            r11 = grad_xy_ref[j]
            r02 = grad_yy_ref[j]

            grad_xx_orig[j] = solution[0, 0] * r20 + solution[0, 1] * r11 + solution[0, 2] * r02
            grad_xy_orig[j] = solution[1, 0] * r20 + solution[1, 1] * r11 + solution[1, 2] * r02
            grad_yy_orig[j] = solution[2, 0] * r20 + solution[2, 1] * r11 + solution[2, 2] * r02
        
        return grad_xx_orig, grad_xy_orig, grad_yy_orig