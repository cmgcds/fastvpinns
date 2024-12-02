# This class is a wrapper class to all the finite element basis functions that 
# are used in the FE2D code. The 2D basis functions, will have the following
# methods:
#   1. value(xi, eta) - This will return the value of the basis function at the reference point (xi, eta)
#   2. gradx(xi, eta) - This will return the value of the derivative of the basis function with respect to xi
#   3. grady(xi, eta) - This will return the value of the derivative of the basis function with respect to eta
#   4. gradxx(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi
#   5. gradxy(xi, eta) - This will return the value of the second derivative of the basis function with respect to xi and eta


# Author: Thivin Anandh D
# Date:  30/Aug/2023
# History: First version - 30/Aug/2023 - Thivin Anandh D

from abc import ABC, abstractmethod

class BasisFunction2D:
    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions
    
    @abstractmethod
    def value(self, xi, eta):
        pass
    
    @abstractmethod
    def gradx(self, xi, eta):
        pass
    
    @abstractmethod
    def grady(self, xi, eta):
        pass
    
    @abstractmethod
    def gradxx(self, xi, eta):
        pass
    
    @abstractmethod
    def gradxy(self, xi, eta):
        pass
    
    @abstractmethod
    def gradyy(self, xi, eta):
        pass


## Mandatory, Import all the basis functions here
from .basis_2d_Q0 import *
from .basis_2d_Q1 import *
from .basis_2d_Q2 import *
from .basis_2d_Q3 import *
from .basis_2d_Q4 import *
from .basis_2d_Q5 import *
from .basis_2d_Q6 import *
from .basis_2d_Q7 import *
from .basis_2d_Q8 import *
from .basis_2d_Q9 import *
from .basis_2d_QN import *
from .basis_2d_QN_Jacobi import *

## Import basis functions for triangular elements
from .basis_2d_P0 import *
from .basis_2d_P1 import *
from .basis_2d_P2 import *
from .basis_2d_P3 import *
from .basis_2d_P4 import *
from .basis_2d_P5 import *
from .basis_2d_P6 import *