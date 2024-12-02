
# Purpose: Defines the basis functions for a 2D Q7 element.
# Reference: ParMooN -  File: BF_C_Q_Q7_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ7(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q7 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=64)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    
  

        xi0= -0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi+0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi-0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi+0.244140625E-2*xi-0.244140625E-2
        xi1= 0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi+0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi-0.3349609375E-1*xi+0.2392578125E-1
        xi2= -0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi+0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi-0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi+0.2791341145833333*xi-0.11962890625
        xi3= 0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi+0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi-0.418701171875E1*xi+0.59814453125
        xi4= -0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi-0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi+0.418701171875E1*xi+0.59814453125
        xi5= 0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi-0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi+0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi-0.2791341145833333*xi-0.11962890625
        xi6= -0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi-0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi+0.3349609375E-1*xi+0.2392578125E-1
        xi7= 0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi+0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi-0.244140625E-2*xi-0.244140625E-2

        eta0= -0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta+0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta-0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta+0.244140625E-2*eta-0.244140625E-2
        eta1= 0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta+0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta-0.3349609375E-1*eta+0.2392578125E-1
        eta2= -0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta+0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta-0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta+0.2791341145833333*eta-0.11962890625
        eta3= 0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta+0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta-0.418701171875E1*eta+0.59814453125
        eta4= -0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta-0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta+0.418701171875E1*eta+0.59814453125
        eta5= 0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta-0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta+0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta-0.2791341145833333*eta-0.11962890625
        eta6= -0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta-0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta+0.3349609375E-1*eta+0.2392578125E-1
        eta7= 0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta+0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta-0.244140625E-2*eta-0.244140625E-2


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values



    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.8936013454861111E1*xi*xi*xi*xi*xi*xi+0.7659440104166667E1*xi*xi*xi*xi*xi+0.4559190538194444E1*xi*xi*xi*xi-0.3647352430555556E1*xi*xi*xi-0.4131184895833333*xi*xi+0.2754123263888889*xi+0.244140625E-2
        xi1= 0.6255209418402778E2*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi-0.5379844835069444E2*xi*xi*xi*xi+0.3074197048611111E2*xi*xi*xi+0.5571516927083333E1*xi*xi-0.2653103298611111E1*xi-0.3349609375E-1
        xi2= -0.1876562825520833E3*xi*xi*xi*xi*xi*xi+0.689349609375E2*xi*xi*xi*xi*xi+0.20516357421875E3*xi*xi*xi*xi-0.70341796875E2*xi*xi*xi-0.4351142578125E2*xi*xi+0.124318359375E2*xi+0.2791341145833333
        xi3= 0.3127604709201389E3*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi-0.3784128146701389E3*xi*xi*xi*xi+0.4324717881944444E2*xi*xi*xi+0.1055685221354167E3*xi*xi-0.1005414496527778E2*xi-0.418701171875E1
        xi4= -0.3127604709201389E3*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi+0.3784128146701389E3*xi*xi*xi*xi+0.4324717881944444E2*xi*xi*xi-0.1055685221354167E3*xi*xi-0.1005414496527778E2*xi+0.418701171875E1
        xi5= 0.1876562825520833E3*xi*xi*xi*xi*xi*xi+0.689349609375E2*xi*xi*xi*xi*xi-0.20516357421875E3*xi*xi*xi*xi-0.70341796875E2*xi*xi*xi+0.4351142578125E2*xi*xi+0.124318359375E2*xi-0.2791341145833333
        xi6= -0.6255209418402778E2*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi+0.5379844835069444E2*xi*xi*xi*xi+0.3074197048611111E2*xi*xi*xi-0.5571516927083333E1*xi*xi-0.2653103298611111E1*xi+0.3349609375E-1
        xi7= 0.8936013454861111E1*xi*xi*xi*xi*xi*xi+0.7659440104166667E1*xi*xi*xi*xi*xi-0.4559190538194444E1*xi*xi*xi*xi-0.3647352430555556E1*xi*xi*xi+0.4131184895833333*xi*xi+0.2754123263888889*xi-0.244140625E-2

        eta0= -0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta+0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta-0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta+0.244140625E-2*eta-0.244140625E-2
        eta1= 0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta+0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta-0.3349609375E-1*eta+0.2392578125E-1
        eta2= -0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta+0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta-0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta+0.2791341145833333*eta-0.11962890625
        eta3= 0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta+0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta-0.418701171875E1*eta+0.59814453125
        eta4= -0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta-0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta+0.418701171875E1*eta+0.59814453125
        eta5= 0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta-0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta+0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta-0.2791341145833333*eta-0.11962890625
        eta6= -0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta-0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta+0.3349609375E-1*eta+0.2392578125E-1
        eta7= 0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta+0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta-0.244140625E-2*eta-0.244140625E-2


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi+0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi-0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi+0.244140625E-2*xi-0.244140625E-2
        xi1= 0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi+0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi-0.3349609375E-1*xi+0.2392578125E-1
        xi2= -0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi+0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi-0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi+0.2791341145833333*xi-0.11962890625
        xi3= 0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi+0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi-0.418701171875E1*xi+0.59814453125
        xi4= -0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi-0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi+0.418701171875E1*xi+0.59814453125
        xi5= 0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi-0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi+0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi-0.2791341145833333*xi-0.11962890625
        xi6= -0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi-0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi+0.3349609375E-1*xi+0.2392578125E-1
        xi7= 0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi+0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi-0.244140625E-2*xi-0.244140625E-2

        eta0= -0.8936013454861111E1*eta*eta*eta*eta*eta*eta+0.7659440104166667E1*eta*eta*eta*eta*eta+0.4559190538194444E1*eta*eta*eta*eta-0.3647352430555556E1*eta*eta*eta-0.4131184895833333*eta*eta+0.2754123263888889*eta+0.244140625E-2
        eta1= 0.6255209418402778E2*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta-0.5379844835069444E2*eta*eta*eta*eta+0.3074197048611111E2*eta*eta*eta+0.5571516927083333E1*eta*eta-0.2653103298611111E1*eta-0.3349609375E-1
        eta2= -0.1876562825520833E3*eta*eta*eta*eta*eta*eta+0.689349609375E2*eta*eta*eta*eta*eta+0.20516357421875E3*eta*eta*eta*eta-0.70341796875E2*eta*eta*eta-0.4351142578125E2*eta*eta+0.124318359375E2*eta+0.2791341145833333
        eta3= 0.3127604709201389E3*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta-0.3784128146701389E3*eta*eta*eta*eta+0.4324717881944444E2*eta*eta*eta+0.1055685221354167E3*eta*eta-0.1005414496527778E2*eta-0.418701171875E1
        eta4= -0.3127604709201389E3*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta+0.3784128146701389E3*eta*eta*eta*eta+0.4324717881944444E2*eta*eta*eta-0.1055685221354167E3*eta*eta-0.1005414496527778E2*eta+0.418701171875E1
        eta5= 0.1876562825520833E3*eta*eta*eta*eta*eta*eta+0.689349609375E2*eta*eta*eta*eta*eta-0.20516357421875E3*eta*eta*eta*eta-0.70341796875E2*eta*eta*eta+0.4351142578125E2*eta*eta+0.124318359375E2*eta-0.2791341145833333
        eta6= -0.6255209418402778E2*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta+0.5379844835069444E2*eta*eta*eta*eta+0.3074197048611111E2*eta*eta*eta-0.5571516927083333E1*eta*eta-0.2653103298611111E1*eta+0.3349609375E-1
        eta7= 0.8936013454861111E1*eta*eta*eta*eta*eta*eta+0.7659440104166667E1*eta*eta*eta*eta*eta-0.4559190538194444E1*eta*eta*eta*eta-0.3647352430555556E1*eta*eta*eta+0.4131184895833333*eta*eta+0.2754123263888889*eta-0.244140625E-2


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values


    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.5361608072916667E2*xi*xi*xi*xi*xi+0.3829720052083333E2*xi*xi*xi*xi+0.1823676215277778E2*xi*xi*xi-0.1094205729166667E2*xi*xi-0.8262369791666667*xi+0.2754123263888889
        xi1= 0.3753125651041667E3*xi*xi*xi*xi*xi-0.1914860026041667E3*xi*xi*xi*xi-0.2151937934027778E3*xi*xi*xi+0.9222591145833333E2*xi*xi+0.1114303385416667E2*xi-0.2653103298611111E1
        xi2= -0.11259376953125E4*xi*xi*xi*xi*xi+0.3446748046875E3*xi*xi*xi*xi+0.820654296875E3*xi*xi*xi-0.211025390625E3*xi*xi-0.870228515625E2*xi+0.124318359375E2
        xi3= 0.1876562825520833E4*xi*xi*xi*xi*xi-0.1914860026041667E3*xi*xi*xi*xi-0.1513651258680556E4*xi*xi*xi+0.1297415364583333E3*xi*xi+0.2111370442708333E3*xi-0.1005414496527778E2
        xi4= -0.1876562825520833E4*xi*xi*xi*xi*xi-0.1914860026041667E3*xi*xi*xi*xi+0.1513651258680556E4*xi*xi*xi+0.1297415364583333E3*xi*xi-0.2111370442708333E3*xi-0.1005414496527778E2
        xi5= 0.11259376953125E4*xi*xi*xi*xi*xi+0.3446748046875E3*xi*xi*xi*xi-0.820654296875E3*xi*xi*xi-0.211025390625E3*xi*xi+0.870228515625E2*xi+0.124318359375E2
        xi6= -0.3753125651041667E3*xi*xi*xi*xi*xi-0.1914860026041667E3*xi*xi*xi*xi+0.2151937934027778E3*xi*xi*xi+0.9222591145833333E2*xi*xi-0.1114303385416667E2*xi-0.2653103298611111E1
        xi7= 0.5361608072916667E2*xi*xi*xi*xi*xi+0.3829720052083333E2*xi*xi*xi*xi-0.1823676215277778E2*xi*xi*xi-0.1094205729166667E2*xi*xi+0.8262369791666667*xi+0.2754123263888889

        eta0= -0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta+0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta-0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta+0.244140625E-2*eta-0.244140625E-2
        eta1= 0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta+0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta-0.3349609375E-1*eta+0.2392578125E-1
        eta2= -0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta+0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta-0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta+0.2791341145833333*eta-0.11962890625
        eta3= 0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta-0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta+0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta-0.418701171875E1*eta+0.59814453125
        eta4= -0.4468006727430556E2*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.7568256293402778E2*eta*eta*eta*eta*eta+0.1081179470486111E2*eta*eta*eta*eta-0.3518950737847222E2*eta*eta*eta-0.5027072482638889E1*eta*eta+0.418701171875E1*eta+0.59814453125
        eta5= 0.2680804036458333E2*eta*eta*eta*eta*eta*eta*eta+0.1148916015625E2*eta*eta*eta*eta*eta*eta-0.4103271484375E2*eta*eta*eta*eta*eta-0.1758544921875E2*eta*eta*eta*eta+0.1450380859375E2*eta*eta*eta+0.621591796875E1*eta*eta-0.2791341145833333*eta-0.11962890625
        eta6= -0.8936013454861111E1*eta*eta*eta*eta*eta*eta*eta-0.6382866753472222E1*eta*eta*eta*eta*eta*eta+0.1075968967013889E2*eta*eta*eta*eta*eta+0.7685492621527778E1*eta*eta*eta*eta-0.1857172309027778E1*eta*eta*eta-0.1326551649305556E1*eta*eta+0.3349609375E-1*eta+0.2392578125E-1
        eta7= 0.1276573350694444E1*eta*eta*eta*eta*eta*eta*eta+0.1276573350694444E1*eta*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta*eta-0.9118381076388889*eta*eta*eta*eta+0.1377061631944444*eta*eta*eta+0.1377061631944444*eta*eta-0.244140625E-2*eta-0.244140625E-2


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values


    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.8936013454861111E1*xi*xi*xi*xi*xi*xi+0.7659440104166667E1*xi*xi*xi*xi*xi+0.4559190538194444E1*xi*xi*xi*xi-0.3647352430555556E1*xi*xi*xi-0.4131184895833333*xi*xi+0.2754123263888889*xi+0.244140625E-2
        xi1= 0.6255209418402778E2*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi-0.5379844835069444E2*xi*xi*xi*xi+0.3074197048611111E2*xi*xi*xi+0.5571516927083333E1*xi*xi-0.2653103298611111E1*xi-0.3349609375E-1
        xi2= -0.1876562825520833E3*xi*xi*xi*xi*xi*xi+0.689349609375E2*xi*xi*xi*xi*xi+0.20516357421875E3*xi*xi*xi*xi-0.70341796875E2*xi*xi*xi-0.4351142578125E2*xi*xi+0.124318359375E2*xi+0.2791341145833333
        xi3= 0.3127604709201389E3*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi-0.3784128146701389E3*xi*xi*xi*xi+0.4324717881944444E2*xi*xi*xi+0.1055685221354167E3*xi*xi-0.1005414496527778E2*xi-0.418701171875E1
        xi4= -0.3127604709201389E3*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi+0.3784128146701389E3*xi*xi*xi*xi+0.4324717881944444E2*xi*xi*xi-0.1055685221354167E3*xi*xi-0.1005414496527778E2*xi+0.418701171875E1
        xi5= 0.1876562825520833E3*xi*xi*xi*xi*xi*xi+0.689349609375E2*xi*xi*xi*xi*xi-0.20516357421875E3*xi*xi*xi*xi-0.70341796875E2*xi*xi*xi+0.4351142578125E2*xi*xi+0.124318359375E2*xi-0.2791341145833333
        xi6= -0.6255209418402778E2*xi*xi*xi*xi*xi*xi-0.3829720052083333E2*xi*xi*xi*xi*xi+0.5379844835069444E2*xi*xi*xi*xi+0.3074197048611111E2*xi*xi*xi-0.5571516927083333E1*xi*xi-0.2653103298611111E1*xi+0.3349609375E-1
        xi7= 0.8936013454861111E1*xi*xi*xi*xi*xi*xi+0.7659440104166667E1*xi*xi*xi*xi*xi-0.4559190538194444E1*xi*xi*xi*xi-0.3647352430555556E1*xi*xi*xi+0.4131184895833333*xi*xi+0.2754123263888889*xi-0.244140625E-2

        eta0= -0.8936013454861111E1*eta*eta*eta*eta*eta*eta+0.7659440104166667E1*eta*eta*eta*eta*eta+0.4559190538194444E1*eta*eta*eta*eta-0.3647352430555556E1*eta*eta*eta-0.4131184895833333*eta*eta+0.2754123263888889*eta+0.244140625E-2
        eta1= 0.6255209418402778E2*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta-0.5379844835069444E2*eta*eta*eta*eta+0.3074197048611111E2*eta*eta*eta+0.5571516927083333E1*eta*eta-0.2653103298611111E1*eta-0.3349609375E-1
        eta2= -0.1876562825520833E3*eta*eta*eta*eta*eta*eta+0.689349609375E2*eta*eta*eta*eta*eta+0.20516357421875E3*eta*eta*eta*eta-0.70341796875E2*eta*eta*eta-0.4351142578125E2*eta*eta+0.124318359375E2*eta+0.2791341145833333
        eta3= 0.3127604709201389E3*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta-0.3784128146701389E3*eta*eta*eta*eta+0.4324717881944444E2*eta*eta*eta+0.1055685221354167E3*eta*eta-0.1005414496527778E2*eta-0.418701171875E1
        eta4= -0.3127604709201389E3*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta+0.3784128146701389E3*eta*eta*eta*eta+0.4324717881944444E2*eta*eta*eta-0.1055685221354167E3*eta*eta-0.1005414496527778E2*eta+0.418701171875E1
        eta5= 0.1876562825520833E3*eta*eta*eta*eta*eta*eta+0.689349609375E2*eta*eta*eta*eta*eta-0.20516357421875E3*eta*eta*eta*eta-0.70341796875E2*eta*eta*eta+0.4351142578125E2*eta*eta+0.124318359375E2*eta-0.2791341145833333
        eta6= -0.6255209418402778E2*eta*eta*eta*eta*eta*eta-0.3829720052083333E2*eta*eta*eta*eta*eta+0.5379844835069444E2*eta*eta*eta*eta+0.3074197048611111E2*eta*eta*eta-0.5571516927083333E1*eta*eta-0.2653103298611111E1*eta+0.3349609375E-1
        eta7= 0.8936013454861111E1*eta*eta*eta*eta*eta*eta+0.7659440104166667E1*eta*eta*eta*eta*eta-0.4559190538194444E1*eta*eta*eta*eta-0.3647352430555556E1*eta*eta*eta+0.4131184895833333*eta*eta+0.2754123263888889*eta-0.244140625E-2


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi+0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi-0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi+0.244140625E-2*xi-0.244140625E-2
        xi1= 0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi+0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi-0.3349609375E-1*xi+0.2392578125E-1
        xi2= -0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi+0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi-0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi+0.2791341145833333*xi-0.11962890625
        xi3= 0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi-0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi+0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi-0.418701171875E1*xi+0.59814453125
        xi4= -0.4468006727430556E2*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.7568256293402778E2*xi*xi*xi*xi*xi+0.1081179470486111E2*xi*xi*xi*xi-0.3518950737847222E2*xi*xi*xi-0.5027072482638889E1*xi*xi+0.418701171875E1*xi+0.59814453125
        xi5= 0.2680804036458333E2*xi*xi*xi*xi*xi*xi*xi+0.1148916015625E2*xi*xi*xi*xi*xi*xi-0.4103271484375E2*xi*xi*xi*xi*xi-0.1758544921875E2*xi*xi*xi*xi+0.1450380859375E2*xi*xi*xi+0.621591796875E1*xi*xi-0.2791341145833333*xi-0.11962890625
        xi6= -0.8936013454861111E1*xi*xi*xi*xi*xi*xi*xi-0.6382866753472222E1*xi*xi*xi*xi*xi*xi+0.1075968967013889E2*xi*xi*xi*xi*xi+0.7685492621527778E1*xi*xi*xi*xi-0.1857172309027778E1*xi*xi*xi-0.1326551649305556E1*xi*xi+0.3349609375E-1*xi+0.2392578125E-1
        xi7= 0.1276573350694444E1*xi*xi*xi*xi*xi*xi*xi+0.1276573350694444E1*xi*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi*xi-0.9118381076388889*xi*xi*xi*xi+0.1377061631944444*xi*xi*xi+0.1377061631944444*xi*xi-0.244140625E-2*xi-0.244140625E-2

        eta0= -0.5361608072916667E2*eta*eta*eta*eta*eta+0.3829720052083333E2*eta*eta*eta*eta+0.1823676215277778E2*eta*eta*eta-0.1094205729166667E2*eta*eta-0.8262369791666667*eta+0.2754123263888889
        eta1= 0.3753125651041667E3*eta*eta*eta*eta*eta-0.1914860026041667E3*eta*eta*eta*eta-0.2151937934027778E3*eta*eta*eta+0.9222591145833333E2*eta*eta+0.1114303385416667E2*eta-0.2653103298611111E1
        eta2= -0.11259376953125E4*eta*eta*eta*eta*eta+0.3446748046875E3*eta*eta*eta*eta+0.820654296875E3*eta*eta*eta-0.211025390625E3*eta*eta-0.870228515625E2*eta+0.124318359375E2
        eta3= 0.1876562825520833E4*eta*eta*eta*eta*eta-0.1914860026041667E3*eta*eta*eta*eta-0.1513651258680556E4*eta*eta*eta+0.1297415364583333E3*eta*eta+0.2111370442708333E3*eta-0.1005414496527778E2
        eta4= -0.1876562825520833E4*eta*eta*eta*eta*eta-0.1914860026041667E3*eta*eta*eta*eta+0.1513651258680556E4*eta*eta*eta+0.1297415364583333E3*eta*eta-0.2111370442708333E3*eta-0.1005414496527778E2
        eta5= 0.11259376953125E4*eta*eta*eta*eta*eta+0.3446748046875E3*eta*eta*eta*eta-0.820654296875E3*eta*eta*eta-0.211025390625E3*eta*eta+0.870228515625E2*eta+0.124318359375E2
        eta6= -0.3753125651041667E3*eta*eta*eta*eta*eta-0.1914860026041667E3*eta*eta*eta*eta+0.2151937934027778E3*eta*eta*eta+0.9222591145833333E2*eta*eta-0.1114303385416667E2*eta-0.2653103298611111E1
        eta7= 0.5361608072916667E2*eta*eta*eta*eta*eta+0.3829720052083333E2*eta*eta*eta*eta-0.1823676215277778E2*eta*eta*eta-0.1094205729166667E2*eta*eta+0.8262369791666667*eta+0.2754123263888889


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi0*eta1
        values[9, :] =  xi1*eta1
        values[10, :] =  xi2*eta1
        values[11, :] =  xi3*eta1
        values[12, :] =  xi4*eta1
        values[13, :] =  xi5*eta1
        values[14, :] =  xi6*eta1
        values[15, :] =  xi7*eta1
        values[16, :] =  xi0*eta2
        values[17, :] =  xi1*eta2
        values[18, :] =  xi2*eta2
        values[19, :] =  xi3*eta2
        values[20, :] =  xi4*eta2
        values[21, :] =  xi5*eta2
        values[22, :] =  xi6*eta2
        values[23, :] =  xi7*eta2
        values[24, :] =  xi0*eta3
        values[25, :] =  xi1*eta3
        values[26, :] =  xi2*eta3
        values[27, :] =  xi3*eta3
        values[28, :] =  xi4*eta3
        values[29, :] =  xi5*eta3
        values[30, :] =  xi6*eta3
        values[31, :] =  xi7*eta3
        values[32, :] =  xi0*eta4
        values[33, :] =  xi1*eta4
        values[34, :] =  xi2*eta4
        values[35, :] =  xi3*eta4
        values[36, :] =  xi4*eta4
        values[37, :] =  xi5*eta4
        values[38, :] =  xi6*eta4
        values[39, :] =  xi7*eta4
        values[40, :] =  xi0*eta5
        values[41, :] =  xi1*eta5
        values[42, :] =  xi2*eta5
        values[43, :] =  xi3*eta5
        values[44, :] =  xi4*eta5
        values[45, :] =  xi5*eta5
        values[46, :] =  xi6*eta5
        values[47, :] =  xi7*eta5
        values[48, :] =  xi0*eta6
        values[49, :] =  xi1*eta6
        values[50, :] =  xi2*eta6
        values[51, :] =  xi3*eta6
        values[52, :] =  xi4*eta6
        values[53, :] =  xi5*eta6
        values[54, :] =  xi6*eta6
        values[55, :] =  xi7*eta6
        values[56, :] =  xi0*eta7
        values[57, :] =  xi1*eta7
        values[58, :] =  xi2*eta7
        values[59, :] =  xi3*eta7
        values[60, :] =  xi4*eta7
        values[61, :] =  xi5*eta7
        values[62, :] =  xi6*eta7
        values[63, :] =  xi7*eta7

        return values




