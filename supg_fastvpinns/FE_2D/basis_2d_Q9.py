
# Purpose: Defines the basis functions for a 2D Q9 element.
# Reference: ParMooN -  File: BF_C_Q_Q9_2D.h
# Author: Thivin Anandh D
# Date: 17/Jan/2024

import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DQ9(BasisFunction2D):
    """
    This class defines the basis functions for a 2D Q9 element.
    """
    def __init__(self):
        super().__init__(num_shape_functions=100)



    #  base function values
    
    def value(self, xi, eta):
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.
        """
    
 

        xi0= -0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi+0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi-0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi+0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi-0.5340576171875E-3*xi+0.5340576171875E-3
        xi1= 0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi-0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi+0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi-0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi+0.7945469447544643E-2*xi-0.61798095703125E-2
        xi2= -0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi+0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi-0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi+0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi-0.6229248046875E-1*xi+0.3460693359375E-1
        xi3= 0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi-0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi+0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi-0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi+0.40374755859375*xi-0.13458251953125
        xi4= -0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi+0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi-0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi+0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi-0.5450592041015625E1*xi+0.605621337890625
        xi5= 0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi-0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi+0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi-0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi+0.5450592041015625E1*xi+0.605621337890625
        xi6= -0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi+0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi-0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi+0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi-0.40374755859375*xi-0.13458251953125
        xi7= 0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi-0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi+0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi-0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi+0.6229248046875E-1*xi+0.3460693359375E-1
        xi8= -0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi+0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi-0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi+0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi-0.7945469447544643E-2*xi-0.61798095703125E-2
        xi9= 0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi-0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi+0.5340576171875E-3*xi+0.5340576171875E-3

        eta0= -0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta+0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta-0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta+0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta-0.5340576171875E-3*eta+0.5340576171875E-3
        eta1= 0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta-0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta+0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta-0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta+0.7945469447544643E-2*eta-0.61798095703125E-2
        eta2= -0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta+0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta-0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta+0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta-0.6229248046875E-1*eta+0.3460693359375E-1
        eta3= 0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta-0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta+0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta-0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta+0.40374755859375*eta-0.13458251953125
        eta4= -0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta+0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta-0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta+0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta-0.5450592041015625E1*eta+0.605621337890625
        eta5= 0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta-0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta+0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta-0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta+0.5450592041015625E1*eta+0.605621337890625
        eta6= -0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta+0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta-0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta+0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta-0.40374755859375*eta-0.13458251953125
        eta7= 0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta-0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta+0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta-0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta+0.6229248046875E-1*eta+0.3460693359375E-1
        eta8= -0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta+0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta-0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta+0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta-0.7945469447544643E-2*eta-0.61798095703125E-2
        eta9= 0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta-0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta+0.5340576171875E-3*eta+0.5340576171875E-3


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values



    #  values of the derivatives in xi direction
    
    def gradx(self, xi, eta):
        """
        This method returns the gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    


        xi0= -0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi+0.1668167201450893E2*xi*xi*xi*xi*xi*xi*xi+0.1513707275390625E2*xi*xi*xi*xi*xi*xi-0.129746337890625E2*xi*xi*xi*xi*xi-0.3136871337890625E1*xi*xi*xi*xi+0.25094970703125E1*xi*xi*xi+0.1520350864955357*xi*xi-0.1013567243303571*xi-0.5340576171875E-3
        xi1= 0.1689019291469029E3*xi*xi*xi*xi*xi*xi*xi*xi-0.1167717041015625E3*xi*xi*xi*xi*xi*xi*xi-0.1881321899414063E3*xi*xi*xi*xi*xi*xi+0.1254214599609375E3*xi*xi*xi*xi*xi+0.4424990844726563E2*xi*xi*xi*xi-0.275332763671875E2*xi*xi*xi-0.2246343122209821E1*xi*xi+0.11647705078125E1*xi+0.7945469447544643E-2
        xi2= -0.6756077165876116E3*xi*xi*xi*xi*xi*xi*xi*xi+0.3336334402901786E3*xi*xi*xi*xi*xi*xi*xi+0.908224365234375E3*xi*xi*xi*xi*xi*xi-0.43248779296875E3*xi*xi*xi*xi*xi-0.3019405517578125E3*xi*xi*xi*xi+0.13419580078125E3*xi*xi*xi+0.1731476702008929E2*xi*xi-0.6412876674107143E1*xi-0.6229248046875E-1
        xi3= 0.1576418005371094E4*xi*xi*xi*xi*xi*xi*xi*xi-0.46708681640625E3*xi*xi*xi*xi*xi*xi*xi-0.2361383349609375E4*xi*xi*xi*xi*xi*xi+0.67468095703125E3*xi*xi*xi*xi*xi+0.9843101806640625E3*xi*xi*xi*xi-0.26248271484375E3*xi*xi*xi-0.105248583984375E3*xi*xi+0.2338857421875E2*xi+0.40374755859375
        xi4= -0.2364627008056641E4*xi*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E3*xi*xi*xi*xi*xi*xi*xi+0.3723719897460938E4*xi*xi*xi*xi*xi*xi-0.354639990234375E3*xi*xi*xi*xi*xi-0.1724745300292969E4*xi*xi*xi*xi+0.153310693359375E3*xi*xi*xi+0.2435280029296875E3*xi*xi-0.18039111328125E2*xi-0.5450592041015625E1
        xi5= 0.2364627008056641E4*xi*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E3*xi*xi*xi*xi*xi*xi*xi-0.3723719897460938E4*xi*xi*xi*xi*xi*xi-0.354639990234375E3*xi*xi*xi*xi*xi+0.1724745300292969E4*xi*xi*xi*xi+0.153310693359375E3*xi*xi*xi-0.2435280029296875E3*xi*xi-0.18039111328125E2*xi+0.5450592041015625E1
        xi6= -0.1576418005371094E4*xi*xi*xi*xi*xi*xi*xi*xi-0.46708681640625E3*xi*xi*xi*xi*xi*xi*xi+0.2361383349609375E4*xi*xi*xi*xi*xi*xi+0.67468095703125E3*xi*xi*xi*xi*xi-0.9843101806640625E3*xi*xi*xi*xi-0.26248271484375E3*xi*xi*xi+0.105248583984375E3*xi*xi+0.2338857421875E2*xi-0.40374755859375
        xi7= 0.6756077165876116E3*xi*xi*xi*xi*xi*xi*xi*xi+0.3336334402901786E3*xi*xi*xi*xi*xi*xi*xi-0.908224365234375E3*xi*xi*xi*xi*xi*xi-0.43248779296875E3*xi*xi*xi*xi*xi+0.3019405517578125E3*xi*xi*xi*xi+0.13419580078125E3*xi*xi*xi-0.1731476702008929E2*xi*xi-0.6412876674107143E1*xi+0.6229248046875E-1
        xi8= -0.1689019291469029E3*xi*xi*xi*xi*xi*xi*xi*xi-0.1167717041015625E3*xi*xi*xi*xi*xi*xi*xi+0.1881321899414063E3*xi*xi*xi*xi*xi*xi+0.1254214599609375E3*xi*xi*xi*xi*xi-0.4424990844726563E2*xi*xi*xi*xi-0.275332763671875E2*xi*xi*xi+0.2246343122209821E1*xi*xi+0.11647705078125E1*xi-0.7945469447544643E-2
        xi9= 0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi+0.1668167201450893E2*xi*xi*xi*xi*xi*xi*xi-0.1513707275390625E2*xi*xi*xi*xi*xi*xi-0.129746337890625E2*xi*xi*xi*xi*xi+0.3136871337890625E1*xi*xi*xi*xi+0.25094970703125E1*xi*xi*xi-0.1520350864955357*xi*xi-0.1013567243303571*xi+0.5340576171875E-3

        eta0= -0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta+0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta-0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta+0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta-0.5340576171875E-3*eta+0.5340576171875E-3
        eta1= 0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta-0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta+0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta-0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta+0.7945469447544643E-2*eta-0.61798095703125E-2
        eta2= -0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta+0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta-0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta+0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta-0.6229248046875E-1*eta+0.3460693359375E-1
        eta3= 0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta-0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta+0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta-0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta+0.40374755859375*eta-0.13458251953125
        eta4= -0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta+0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta-0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta+0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta-0.5450592041015625E1*eta+0.605621337890625
        eta5= 0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta-0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta+0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta-0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta+0.5450592041015625E1*eta+0.605621337890625
        eta6= -0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta+0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta-0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta+0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta-0.40374755859375*eta-0.13458251953125
        eta7= 0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta-0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta+0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta-0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta+0.6229248046875E-1*eta+0.3460693359375E-1
        eta8= -0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta+0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta-0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta+0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta-0.7945469447544643E-2*eta-0.61798095703125E-2
        eta9= 0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta-0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta+0.5340576171875E-3*eta+0.5340576171875E-3


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values


    #  values of the derivatives in eta direction
    
    def grady(self, xi, eta):
        """
        This method returns the gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0= -0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi+0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi-0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi+0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi-0.5340576171875E-3*xi+0.5340576171875E-3
        xi1= 0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi-0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi+0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi-0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi+0.7945469447544643E-2*xi-0.61798095703125E-2
        xi2= -0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi+0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi-0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi+0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi-0.6229248046875E-1*xi+0.3460693359375E-1
        xi3= 0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi-0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi+0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi-0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi+0.40374755859375*xi-0.13458251953125
        xi4= -0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi+0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi-0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi+0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi-0.5450592041015625E1*xi+0.605621337890625
        xi5= 0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi-0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi+0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi-0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi+0.5450592041015625E1*xi+0.605621337890625
        xi6= -0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi+0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi-0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi+0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi-0.40374755859375*xi-0.13458251953125
        xi7= 0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi-0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi+0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi-0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi+0.6229248046875E-1*xi+0.3460693359375E-1
        xi8= -0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi+0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi-0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi+0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi-0.7945469447544643E-2*xi-0.61798095703125E-2
        xi9= 0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi-0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi+0.5340576171875E-3*xi+0.5340576171875E-3

        eta0= -0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta+0.1668167201450893E2*eta*eta*eta*eta*eta*eta*eta+0.1513707275390625E2*eta*eta*eta*eta*eta*eta-0.129746337890625E2*eta*eta*eta*eta*eta-0.3136871337890625E1*eta*eta*eta*eta+0.25094970703125E1*eta*eta*eta+0.1520350864955357*eta*eta-0.1013567243303571*eta-0.5340576171875E-3
        eta1= 0.1689019291469029E3*eta*eta*eta*eta*eta*eta*eta*eta-0.1167717041015625E3*eta*eta*eta*eta*eta*eta*eta-0.1881321899414063E3*eta*eta*eta*eta*eta*eta+0.1254214599609375E3*eta*eta*eta*eta*eta+0.4424990844726563E2*eta*eta*eta*eta-0.275332763671875E2*eta*eta*eta-0.2246343122209821E1*eta*eta+0.11647705078125E1*eta+0.7945469447544643E-2
        eta2= -0.6756077165876116E3*eta*eta*eta*eta*eta*eta*eta*eta+0.3336334402901786E3*eta*eta*eta*eta*eta*eta*eta+0.908224365234375E3*eta*eta*eta*eta*eta*eta-0.43248779296875E3*eta*eta*eta*eta*eta-0.3019405517578125E3*eta*eta*eta*eta+0.13419580078125E3*eta*eta*eta+0.1731476702008929E2*eta*eta-0.6412876674107143E1*eta-0.6229248046875E-1
        eta3= 0.1576418005371094E4*eta*eta*eta*eta*eta*eta*eta*eta-0.46708681640625E3*eta*eta*eta*eta*eta*eta*eta-0.2361383349609375E4*eta*eta*eta*eta*eta*eta+0.67468095703125E3*eta*eta*eta*eta*eta+0.9843101806640625E3*eta*eta*eta*eta-0.26248271484375E3*eta*eta*eta-0.105248583984375E3*eta*eta+0.2338857421875E2*eta+0.40374755859375
        eta4= -0.2364627008056641E4*eta*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E3*eta*eta*eta*eta*eta*eta*eta+0.3723719897460938E4*eta*eta*eta*eta*eta*eta-0.354639990234375E3*eta*eta*eta*eta*eta-0.1724745300292969E4*eta*eta*eta*eta+0.153310693359375E3*eta*eta*eta+0.2435280029296875E3*eta*eta-0.18039111328125E2*eta-0.5450592041015625E1
        eta5= 0.2364627008056641E4*eta*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E3*eta*eta*eta*eta*eta*eta*eta-0.3723719897460938E4*eta*eta*eta*eta*eta*eta-0.354639990234375E3*eta*eta*eta*eta*eta+0.1724745300292969E4*eta*eta*eta*eta+0.153310693359375E3*eta*eta*eta-0.2435280029296875E3*eta*eta-0.18039111328125E2*eta+0.5450592041015625E1
        eta6= -0.1576418005371094E4*eta*eta*eta*eta*eta*eta*eta*eta-0.46708681640625E3*eta*eta*eta*eta*eta*eta*eta+0.2361383349609375E4*eta*eta*eta*eta*eta*eta+0.67468095703125E3*eta*eta*eta*eta*eta-0.9843101806640625E3*eta*eta*eta*eta-0.26248271484375E3*eta*eta*eta+0.105248583984375E3*eta*eta+0.2338857421875E2*eta-0.40374755859375
        eta7= 0.6756077165876116E3*eta*eta*eta*eta*eta*eta*eta*eta+0.3336334402901786E3*eta*eta*eta*eta*eta*eta*eta-0.908224365234375E3*eta*eta*eta*eta*eta*eta-0.43248779296875E3*eta*eta*eta*eta*eta+0.3019405517578125E3*eta*eta*eta*eta+0.13419580078125E3*eta*eta*eta-0.1731476702008929E2*eta*eta-0.6412876674107143E1*eta+0.6229248046875E-1
        eta8= -0.1689019291469029E3*eta*eta*eta*eta*eta*eta*eta*eta-0.1167717041015625E3*eta*eta*eta*eta*eta*eta*eta+0.1881321899414063E3*eta*eta*eta*eta*eta*eta+0.1254214599609375E3*eta*eta*eta*eta*eta-0.4424990844726563E2*eta*eta*eta*eta-0.275332763671875E2*eta*eta*eta+0.2246343122209821E1*eta*eta+0.11647705078125E1*eta-0.7945469447544643E-2
        eta9= 0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta+0.1668167201450893E2*eta*eta*eta*eta*eta*eta*eta-0.1513707275390625E2*eta*eta*eta*eta*eta*eta-0.129746337890625E2*eta*eta*eta*eta*eta+0.3136871337890625E1*eta*eta*eta*eta+0.25094970703125E1*eta*eta*eta-0.1520350864955357*eta*eta-0.1013567243303571*eta+0.5340576171875E-3


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values


    #  values of the derivatives in xi-xi  direction
    
    def gradxx(self, xi, eta):
        """
        This method returns the double gradient in x-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0= -0.1501350481305804E3*xi*xi*xi*xi*xi*xi*xi+0.1167717041015625E3*xi*xi*xi*xi*xi*xi+0.908224365234375E2*xi*xi*xi*xi*xi-0.648731689453125E2*xi*xi*xi*xi-0.125474853515625E2*xi*xi*xi+0.75284912109375E1*xi*xi+0.3040701729910714*xi-0.1013567243303571
        xi1= 0.1351215433175223E4*xi*xi*xi*xi*xi*xi*xi-0.8174019287109375E3*xi*xi*xi*xi*xi*xi-0.1128793139648438E4*xi*xi*xi*xi*xi+0.6271072998046875E3*xi*xi*xi*xi+0.1769996337890625E3*xi*xi*xi-0.825998291015625E2*xi*xi-0.4492686244419643E1*xi+0.11647705078125E1
        xi2= -0.5404861732700893E4*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E4*xi*xi*xi*xi*xi*xi+0.544934619140625E4*xi*xi*xi*xi*xi-0.216243896484375E4*xi*xi*xi*xi-0.120776220703125E4*xi*xi*xi+0.40258740234375E3*xi*xi+0.3462953404017857E2*xi-0.6412876674107143E1
        xi3= 0.1261134404296875E5*xi*xi*xi*xi*xi*xi*xi-0.326960771484375E4*xi*xi*xi*xi*xi*xi-0.1416830009765625E5*xi*xi*xi*xi*xi+0.337340478515625E4*xi*xi*xi*xi+0.393724072265625E4*xi*xi*xi-0.78744814453125E3*xi*xi-0.21049716796875E3*xi+0.2338857421875E2
        xi4= -0.1891701606445313E5*xi*xi*xi*xi*xi*xi*xi+0.1634803857421875E4*xi*xi*xi*xi*xi*xi+0.2234231938476563E5*xi*xi*xi*xi*xi-0.1773199951171875E4*xi*xi*xi*xi-0.6898981201171875E4*xi*xi*xi+0.459932080078125E3*xi*xi+0.487056005859375E3*xi-0.18039111328125E2
        xi5= 0.1891701606445313E5*xi*xi*xi*xi*xi*xi*xi+0.1634803857421875E4*xi*xi*xi*xi*xi*xi-0.2234231938476563E5*xi*xi*xi*xi*xi-0.1773199951171875E4*xi*xi*xi*xi+0.6898981201171875E4*xi*xi*xi+0.459932080078125E3*xi*xi-0.487056005859375E3*xi-0.18039111328125E2
        xi6= -0.1261134404296875E5*xi*xi*xi*xi*xi*xi*xi-0.326960771484375E4*xi*xi*xi*xi*xi*xi+0.1416830009765625E5*xi*xi*xi*xi*xi+0.337340478515625E4*xi*xi*xi*xi-0.393724072265625E4*xi*xi*xi-0.78744814453125E3*xi*xi+0.21049716796875E3*xi+0.2338857421875E2
        xi7= 0.5404861732700893E4*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E4*xi*xi*xi*xi*xi*xi-0.544934619140625E4*xi*xi*xi*xi*xi-0.216243896484375E4*xi*xi*xi*xi+0.120776220703125E4*xi*xi*xi+0.40258740234375E3*xi*xi-0.3462953404017857E2*xi-0.6412876674107143E1
        xi8= -0.1351215433175223E4*xi*xi*xi*xi*xi*xi*xi-0.8174019287109375E3*xi*xi*xi*xi*xi*xi+0.1128793139648438E4*xi*xi*xi*xi*xi+0.6271072998046875E3*xi*xi*xi*xi-0.1769996337890625E3*xi*xi*xi-0.825998291015625E2*xi*xi+0.4492686244419643E1*xi+0.11647705078125E1
        xi9= 0.1501350481305804E3*xi*xi*xi*xi*xi*xi*xi+0.1167717041015625E3*xi*xi*xi*xi*xi*xi-0.908224365234375E2*xi*xi*xi*xi*xi-0.648731689453125E2*xi*xi*xi*xi+0.125474853515625E2*xi*xi*xi+0.75284912109375E1*xi*xi-0.3040701729910714*xi-0.1013567243303571

        eta0= -0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta+0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta-0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta+0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta-0.5340576171875E-3*eta+0.5340576171875E-3
        eta1= 0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta-0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta+0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta-0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta+0.7945469447544643E-2*eta-0.61798095703125E-2
        eta2= -0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta+0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta-0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta+0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta-0.6229248046875E-1*eta+0.3460693359375E-1
        eta3= 0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta-0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta+0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta-0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta+0.40374755859375*eta-0.13458251953125
        eta4= -0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta+0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta-0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta+0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta-0.5450592041015625E1*eta+0.605621337890625
        eta5= 0.2627363342285156E3*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2919292602539063E2*eta*eta*eta*eta*eta*eta*eta*eta-0.5319599853515625E3*eta*eta*eta*eta*eta*eta*eta-0.591066650390625E2*eta*eta*eta*eta*eta*eta+0.3449490600585938E3*eta*eta*eta*eta*eta+0.3832767333984375E2*eta*eta*eta*eta-0.811760009765625E2*eta*eta*eta-0.90195556640625E1*eta*eta+0.5450592041015625E1*eta+0.605621337890625
        eta6= -0.1751575561523438E3*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.5838585205078125E2*eta*eta*eta*eta*eta*eta*eta*eta+0.337340478515625E3*eta*eta*eta*eta*eta*eta*eta+0.112446826171875E3*eta*eta*eta*eta*eta*eta-0.1968620361328125E3*eta*eta*eta*eta*eta-0.656206787109375E2*eta*eta*eta*eta+0.35082861328125E2*eta*eta*eta+0.11694287109375E2*eta*eta-0.40374755859375*eta-0.13458251953125
        eta7= 0.7506752406529018E2*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.4170418003627232E2*eta*eta*eta*eta*eta*eta*eta*eta-0.129746337890625E3*eta*eta*eta*eta*eta*eta*eta-0.72081298828125E2*eta*eta*eta*eta*eta*eta+0.603881103515625E2*eta*eta*eta*eta*eta+0.335489501953125E2*eta*eta*eta*eta-0.5771589006696429E1*eta*eta*eta-0.3206438337053571E1*eta*eta+0.6229248046875E-1*eta+0.3460693359375E-1
        eta8= -0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta*eta-0.1459646301269531E2*eta*eta*eta*eta*eta*eta*eta*eta+0.2687602713448661E2*eta*eta*eta*eta*eta*eta*eta+0.2090357666015625E2*eta*eta*eta*eta*eta*eta-0.8849981689453125E1*eta*eta*eta*eta*eta-0.6883319091796875E1*eta*eta*eta*eta+0.7487810407366071*eta*eta*eta+0.58238525390625*eta*eta-0.7945469447544643E-2*eta-0.61798095703125E-2
        eta9= 0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta*eta+0.2085209001813616E1*eta*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta*eta-0.216243896484375E1*eta*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta*eta+0.627374267578125*eta*eta*eta*eta-0.5067836216517857E-1*eta*eta*eta-0.5067836216517857E-1*eta*eta+0.5340576171875E-3*eta+0.5340576171875E-3


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values


    #  values of the derivatives in xi-eta direction
    
    def gradxy(self, xi, eta):
        """
        This method returns the  gradxy of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0= -0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi+0.1668167201450893E2*xi*xi*xi*xi*xi*xi*xi+0.1513707275390625E2*xi*xi*xi*xi*xi*xi-0.129746337890625E2*xi*xi*xi*xi*xi-0.3136871337890625E1*xi*xi*xi*xi+0.25094970703125E1*xi*xi*xi+0.1520350864955357*xi*xi-0.1013567243303571*xi-0.5340576171875E-3
        xi1= 0.1689019291469029E3*xi*xi*xi*xi*xi*xi*xi*xi-0.1167717041015625E3*xi*xi*xi*xi*xi*xi*xi-0.1881321899414063E3*xi*xi*xi*xi*xi*xi+0.1254214599609375E3*xi*xi*xi*xi*xi+0.4424990844726563E2*xi*xi*xi*xi-0.275332763671875E2*xi*xi*xi-0.2246343122209821E1*xi*xi+0.11647705078125E1*xi+0.7945469447544643E-2
        xi2= -0.6756077165876116E3*xi*xi*xi*xi*xi*xi*xi*xi+0.3336334402901786E3*xi*xi*xi*xi*xi*xi*xi+0.908224365234375E3*xi*xi*xi*xi*xi*xi-0.43248779296875E3*xi*xi*xi*xi*xi-0.3019405517578125E3*xi*xi*xi*xi+0.13419580078125E3*xi*xi*xi+0.1731476702008929E2*xi*xi-0.6412876674107143E1*xi-0.6229248046875E-1
        xi3= 0.1576418005371094E4*xi*xi*xi*xi*xi*xi*xi*xi-0.46708681640625E3*xi*xi*xi*xi*xi*xi*xi-0.2361383349609375E4*xi*xi*xi*xi*xi*xi+0.67468095703125E3*xi*xi*xi*xi*xi+0.9843101806640625E3*xi*xi*xi*xi-0.26248271484375E3*xi*xi*xi-0.105248583984375E3*xi*xi+0.2338857421875E2*xi+0.40374755859375
        xi4= -0.2364627008056641E4*xi*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E3*xi*xi*xi*xi*xi*xi*xi+0.3723719897460938E4*xi*xi*xi*xi*xi*xi-0.354639990234375E3*xi*xi*xi*xi*xi-0.1724745300292969E4*xi*xi*xi*xi+0.153310693359375E3*xi*xi*xi+0.2435280029296875E3*xi*xi-0.18039111328125E2*xi-0.5450592041015625E1
        xi5= 0.2364627008056641E4*xi*xi*xi*xi*xi*xi*xi*xi+0.233543408203125E3*xi*xi*xi*xi*xi*xi*xi-0.3723719897460938E4*xi*xi*xi*xi*xi*xi-0.354639990234375E3*xi*xi*xi*xi*xi+0.1724745300292969E4*xi*xi*xi*xi+0.153310693359375E3*xi*xi*xi-0.2435280029296875E3*xi*xi-0.18039111328125E2*xi+0.5450592041015625E1
        xi6= -0.1576418005371094E4*xi*xi*xi*xi*xi*xi*xi*xi-0.46708681640625E3*xi*xi*xi*xi*xi*xi*xi+0.2361383349609375E4*xi*xi*xi*xi*xi*xi+0.67468095703125E3*xi*xi*xi*xi*xi-0.9843101806640625E3*xi*xi*xi*xi-0.26248271484375E3*xi*xi*xi+0.105248583984375E3*xi*xi+0.2338857421875E2*xi-0.40374755859375
        xi7= 0.6756077165876116E3*xi*xi*xi*xi*xi*xi*xi*xi+0.3336334402901786E3*xi*xi*xi*xi*xi*xi*xi-0.908224365234375E3*xi*xi*xi*xi*xi*xi-0.43248779296875E3*xi*xi*xi*xi*xi+0.3019405517578125E3*xi*xi*xi*xi+0.13419580078125E3*xi*xi*xi-0.1731476702008929E2*xi*xi-0.6412876674107143E1*xi+0.6229248046875E-1
        xi8= -0.1689019291469029E3*xi*xi*xi*xi*xi*xi*xi*xi-0.1167717041015625E3*xi*xi*xi*xi*xi*xi*xi+0.1881321899414063E3*xi*xi*xi*xi*xi*xi+0.1254214599609375E3*xi*xi*xi*xi*xi-0.4424990844726563E2*xi*xi*xi*xi-0.275332763671875E2*xi*xi*xi+0.2246343122209821E1*xi*xi+0.11647705078125E1*xi-0.7945469447544643E-2
        xi9= 0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi+0.1668167201450893E2*xi*xi*xi*xi*xi*xi*xi-0.1513707275390625E2*xi*xi*xi*xi*xi*xi-0.129746337890625E2*xi*xi*xi*xi*xi+0.3136871337890625E1*xi*xi*xi*xi+0.25094970703125E1*xi*xi*xi-0.1520350864955357*xi*xi-0.1013567243303571*xi+0.5340576171875E-3

        eta0= -0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta+0.1668167201450893E2*eta*eta*eta*eta*eta*eta*eta+0.1513707275390625E2*eta*eta*eta*eta*eta*eta-0.129746337890625E2*eta*eta*eta*eta*eta-0.3136871337890625E1*eta*eta*eta*eta+0.25094970703125E1*eta*eta*eta+0.1520350864955357*eta*eta-0.1013567243303571*eta-0.5340576171875E-3
        eta1= 0.1689019291469029E3*eta*eta*eta*eta*eta*eta*eta*eta-0.1167717041015625E3*eta*eta*eta*eta*eta*eta*eta-0.1881321899414063E3*eta*eta*eta*eta*eta*eta+0.1254214599609375E3*eta*eta*eta*eta*eta+0.4424990844726563E2*eta*eta*eta*eta-0.275332763671875E2*eta*eta*eta-0.2246343122209821E1*eta*eta+0.11647705078125E1*eta+0.7945469447544643E-2
        eta2= -0.6756077165876116E3*eta*eta*eta*eta*eta*eta*eta*eta+0.3336334402901786E3*eta*eta*eta*eta*eta*eta*eta+0.908224365234375E3*eta*eta*eta*eta*eta*eta-0.43248779296875E3*eta*eta*eta*eta*eta-0.3019405517578125E3*eta*eta*eta*eta+0.13419580078125E3*eta*eta*eta+0.1731476702008929E2*eta*eta-0.6412876674107143E1*eta-0.6229248046875E-1
        eta3= 0.1576418005371094E4*eta*eta*eta*eta*eta*eta*eta*eta-0.46708681640625E3*eta*eta*eta*eta*eta*eta*eta-0.2361383349609375E4*eta*eta*eta*eta*eta*eta+0.67468095703125E3*eta*eta*eta*eta*eta+0.9843101806640625E3*eta*eta*eta*eta-0.26248271484375E3*eta*eta*eta-0.105248583984375E3*eta*eta+0.2338857421875E2*eta+0.40374755859375
        eta4= -0.2364627008056641E4*eta*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E3*eta*eta*eta*eta*eta*eta*eta+0.3723719897460938E4*eta*eta*eta*eta*eta*eta-0.354639990234375E3*eta*eta*eta*eta*eta-0.1724745300292969E4*eta*eta*eta*eta+0.153310693359375E3*eta*eta*eta+0.2435280029296875E3*eta*eta-0.18039111328125E2*eta-0.5450592041015625E1
        eta5= 0.2364627008056641E4*eta*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E3*eta*eta*eta*eta*eta*eta*eta-0.3723719897460938E4*eta*eta*eta*eta*eta*eta-0.354639990234375E3*eta*eta*eta*eta*eta+0.1724745300292969E4*eta*eta*eta*eta+0.153310693359375E3*eta*eta*eta-0.2435280029296875E3*eta*eta-0.18039111328125E2*eta+0.5450592041015625E1
        eta6= -0.1576418005371094E4*eta*eta*eta*eta*eta*eta*eta*eta-0.46708681640625E3*eta*eta*eta*eta*eta*eta*eta+0.2361383349609375E4*eta*eta*eta*eta*eta*eta+0.67468095703125E3*eta*eta*eta*eta*eta-0.9843101806640625E3*eta*eta*eta*eta-0.26248271484375E3*eta*eta*eta+0.105248583984375E3*eta*eta+0.2338857421875E2*eta-0.40374755859375
        eta7= 0.6756077165876116E3*eta*eta*eta*eta*eta*eta*eta*eta+0.3336334402901786E3*eta*eta*eta*eta*eta*eta*eta-0.908224365234375E3*eta*eta*eta*eta*eta*eta-0.43248779296875E3*eta*eta*eta*eta*eta+0.3019405517578125E3*eta*eta*eta*eta+0.13419580078125E3*eta*eta*eta-0.1731476702008929E2*eta*eta-0.6412876674107143E1*eta+0.6229248046875E-1
        eta8= -0.1689019291469029E3*eta*eta*eta*eta*eta*eta*eta*eta-0.1167717041015625E3*eta*eta*eta*eta*eta*eta*eta+0.1881321899414063E3*eta*eta*eta*eta*eta*eta+0.1254214599609375E3*eta*eta*eta*eta*eta-0.4424990844726563E2*eta*eta*eta*eta-0.275332763671875E2*eta*eta*eta+0.2246343122209821E1*eta*eta+0.11647705078125E1*eta-0.7945469447544643E-2
        eta9= 0.1876688101632254E2*eta*eta*eta*eta*eta*eta*eta*eta+0.1668167201450893E2*eta*eta*eta*eta*eta*eta*eta-0.1513707275390625E2*eta*eta*eta*eta*eta*eta-0.129746337890625E2*eta*eta*eta*eta*eta+0.3136871337890625E1*eta*eta*eta*eta+0.25094970703125E1*eta*eta*eta-0.1520350864955357*eta*eta-0.1013567243303571*eta+0.5340576171875E-3


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values


    #  values of the derivatives in eta-eta direction
    
    def gradyy(self, xi, eta):
        """
        This method returns the double gradient in y-direction of the basis functions at the given (xi, eta) coordinates.
        """
    

        xi0= -0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi+0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi-0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi+0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi-0.5340576171875E-3*xi+0.5340576171875E-3
        xi1= 0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi-0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi+0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi-0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi+0.7945469447544643E-2*xi-0.61798095703125E-2
        xi2= -0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi+0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi-0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi+0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi-0.6229248046875E-1*xi+0.3460693359375E-1
        xi3= 0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi-0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi+0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi-0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi+0.40374755859375*xi-0.13458251953125
        xi4= -0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi+0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi-0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi+0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi-0.5450592041015625E1*xi+0.605621337890625
        xi5= 0.2627363342285156E3*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2919292602539063E2*xi*xi*xi*xi*xi*xi*xi*xi-0.5319599853515625E3*xi*xi*xi*xi*xi*xi*xi-0.591066650390625E2*xi*xi*xi*xi*xi*xi+0.3449490600585938E3*xi*xi*xi*xi*xi+0.3832767333984375E2*xi*xi*xi*xi-0.811760009765625E2*xi*xi*xi-0.90195556640625E1*xi*xi+0.5450592041015625E1*xi+0.605621337890625
        xi6= -0.1751575561523438E3*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.5838585205078125E2*xi*xi*xi*xi*xi*xi*xi*xi+0.337340478515625E3*xi*xi*xi*xi*xi*xi*xi+0.112446826171875E3*xi*xi*xi*xi*xi*xi-0.1968620361328125E3*xi*xi*xi*xi*xi-0.656206787109375E2*xi*xi*xi*xi+0.35082861328125E2*xi*xi*xi+0.11694287109375E2*xi*xi-0.40374755859375*xi-0.13458251953125
        xi7= 0.7506752406529018E2*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.4170418003627232E2*xi*xi*xi*xi*xi*xi*xi*xi-0.129746337890625E3*xi*xi*xi*xi*xi*xi*xi-0.72081298828125E2*xi*xi*xi*xi*xi*xi+0.603881103515625E2*xi*xi*xi*xi*xi+0.335489501953125E2*xi*xi*xi*xi-0.5771589006696429E1*xi*xi*xi-0.3206438337053571E1*xi*xi+0.6229248046875E-1*xi+0.3460693359375E-1
        xi8= -0.1876688101632254E2*xi*xi*xi*xi*xi*xi*xi*xi*xi-0.1459646301269531E2*xi*xi*xi*xi*xi*xi*xi*xi+0.2687602713448661E2*xi*xi*xi*xi*xi*xi*xi+0.2090357666015625E2*xi*xi*xi*xi*xi*xi-0.8849981689453125E1*xi*xi*xi*xi*xi-0.6883319091796875E1*xi*xi*xi*xi+0.7487810407366071*xi*xi*xi+0.58238525390625*xi*xi-0.7945469447544643E-2*xi-0.61798095703125E-2
        xi9= 0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi*xi+0.2085209001813616E1*xi*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi*xi-0.216243896484375E1*xi*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi*xi+0.627374267578125*xi*xi*xi*xi-0.5067836216517857E-1*xi*xi*xi-0.5067836216517857E-1*xi*xi+0.5340576171875E-3*xi+0.5340576171875E-3

        eta0= -0.1501350481305804E3*eta*eta*eta*eta*eta*eta*eta+0.1167717041015625E3*eta*eta*eta*eta*eta*eta+0.908224365234375E2*eta*eta*eta*eta*eta-0.648731689453125E2*eta*eta*eta*eta-0.125474853515625E2*eta*eta*eta+0.75284912109375E1*eta*eta+0.3040701729910714*eta-0.1013567243303571
        eta1= 0.1351215433175223E4*eta*eta*eta*eta*eta*eta*eta-0.8174019287109375E3*eta*eta*eta*eta*eta*eta-0.1128793139648438E4*eta*eta*eta*eta*eta+0.6271072998046875E3*eta*eta*eta*eta+0.1769996337890625E3*eta*eta*eta-0.825998291015625E2*eta*eta-0.4492686244419643E1*eta+0.11647705078125E1
        eta2= -0.5404861732700893E4*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E4*eta*eta*eta*eta*eta*eta+0.544934619140625E4*eta*eta*eta*eta*eta-0.216243896484375E4*eta*eta*eta*eta-0.120776220703125E4*eta*eta*eta+0.40258740234375E3*eta*eta+0.3462953404017857E2*eta-0.6412876674107143E1
        eta3= 0.1261134404296875E5*eta*eta*eta*eta*eta*eta*eta-0.326960771484375E4*eta*eta*eta*eta*eta*eta-0.1416830009765625E5*eta*eta*eta*eta*eta+0.337340478515625E4*eta*eta*eta*eta+0.393724072265625E4*eta*eta*eta-0.78744814453125E3*eta*eta-0.21049716796875E3*eta+0.2338857421875E2
        eta4= -0.1891701606445313E5*eta*eta*eta*eta*eta*eta*eta+0.1634803857421875E4*eta*eta*eta*eta*eta*eta+0.2234231938476563E5*eta*eta*eta*eta*eta-0.1773199951171875E4*eta*eta*eta*eta-0.6898981201171875E4*eta*eta*eta+0.459932080078125E3*eta*eta+0.487056005859375E3*eta-0.18039111328125E2
        eta5= 0.1891701606445313E5*eta*eta*eta*eta*eta*eta*eta+0.1634803857421875E4*eta*eta*eta*eta*eta*eta-0.2234231938476563E5*eta*eta*eta*eta*eta-0.1773199951171875E4*eta*eta*eta*eta+0.6898981201171875E4*eta*eta*eta+0.459932080078125E3*eta*eta-0.487056005859375E3*eta-0.18039111328125E2
        eta6= -0.1261134404296875E5*eta*eta*eta*eta*eta*eta*eta-0.326960771484375E4*eta*eta*eta*eta*eta*eta+0.1416830009765625E5*eta*eta*eta*eta*eta+0.337340478515625E4*eta*eta*eta*eta-0.393724072265625E4*eta*eta*eta-0.78744814453125E3*eta*eta+0.21049716796875E3*eta+0.2338857421875E2
        eta7= 0.5404861732700893E4*eta*eta*eta*eta*eta*eta*eta+0.233543408203125E4*eta*eta*eta*eta*eta*eta-0.544934619140625E4*eta*eta*eta*eta*eta-0.216243896484375E4*eta*eta*eta*eta+0.120776220703125E4*eta*eta*eta+0.40258740234375E3*eta*eta-0.3462953404017857E2*eta-0.6412876674107143E1
        eta8= -0.1351215433175223E4*eta*eta*eta*eta*eta*eta*eta-0.8174019287109375E3*eta*eta*eta*eta*eta*eta+0.1128793139648438E4*eta*eta*eta*eta*eta+0.6271072998046875E3*eta*eta*eta*eta-0.1769996337890625E3*eta*eta*eta-0.825998291015625E2*eta*eta+0.4492686244419643E1*eta+0.11647705078125E1
        eta9= 0.1501350481305804E3*eta*eta*eta*eta*eta*eta*eta+0.1167717041015625E3*eta*eta*eta*eta*eta*eta-0.908224365234375E2*eta*eta*eta*eta*eta-0.648731689453125E2*eta*eta*eta*eta+0.125474853515625E2*eta*eta*eta+0.75284912109375E1*eta*eta-0.3040701729910714*eta-0.1013567243303571


        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        values[0, :] =  xi0*eta0
        values[1, :] =  xi1*eta0
        values[2, :] =  xi2*eta0
        values[3, :] =  xi3*eta0
        values[4, :] =  xi4*eta0
        values[5, :] =  xi5*eta0
        values[6, :] =  xi6*eta0
        values[7, :] =  xi7*eta0
        values[8, :] =  xi8*eta0
        values[9, :] =  xi9*eta0
        values[10, :] =  xi0*eta1
        values[11, :] =  xi1*eta1
        values[12, :] =  xi2*eta1
        values[13, :] =  xi3*eta1
        values[14, :] =  xi4*eta1
        values[15, :] =  xi5*eta1
        values[16, :] =  xi6*eta1
        values[17, :] =  xi7*eta1
        values[18, :] =  xi8*eta1
        values[19, :] =  xi9*eta1
        values[20, :] =  xi0*eta2
        values[21, :] =  xi1*eta2
        values[22, :] =  xi2*eta2
        values[23, :] =  xi3*eta2
        values[24, :] =  xi4*eta2
        values[25, :] =  xi5*eta2
        values[26, :] =  xi6*eta2
        values[27, :] =  xi7*eta2
        values[28, :] =  xi8*eta2
        values[29, :] =  xi9*eta2
        values[30, :] =  xi0*eta3
        values[31, :] =  xi1*eta3
        values[32, :] =  xi2*eta3
        values[33, :] =  xi3*eta3
        values[34, :] =  xi4*eta3
        values[35, :] =  xi5*eta3
        values[36, :] =  xi6*eta3
        values[37, :] =  xi7*eta3
        values[38, :] =  xi8*eta3
        values[39, :] =  xi9*eta3
        values[40, :] =  xi0*eta4
        values[41, :] =  xi1*eta4
        values[42, :] =  xi2*eta4
        values[43, :] =  xi3*eta4
        values[44, :] =  xi4*eta4
        values[45, :] =  xi5*eta4
        values[46, :] =  xi6*eta4
        values[47, :] =  xi7*eta4
        values[48, :] =  xi8*eta4
        values[49, :] =  xi9*eta4
        values[50, :] =  xi0*eta5
        values[51, :] =  xi1*eta5
        values[52, :] =  xi2*eta5
        values[53, :] =  xi3*eta5
        values[54, :] =  xi4*eta5
        values[55, :] =  xi5*eta5
        values[56, :] =  xi6*eta5
        values[57, :] =  xi7*eta5
        values[58, :] =  xi8*eta5
        values[59, :] =  xi9*eta5
        values[60, :] =  xi0*eta6
        values[61, :] =  xi1*eta6
        values[62, :] =  xi2*eta6
        values[63, :] =  xi3*eta6
        values[64, :] =  xi4*eta6
        values[65, :] =  xi5*eta6
        values[66, :] =  xi6*eta6
        values[67, :] =  xi7*eta6
        values[68, :] =  xi8*eta6
        values[69, :] =  xi9*eta6
        values[70, :] =  xi0*eta7
        values[71, :] =  xi1*eta7
        values[72, :] =  xi2*eta7
        values[73, :] =  xi3*eta7
        values[74, :] =  xi4*eta7
        values[75, :] =  xi5*eta7
        values[76, :] =  xi6*eta7
        values[77, :] =  xi7*eta7
        values[78, :] =  xi8*eta7
        values[79, :] =  xi9*eta7
        values[80, :] =  xi0*eta8
        values[81, :] =  xi1*eta8
        values[82, :] =  xi2*eta8
        values[83, :] =  xi3*eta8
        values[84, :] =  xi4*eta8
        values[85, :] =  xi5*eta8
        values[86, :] =  xi6*eta8
        values[87, :] =  xi7*eta8
        values[88, :] =  xi8*eta8
        values[89, :] =  xi9*eta8
        values[90, :] =  xi0*eta9
        values[91, :] =  xi1*eta9
        values[92, :] =  xi2*eta9
        values[93, :] =  xi3*eta9
        values[94, :] =  xi4*eta9
        values[95, :] =  xi5*eta9
        values[96, :] =  xi6*eta9
        values[97, :] =  xi7*eta9
        values[98, :] =  xi8*eta9
        values[99, :] =  xi9*eta9

        return values



