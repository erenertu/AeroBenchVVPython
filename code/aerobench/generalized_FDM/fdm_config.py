'''
Python Version of F-16
ODE derivative code
'''

from math import sin, cos

import numpy as np
from numpy import deg2rad

from sub_models import sub_models

def fdm_config(x_f16, u_deg, fdm_model='morelli_f16', v2_integrators=False):
    
    assert isinstance(x_f16, np.ndarray)
    assert fdm_model in ['stevens_f16', 'morelli_f16'], 'Unknown model name: {}'.format(fdm_model)

    if fdm_model == 'stevens_f16' or fdm_model == 'morelli_f16':
        ############################################################
        # Autopilot and low-level controllers should be added here #
        # autopilot outputs --> low-level controllers inputs       #
        ############################################################
        
        # Note: Control vector (u) for subF16 is in units of degrees
        xd_model = sub_models(x_f16[0:13], u_deg, fdm_model) 


    return xd_model
