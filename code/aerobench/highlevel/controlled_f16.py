'''
Stanley Bak
Python Version of F-16 GCAS
ODE derivative code (controlled F16)
'''

from math import sin, cos

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.subf16_model import subf16_model
from aerobench.lowlevel.low_level_controller import LowLevelController

def controlled_f16(x_f16, u_deg, f16_model='morelli', v2_integrators=False):
    'returns the LQR-controlled F-16 state derivatives and more'

    assert isinstance(x_f16, np.ndarray)

    assert f16_model in ['stevens', 'morelli'], 'Unknown F16_model: {}'.format(f16_model)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg, f16_model)  # We can give u_deg reference values from there for manual control

    if v2_integrators:
        # integrators from matlab v2 model
        ps = xd_model[6] * cos(xd_model[1]) + xd_model[8] * sin(xd_model[1])

        Ny_r = Ny + xd_model[8]

    xd = np.zeros((x_f16.shape[0],))
    xd[:len(xd_model)] = xd_model

    # Convert all degree values to radians for output
    u_rad = np.zeros((7,)) # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0] # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    return xd, u_rad, Nz, ps, Ny_r
