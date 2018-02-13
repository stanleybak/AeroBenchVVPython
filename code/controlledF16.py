'''
Stanley Bak
Python Version of F-16 GCAS
ODE derivative code (controlled F16)
'''

from math import sin, cos
import numpy as np
from numpy import deg2rad

from subf16_model import subf16_model
from autopilot import Autopilot

def controlledF16(t, x_f16, xequil, uequil, K_lqr, F16_model, ctrlLimits, autopilot):
    'returns the LQR-controlled F-16 state derivatives and more'

    assert isinstance(x_f16, np.ndarray)
    assert isinstance(xequil, np.ndarray)
    assert isinstance(uequil, np.ndarray)
    assert isinstance(autopilot, Autopilot)

    # Get Reference Control Vector (commanded Nz, ps, Ny + r, throttle)
    u_ref = autopilot.get_u_ref(t, x_f16) # in g's & rads / sec

    # Calculate perturbation from trim state
    x_delta = x_f16.copy()
    x_delta[:len(xequil)] -= xequil

    ## Implement LQR Feedback Control
    # Reorder states to match controller:
    # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

    # Initialize control vectors
    u_deg = np.zeros((4,)) # throt, ele, ail, rud

    u = np.zeros((7,)) # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    # Calculate control using LQR gains
    u_deg[1:4] = np.dot(-K_lqr, x_ctrl) # Full Control

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += uequil

    ## Limit controls to saturation limits

    # Limit throttle from 0 to 1
    u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)

    # Limit elevator from -25 to 25 deg
    u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)

    # Limit aileron from -21.5 to 21.5 deg
    u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)

    # Limit rudder from -30 to 30 deg
    u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

    ## Generate xd using user - defined method:

    #   Note: Control vector (u) for subF16 is in units of degrees
    assert F16_model == 'stevens' or F16_model == 'morelli', 'Unknown F16_model: {}'.format(F16_model)

    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg, F16_model)

    xd = np.zeros((16,))
    xd[:len(xd_model)] = xd_model

    # Nonlinear (Actual): ps = p * cos(alpha) +     r * sin(alpha)
    ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    # Convert all degree values to radians for output
    u[0] = u_deg[0]

    for i in xrange(1, 4):
        u[i] = deg2rad(u_deg[i])

    u[4:7] = u_ref[0:3]

    # Add tracked error states to xd for integration
    xd[13:16] = [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]

    return xd, u, Nz, ps, Ny_r
