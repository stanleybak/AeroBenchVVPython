'''
Stanley Bak
Low-level flight controller
'''

import numpy as np
from util import Freezable

from math import sin, cos

class LowLevelController(Freezable):
    '''low level flight controller

    gain matrices and equilibrium points are computed by running BuildLqrControllers.py
    '''

    def __init__(self, ctrlLimits):
        # Hard coded LQR gain matrix from BuildLqrControllers.py

        # Longitudinal Gains
        K_lqr_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)

        # Lateral Gains
        K_lqr_lat = np.array([[30.511411060051355, -5.705403676148551, -9.310178739319714, \
                                                    -33.97951344944365, -10.652777306717681], \
                              [-22.65901530645282, 1.3193739204719577, -14.2051751789712, \
                                                    6.7374079391328845, -53.726328142239225]], dtype=float)

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_lqr_long
        self.K_lqr[1:, 3:] = K_lqr_lat

        # equilibrium points from BuildLqrControllers.py
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.05666543872074], dtype=float).transpose()
        self.uequil = np.array([0.13946204864060271, -0.7495784725828754, 0.0, 0.0], dtype=float).transpose()

        self.ctrlLimits = ctrlLimits

        self.freeze_attrs()

    def get_u_deg(self, u_ref, f16_state):
        'get the reference commands for the control surfaces'

        # Calculate perturbation from trim state
        x_delta = f16_state.copy()
        x_delta[:len(self.xequil)] -= self.xequil

        ## Implement LQR Feedback Control
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

        # Initialize control vectors
        u_deg = np.zeros((4,)) # throt, ele, ail, rud

        # Calculate control using LQR gains
        u_deg[1:4] = np.dot(-self.K_lqr, x_ctrl) # Full Control

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil

        ## Limit controls to saturation limits
        ctrlLimits = self.ctrlLimits

        # Limit throttle from 0 to 1
        u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)

        # Limit elevator from -25 to 25 deg
        u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)

        # Limit aileron from -21.5 to 21.5 deg
        u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)

        # Limit rudder from -30 to 30 deg
        u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    def get_num_integrators(self):
        'get the number of integrators in the low-level controller'

        return 3

    def get_integrator_derivatives(self, t, x_f16, u_ref, x_ctrl, Nz, Ny):
        'get the derivatives of the integrators in the low-level controller'

        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        Ny_r = Ny + x_ctrl[5]

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]
