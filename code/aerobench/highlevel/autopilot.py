'''
Stanley Bak
Autopilot State-Machine Logic

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
'''

import abc
from math import pi

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.util import Freezable

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, init_mode, llc=None):

        assert isinstance(init_mode, str), 'init_mode should be a string'

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.mode = init_mode # discrete state, this should be overwritten by subclasses

        self.freeze_attrs()

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete mode based on the current aircraft state. Returns True iff the discrete mode
        has changed. It's also suggested to update self.mode to the current mode name.
        '''

        return False

    def is_finished(self):
        '''
        returns True if the simulation should stop (for example, after maneuver completes)

        this is called after advance_discrete_state
        '''

        return False

    @abc.abstractmethod
    def _get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals

        returns a tuple: Nz, ps, Ny_r, throttle
        '''
        
        return

    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals

        returns an np.ndarray: u_ref = [Nz, ps, Ny_r, throttle]
        '''

        Nz, ps, Ny_r, throttle = self._get_u_ref(t, x_f16)

        assert Nz <= self.llc.ctrlLimits.NzMax, "autopilot commanded too low Nz ({})".format(Nz)
        assert Nz >= self.llc.ctrlLimits.NzMin, "autopilot commanded too high Nz ({})".format(Nz)

        u_ref = np.array([Nz, ps, Ny_r, throttle], dtype=float)

        return u_ref

    def get_num_integrators(self):
        'get the number of integrators in the autopilot'

        return 0

    def get_integrator_derivatives(self, t, x_f16, u_ref, x_ctrl, Nz, Ny):
        'get the derivatives of the integrators in the autopilot'

        return []


class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain):
        self.setpoint = setpoint
        self.p_gain = p_gain

        init_mode = 'tracking speed'
        Autopilot.__init__(self, init_mode)

    def _get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif

