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

from util import Freezable

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, xequil, uequil, flightLimits, ctrlLimits):

        self.xequil = xequil
        self.uequil = uequil
        self.flightLimits = flightLimits
        self.ctrlLimits = ctrlLimits
        self.state = '<Unknown State>' # this should be overwritten by subclasses

        self.freeze_attrs()

    @abc.abstractmethod
    def advance_discrete_state(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
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

        Nz, ps, Ny_r, throttle = self.get_u_ref(t, x_f16)

        assert Nz <= self.ctrlLimits.NzMax, "autopilot commanded too low Nz ({})".format(Nz)
        assert Nz >= self.ctrlLimits.NzMin, "autopilot commanded too high Nz ({})".format(Nz)

        u_ref = np.array([Nz, ps, Ny_r, throttle], dtype=float)

        return u_ref

    def get_num_integrators(self):
        'get the number of integrators in the autopilot'

        return 0

    def get_integrator_derivatives(self, t, x_f16, u_ref, x_ctrl, Nz, Ny):
        'get the derivatives of the integrators in the autopilot'

        return []


class GcasAutopilot(Autopilot):
    '''The ground-collision avoidance system autopilot logic'''

    STATE_START = 'Standby'
    STATE_ROLL = 'Roll'
    STATE_PULL = 'Pull'
    STATE_DONE = 'Level Flight'

    def __init__(self, xequil, uequil, flightLimits, ctrlLimits):
        Autopilot.__init__(self, xequil, uequil, flightLimits, ctrlLimits)

        self.state = GcasAutopilot.STATE_START

    @abc.abstractmethod
    def advance_discrete_state(self, t, x_f16):
        '''advance the discrete state based on the current aircraft state'''

        rv = False

        # Pull out important variables for ease of use
        phi = x_f16[3]             # Roll angle    (rad)
        p = x_f16[6]               # Roll rate     (rad/sec)
        theta = x_f16[4]           # Pitch angle   (rad)
        alpha = x_f16[1]           # AoA           (rad)

        eps_phi = deg2rad(5)   # Max roll angle magnitude before pulling g's
        eps_p = deg2rad(1)     # Max roll rate magnitude before pulling g's
        path_goal = deg2rad(0) # Final desired path angle
        man_start = 2 # maneuver starts after 2 seconds

        if self.state == GcasAutopilot.STATE_START:
            if t >= man_start:
                self.state = GcasAutopilot.STATE_ROLL
                rv = True

        elif self.state == GcasAutopilot.STATE_ROLL:
            # Determine which angle is "level" (0, 180, 360, 720, etc)
            radsFromWingsLevel = round(phi/pi)

            # Until wings are "level" & roll rate is small
            if abs(phi - pi * radsFromWingsLevel) < eps_phi and abs(p) < eps_p:
                self.state = GcasAutopilot.STATE_PULL
                rv = True

        elif self.state == GcasAutopilot.STATE_PULL:
            radsFromNoseLevel = round((theta - alpha) / (2 * pi))

            if (theta - alpha) - 2 * pi * radsFromNoseLevel > path_goal:
                self.state = GcasAutopilot.STATE_DONE
                rv = True

        return rv

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        # Zero default commands
        Nz = 0
        ps = 0
        Ny_r = 0
        throttle = 0

        # GCAS logic
        # Concept:
        # Roll until wings level (in the shortest direction)
        # When abs(roll rate) < threshold, pull X g's until pitch angle > X deg
        # Choose threshold values:

        Nz_des = min(5, self.ctrlLimits.NzMax) # Desired maneuver g's

        # Pull out important variables for ease of use
        phi = x_f16[3]             # Roll angle    (rad)
        p = x_f16[6]               # Roll rate     (rad/sec)
        q = x_f16[7]               # Pitch rate    (rad/sec)
        theta = x_f16[4]           # Pitch angle   (rad)
        alpha = x_f16[1]           # AoA           (rad)
        # Note: pathAngle = theta - alpha

        if self.state == GcasAutopilot.STATE_START:
            pass # Do nothing
        elif self.state == GcasAutopilot.STATE_ROLL:
            # Determine which angle is "level" (0, 180, 360, 720, etc)
            radsFromWingsLevel = round(phi/pi)

            # PD Control until phi == pi*radsFromWingsLevel
            K_prop = 4
            K_der = K_prop * 0.3

            ps = -(phi - pi * radsFromWingsLevel) * K_prop - p * K_der
        elif self.state == GcasAutopilot.STATE_PULL:
            Nz = Nz_des
        elif self.state == GcasAutopilot.STATE_DONE:
            # steady-level hold
            # Set Proportional-Derivative control gains for roll
            K_prop = 1
            K_der = K_prop*0.3

            # Determine which angle is "level" (0, 180, 360, 720, etc)
            radsFromWingsLevel = round(phi/pi)
            # PD Control on phi using roll rate
            ps = -(phi-pi*radsFromWingsLevel)*K_prop - p*K_der

            # Set Proportional-Derivative control gains for pitch
            K_prop2 = 2
            K_der2 = K_prop*0.3

            # Determine "which" angle is level (0, 360, 720, etc)
            radsFromNoseLevel = round((theta-alpha)/pi)

            # PD Control on theta using Nz
            Nz = -(theta - alpha - pi*radsFromNoseLevel) * K_prop2 - p*K_der2

        # basic speed control
        K_vt = 0.25
        throttle = -K_vt * (x_f16[0] - self.xequil[0])

        return Nz, ps, Ny_r, throttle

class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain, xequil, uequil, flightLimits, ctrlLimits):
        self.setpoint = setpoint
        self.p_gain = p_gain

        Autopilot.__init__(self, xequil, uequil, flightLimits, ctrlLimits)

    @abc.abstractmethod
    def advance_discrete_state(self, t, x_f16):
        '''advance the discrete state based on the current aircraft state'''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif

class FixedAltitudeAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, xequil, uequil, flightLimits, ctrlLimits):
        self.setpoint = setpoint

        Autopilot.__init__(self, xequil, uequil, flightLimits, ctrlLimits)

    @abc.abstractmethod
    def advance_discrete_state(self, t, x_f16):
        '''advance the discrete state based on the current aircraft state'''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        airspeed = x_f16[0]   # Vt            (ft/sec)
        alpha = x_f16[1]      # AoA           (rad)
        theta = x_f16[4]      # Pitch angle   (rad)
        gamma = theta - alpha # Path angle    (rad)
        h = x_f16[11]         # Altitude      (feet)

        # Proportional Control
        k_alt = 0.025
        h_error = self.setpoint - h
        Nz = k_alt * h_error # Allows stacking of cmds

        # (Psuedo) Derivative control using path angle
        k_gamma = 25
        Nz = Nz - k_gamma*gamma

        # try to maintain a fixed airspeed near trim point
        K_vt = 0.25
        airspeed_setpoint = 540
        throttle = -K_vt * (airspeed - self.xequil[0])

        return Nz, 0, 0, throttle
