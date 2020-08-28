'''gcas autopilot

copied from matlab v2
'''

import math

import numpy as np
from numpy import deg2rad

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController

class GcasAutopilot(Autopilot):
    '''ground collision avoidance autopilot'''

    def __init__(self, init_mode='standby', gain_str='old', stdout=False):

        assert init_mode in ['standby', 'roll', 'pull', 'waiting']

        # config
        self.cfg_eps_phi = deg2rad(5)       # Max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # Max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # Min path angle before completion
        self.cfg_k_prop = 4                 # Proportional control gain
        self.cfg_k_der = 2                  # Derivative control gain
        self.cfg_flight_deck = 1000         # Altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # Min duration of pull up

        self.cfg_nz_des = 5

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = np.zeros(4)
        self.waiting_time = 2

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, init_mode, llc=llc)

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        premode = self.mode

        if self.mode == 'waiting':
            # time-triggered start after two seconds
            if t + 1e-6 >= self.waiting_time:
                self.mode = 'roll'
        elif self.mode == 'standby':
            if not self.is_nose_high_enough(x_f16) and not self.is_above_flight_deck(x_f16):
                self.mode = 'roll'
        elif self.mode == 'roll':
            if self.is_roll_rate_low(x_f16) and self.are_wings_level(x_f16):
                self.mode = 'pull'
                self.pull_start_time = t
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"

            if self.is_nose_high_enough(x_f16) and t >= self.pull_start_time + self.cfg_min_pull_time:
                self.mode = 'standby'

        rv = premode != self.mode

        if rv:
            self.log(f"GCAS transition {premode} -> {self.mode} at time {t}")

        return rv

    def are_wings_level(self, x_f16):
        'are the wings level?'

        phi = x_f16[StateIndex.PHI]

        radsFromWingsLevel = round(phi / (2 * math.pi))

        return abs(phi - (2 * math.pi)  * radsFromWingsLevel) < self.cfg_eps_phi

    def is_roll_rate_low(self, x_f16):
        'is the roll rate low enough to switch to pull?'

        p = x_f16[StateIndex.P]

        return abs(p) < self.cfg_eps_p

    def is_above_flight_deck(self, x_f16):
        'is the aircraft above the flight deck?'

        alt = x_f16[StateIndex.ALT]

        return alt >= self.cfg_flight_deck

    def is_nose_high_enough(self, x_f16):
        'is the nose high enough?'

        theta = x_f16[StateIndex.THETA]
        alpha = x_f16[StateIndex.ALPHA]

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromNoseLevel = round((theta-alpha)/(2 * math.pi))

        # Evaluate boolean
        return (theta-alpha) - 2 * math.pi * radsFromNoseLevel > self.cfg_path_goal

    def get_u_ref(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode == 'standby':
            rv = np.zeros(4)
        elif self.mode == 'waiting':
            rv = self.waiting_cmd
        elif self.mode == 'roll':
            rv = self.roll_wings_level(x_f16)
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"
            rv = self.pull_nose_level()

        return rv

    def pull_nose_level(self):
        'get commands in mode PULL'

        rv = np.zeros(4)

        rv[0] = self.cfg_nz_des

        return rv

    def roll_wings_level(self, x_f16):
        'get commands in mode ROLL'

        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        rv = np.zeros(4)

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromWingsLevel = round(phi / (2 * math.pi))

        # PD Control until phi == pi * radsFromWingsLevel
        ps = -(phi - (2 * math.pi) * radsFromWingsLevel) * self.cfg_k_prop - p * self.cfg_k_der

        # Build commands to roll wings level
        rv[1] = ps

        return rv

if __name__ == '__main__':
    print("The correct high-level script to run is run_GCAS.py or run_GCAS_inverted.py")
