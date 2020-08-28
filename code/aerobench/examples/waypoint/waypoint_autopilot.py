'''waypoint autopilot

ported from matlab v2
'''

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController

class WaypointAutopilot(Autopilot):
    '''waypoint follower autopilot'''

    def __init__(self, waypoints, gain_str='old', stdout=False):
        'waypoints is a list of 3-tuples'

        self.stdout = stdout
        self.waypoints = waypoints
        self.waypoint_index = 0

        # waypoint config
        self.cfg_slant_range_threshold = 250

        # default control when not waypoint tracking
        self.cfg_u_ol_default = (0, 0, 0, 0.3)

        # control config
        # Gains for speed control
        self.cfg_k_vt = 0.25
        self.cfg_airspeed = 550

        # Gains for altitude tracking
        self.cfg_k_alt = 0.005
        self.cfg_k_h_dot = 0.02

        # Gains for heading tracking
        self.cfg_k_prop_psi = 5
        self.cfg_k_der_psi = 0.5

        # Gains for roll tracking
        self.cfg_k_prop_phi = 0.75
        self.cfg_k_der_phi = 0.5
        self.cfg_max_bank_deg = 65 # maximum bank angle setpoint
        # v2 was 0.5, 0.9

        # Ranges for Nz
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1

        self.done_time = 0.0

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, 'Waypoint 1', llc=llc)

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def get_u_ref(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode != "Done":
            psi_cmd = self.get_waypoint_data(x_f16)[0]

            # Get desired roll angle given desired heading
            phi_cmd = self.get_phi_to_track_heading(x_f16, psi_cmd)
            ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

            nz_cmd = self.track_altitude(x_f16)
            throttle = self.track_airspeed(x_f16)
        else:
           # Waypoint Following complete: fly level.
            throttle = self.track_airspeed(x_f16)
            ps_cmd = self.track_roll_angle(x_f16, 0)
            nz_cmd = self.track_altitude_wings_level(x_f16)

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        throttle = max(min(throttle, 1), 0)

        # Create reference vector
        rv = [nz_cmd, ps_cmd, 0, throttle]

        return rv

    def track_altitude(self, x_f16):
        'get nz to track altitude, taking turning into account'

        h_cmd = self.waypoints[self.waypoint_index][2]

        h = x_f16[StateIndex.ALT]
        phi = x_f16[StateIndex.PHI]

        # Calculate altitude error (positive => below target alt)
        h_error = h_cmd - h
        nz_alt = self.track_altitude_wings_level(x_f16)
        nz_roll = get_nz_for_level_turn_ol(x_f16)

        if h_error > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)

        return nz

    def get_phi_to_track_heading(self, x_f16, psi_cmd):
        'get phi from psi_cmd'

        # PD Control on heading angle using phi_cmd as control

        # Pull out important variables for ease of use
        psi = wrap_to_pi(x_f16[StateIndex.PSI])
        r = x_f16[StateIndex.R]

        # Calculate PD control
        psi_err = wrap_to_pi(psi_cmd - psi)

        phi_cmd = psi_err * self.cfg_k_prop_psi - r * self.cfg_k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)

        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        return phi_cmd

    def track_roll_angle(self, x_f16, phi_cmd):
        'get roll angle command (ps_cmd)'

        # PD control on roll angle using stability roll rate

        # Pull out important variables for ease of use
        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        # Calculate PD control
        ps = (phi_cmd-phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi

        return ps

    def track_airspeed(self, x_f16):
        'get throttle command'

        vt_cmd = self.cfg_airspeed

        # Proportional control on airspeed using throttle
        throttle = self.cfg_k_vt * (vt_cmd - x_f16[StateIndex.VT])

        return throttle

    def track_altitude_wings_level(self, x_f16):
        'get nz to track altitude'

        i = self.waypoint_index if self.waypoint_index < len(self.waypoints) else -1

        h_cmd = self.waypoints[i][2]

        vt = x_f16[StateIndex.VT]
        h = x_f16[StateIndex.ALT]

        # Proportional-Derivative Control
        h_error = h_cmd - h
        gamma = get_path_angle(x_f16)
        h_dot = vt * sin(gamma) # Calculated, not differentiated

        # Calculate Nz command
        nz = self.cfg_k_alt*h_error - self.cfg_k_h_dot*h_dot

        return nz

    def is_finished(self, t, x_f16):
        'is the maneuver done?'

        rv = self.waypoint_index >= len(self.waypoints) and self.done_time + 5.0 < t

        return rv

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        if self.waypoint_index < len(self.waypoints):
            slant_range = self.get_waypoint_data(x_f16)[-1]

            if slant_range < self.cfg_slant_range_threshold:
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.waypoints):
                    self.done_time = t

        premode = self.mode

        if self.waypoint_index >= len(self.waypoints):
            self.mode = 'Done'
        else:
            self.mode = f'Waypoint {self.waypoint_index + 1}'

        rv = premode != self.mode

        if rv:
            self.log(f"Waypoint transition {premode} -> {self.mode} at time {t}")

        return rv

    def get_waypoint_data(self, x_f16):
        '''returns current waypoint data tuple based on the current waypoint:

        (heading, inclination, horiz_range, vert_range, slant_range)

        heading = heading to tgt, equivalent to psi (rad)
        inclination = polar angle to tgt, equivalent to theta (rad)
        horiz_range = horizontal range to tgt (ft)
        vert_range = vertical range to tgt (ft)
        slant_range = total range to tgt (ft)
        '''

        waypoint = self.waypoints[self.waypoint_index]

        e_pos = x_f16[StateIndex.POSE]
        n_pos = x_f16[StateIndex.POSN]
        alt = x_f16[StateIndex.ALT]

        delta = [waypoint[i] - [e_pos, n_pos, alt][i] for i in range(3)]

        _, inclination, slant_range = cart2sph(delta)

        heading = wrap_to_pi(pi/2 - atan2(delta[1], delta[0]))

        horiz_range = np.linalg.norm(delta[0:2])
        vert_range = np.linalg.norm(delta[2])

        return heading, inclination, horiz_range, vert_range, slant_range

def get_nz_for_level_turn_ol(x_f16):
    'get nz to do a level turn'

    # Pull g's to maintain altitude during bank based on trig

    # Calculate theta
    phi = x_f16[StateIndex.PHI]

    if abs(phi): # if cos(phi) ~= 0, basically
        nz = 1 / cos(phi) - 1 # Keeps plane at altitude
    else:
        nz = 0

    return nz

def get_path_angle(x_f16):
    'get the path angle gamma'

    alpha = x_f16[StateIndex.ALPHA]       # AoA           (rad)
    beta = x_f16[StateIndex.BETA]         # Sideslip      (rad)
    phi = x_f16[StateIndex.PHI]           # Roll anle     (rad)
    theta = x_f16[StateIndex.THETA]       # Pitch angle   (rad)

    gamma = asin((cos(alpha)*sin(theta)- \
        sin(alpha)*cos(theta)*cos(phi))*cos(beta) - \
        (cos(theta)*sin(phi))*sin(beta))

    return gamma

def wrap_to_pi(psi_rad):
    '''handle angle wrapping

    returns equivelent angle in range [-pi, pi]
    '''

    rv = psi_rad % (2 * pi)

    if rv > pi:
        rv -= 2 * pi

    return rv

def cart2sph(pt3d):
    '''
    Cartesian to spherical coordinates

    returns az, elev, r
    '''

    x, y, z = pt3d

    h = sqrt(x*x + y*y)
    r = sqrt(h*h + z*z)

    elev = atan2(z, h)
    az = atan2(y, x)

    return az, elev, r

if __name__ == '__main__':
    print("Autopulot script not meant to be run directly.")
