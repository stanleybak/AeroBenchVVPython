'''waypoint autopilot

ported from matlab v2
'''

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController

class AcasXuAutopilot(Autopilot):
    '''AcasXu autopilot'''

    def __init__(self, llc):
        'waypoints is a list of 3-tuples'

        Autopilot.__init__(self, 'init_mode', llc=llc)

    def get_u_ref(self, t, x_f16):
        '''get the reference input signals'''

        rv = [0, 0, 0, 0] + [0, 0, 0, 0]

        return rv

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
