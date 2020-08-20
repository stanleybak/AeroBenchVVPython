'''
Stanley Bak
Python F-16

Apply constraints to x variable
used when finding trim conditions
'''

from math import sin, cos, asin
from tgear import tgear

def conf16(x, u, const):
    'apply constraints to x'

    radgam, singam, rr, pr, tr, phi, cphi, sphi, thetadot, coord, stab, orient = const
    gamm = asin(singam)

    #
    # Steady Level Flight
    #
    if orient == 1:
        x[3] = phi          # Phi
        x[4] = x[1]         # Theta
        x[6] = rr           # Roll Rate
        x[7] = pr           # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    #
    # Steady Climb
    #
    if orient == 2:
        x[3] = phi          # Phi
        x[4] = x[1] + radgam  # Theta
        x[6] = rr           # Roll Rate
        x[7] = pr           # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    #
    # orient=3 implies coordinated turn
    #
    if orient == 3:
        x[6] = -tr * sin(x[4])            # Roll Rate
        x[7] = tr * cos(x[4]) * sin(x[3])  # Pitch Rate
        x[8] = tr * cos(x[4]) * cos(x[3])  # Yaw Rate

    #
    # Pitch Pull Up
    #
    if orient == 4:
        x[4] = x[1]         # Theta = alpha
        x[3] = phi          # Phi
        x[6] = rr           # Roll Rate
        x[7] = thetadot     # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    x[12] = tgear(u[0])

    return x, u
