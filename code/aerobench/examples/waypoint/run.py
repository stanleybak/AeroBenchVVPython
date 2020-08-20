'''
Stanley Bak

should match 'waypoint' scenario from matlab version
'''

import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.highlevel.autopilot import Autopilot

from aerobench.visualize import plot

class WaypointAutopilot(Autopilot):
    '''Simple Autopilot with proportional control'''

    def __init__(self, alt_setpoint):
        self.alt_setpoint = alt_setpoint

        Autopilot.__init__(self)

    def _get_u_ref(self, _t, x_f16):
        '''get the reference inputs signals'''

        airspeed = x_f16[0]   # Vt            (ft/sec)
        alpha = x_f16[1]      # AoA           (rad)
        theta = x_f16[4]      # Pitch angle   (rad)
        gamma = theta - alpha # Path angle    (rad)
        h = x_f16[11]         # Altitude      (feet)

        # Proportional Control
        k_alt = 0.025
        h_error = self.alt_setpoint - h
        Nz = k_alt * h_error # Allows stacking of cmds

        # (Psuedo) Derivative control using path angle
        k_gamma = 25
        Nz = Nz - k_gamma*gamma

        # try to maintain a fixed airspeed near trim point
        K_vt = 0.25
        throttle = -K_vt * (airspeed - self.xequil[0])

        return Nz, 0, 0, throttle

def main():
    'main function'

    ### Initial Conditions ###
    power = 7.6 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(1.8) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = math.pi/4   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 25 # simulation time

    # make waypoints
    e_pt = 3000
    n_pt = 3000
    h_pt = 4000

    waypoints = [
        (e_pt, n_pt, h_pt),
        (e_pt + 2000, n_pt + 5000, h_pt - 500),
        (e_pt + 1000, n_pt + 10000, h_pt - 750),
        (e_pt - 500, n_pt + 15000, h_pt - 1250)
    ]
    
    ap = WaypointAutopilot(alt)

    res = run_f16_sim(init, tmax, ap)

    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")

    plot.plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    
    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
