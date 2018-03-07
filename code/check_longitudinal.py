'''
Stanley Bak

Longitudinal Controller Specification Checking
'''

import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from RunF16Sim import RunF16Sim
from PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import FixedAltitudeAutopilot
from controlledF16 import controlledF16

from plot import plot2d

def main():
    'main function'

    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()
    llc = LowLevelController(ctrlLimits)

    setpoint = 550 # altitude setpoint
    ap = FixedAltitudeAutopilot(setpoint, llc.xequil, llc.uequil, flightLimits, ctrlLimits)

    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    ### Initial Conditions ###
    power = 0 # Power

    # Default alpha & beta
    alpha = 0 # angle of attack (rad)
    beta = 0  # Side slip angle (rad)

    alt = 500 # Initial Attitude
    Vt = 540 # Initial Speed
    phi = 0
    theta = alpha
    psi = 0

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 15.0 # simulation time

    def der_func(t, y):
        'derivative function for RK45'

        der = controlledF16(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0] # air speed
        rv[1] = der[1] # alpha
        rv[4] = der[4] # pitch angle
        rv[7] = der[7] # pitch rate
        rv[11] = der[11] # altitude
        rv[12] = der[12] # power lag term
        rv[13] = der[13] # Nz integrator

        return rv

    passed, times, states, modes, ps_list, Nz_list, u_list = \
        RunF16Sim(initialState, tMax, der_func, f16_plant, ap, llc, pass_fail, sim_step=0.01)

    print "Simulation Conditions Passed: {}".format(passed)

    # plot
    filename = None # longitudinal.png
    plot2d(filename, times, [
        (states, [(0, 'Vt'), (11, 'Altitude')]), \
        (u_list, [(0, 'Throttle'), (1, 'elevator')]), \
        (Nz_list, [(0, 'Nz')]) \
        ])

if __name__ == '__main__':
    main()
