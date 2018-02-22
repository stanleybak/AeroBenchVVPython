'''
Stanley Bak
Python version of GCAS maneuver benchmark
'''

from math import pi
from numpy import deg2rad

from RunF16Sim import RunF16Sim
from PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import GcasAutopilot

from plot3d_anim import plot3d_anim

def main():
    'main function'

    flightLimits = FlightLimits()
    ctrlLimits = CtrlLimits()
    llc = LowLevelController(ctrlLimits)
    autopilot = GcasAutopilot(llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    ### Initial Conditions ###
    powg = 9 # Power

    # Default alpha & beta
    alphag = deg2rad(2.1215) # Trim Angle of Attack (rad)
    betag = 0                   # Side slip angle (rad)

    # Initial Attitude
    altg = 3600
    Vtg = 540                   # Pass at Vtg = 540;    Fail at Vtg = 550;
    phig = (pi/2)*0.5           # Roll angle from wings level (rad)
    thetag = (-pi/2)*0.8        # Pitch angle from nose level (rad)
    psig = -pi/4                # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vtg, alphag, betag, phig, thetag, psig, 0, 0, 0, 0, 0, altg, powg]

    # save an animation video? Try a filename ending in .gif or .mp4 (slow). Using '' will plot to the screen.
    animFilename = ''

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 15 # simulation time

    passed, times, states, modes, ps_list, Nz_list = RunF16Sim(initialState, tMax, f16_plant, autopilot, llc, pass_fail)

    print "Simulation Conditions Passed: {}".format(passed)

    if animFilename is not None:
        plot3d_anim(times, states, modes, ps_list, Nz_list, filename=animFilename)

if __name__ == '__main__':
    main()
