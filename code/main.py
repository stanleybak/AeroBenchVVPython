'''
Stanley Bak
Python version of GCAS maneuver benchmark
'''

import time

from math import pi
from numpy import deg2rad # pylint: disable=E0611

from RunF16Sim import RunF16Sim
from PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits

def main():
    'main function'

    printOn = False # print to console

    # Initial Conditions
    powg = 9 # Power

    # Default alpha & beta
    alphag = deg2rad(2.1215) # Trim Angle of Attack (rad)
    betag = 0                   # Side slip angle (rad)

    # Initial Attitude (for simpleGCAS)
    altg = 3500
    Vtg = 540                        # Pass at Vtg = 540;    Fail at Vtg = 550;
    phig = (pi/2)*0.5           # Roll angle from wings level (rad)
    thetag = (-pi/2)*0.8        # Pitch angle from nose level (rad)
    psig = -pi/4                # Yaw angle from North (rad)
    tMax = 15

    flightLimits = FlightLimits()
    ctrlLimits = CtrlLimits()

    print "main.py: Can't initialize autopilot here because no trim conditions (yet)"
    autopilot = None
    #autopilot = GcasAutopilot()

    pass_fail = FlightLimitsPFA(printOn, flightLimits)

    ctrlLimits.ThrottleMax = 0.7 # Limit to Mil power (no afterburner)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vtg, alphag, betag, phig, thetag, psig, 0, 0, 0, 0, 0, altg, powg]
    orient = 4 # Orientation for trim

    # save an animation video? Try a filename ending in .gif or .mp4 (slow). Using '' will plot to the screen.
    animFilename = None

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # or 'stevens'

    start = time.time()
    passed = RunF16Sim(initialState, tMax, orient, f16_plant, \
        flightLimits, ctrlLimits, autopilot, pass_fail, printOn, animFilename)

    print "Runtime: {:.2f}s".format(time.time() - start)
    print "Simulation Passed: {}".format(passed)

if __name__ == '__main__':
    main()
