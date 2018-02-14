'''
Stanley Bak
RunF16Sim python version
'''

import time

import numpy as np

from scipy.integrate import RK45

from util import printmat
from trimmerFun import trimmerFun
from autopilot import GcasAutopilot
from controlledF16 import controlledF16
from plot3d_anim import plot3d_anim

def RunF16Sim(initialState, tMax, orient, F16Model, flightLimits, ctrlLimits, autopilot, pass_fail, printOn, \
             animFilename):
    'Simulates and analyzes autonomous F-16 maneuvers'

    assert not (flightLimits.vMin < 300 or flightLimits.vMax > 900), \
        'flightLimits: Airspeed limits outside model limits (300 to 900)'

    assert not (flightLimits.alphaMinDeg < -10 or flightLimits.alphaMaxDeg > 45), \
        'flightLimits: Alpha limits outside model limits (-10 to 45)'

    assert not (abs(flightLimits.betaMaxDeg) > 30), 'flightLimits: Beta limit outside model limits (30 deg)'

    assert not (ctrlLimits.ThrottleMin < 0 or ctrlLimits.ThrottleMax > 1), 'ctrlLimits: Throttle Limits (0 to 1)'

    assert not (ctrlLimits.ElevatorMaxDeg > 25 or ctrlLimits.ElevatorMinDeg < -25), \
        'ctrlLimits: Elevator Limits (-25 deg to 25 deg)'

    assert not (ctrlLimits.AileronMaxDeg > 21.5 or ctrlLimits.AileronMinDeg < -21.5), \
        'ctrlLimits: Aileron Limits (-21.5 deg to 21.5 deg)'

    assert not (ctrlLimits.RudderMaxDeg > 30 or ctrlLimits.RudderMinDeg < -30), \
        'ctrlLimits: Rudder Limits (-30 deg to 30 deg)'

    # Get Trim / Equilibrium Conditions
    if printOn:
        print '------------------------------------------------------------'
        print 'F-16 Decoupled LQR Controller for Nz, P_s, and Ny+r tracking'
        print '------------------------------------------------------------'

    # Format initial Conditions for Simulation and append integral error states
    initialState = np.array(initialState, dtype=float)

    x0 = np.zeros((initialState.shape[0] + 3,))
    x0[:initialState.shape[0]] = initialState

    # Define Control Guess
    uguess = np.array([.2, 0, 0, 0], dtype=float)

    # Format inputs for trimmerFun
    inputs = np.array([initialState[0], initialState[11], 0, 0, 0], dtype=float)

    if printOn:
        printmat(inputs, 'Operator Inputs', [], 'Vt h gamma psidot thetadot')

        print 'Trim Orientation Selected:   ',
        if orient == 1:
            print 'Wings Level (gamma = 0)'
        elif orient == 2:
            print 'Wings Level (gamma <> 0)'
        elif orient == 3:
            print 'Constant Altitude Turn'
        elif orient == 4:
            print 'Steady Pull Up'
        else:
            assert False, 'Invalid Orientation (orient) for trimmerFun'

        printmat(initialState, 'Inititial Conditions', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
        printmat(uguess, 'Control Guess', [], 'throttle elevator aileron rudder')

    xequil, uequil = trimmerFun(initialState, uguess, orient, inputs, printOn)

    if printOn:
        print '------------------------------------------------------------'
        print 'Equilibrium / Trim Conditions'
        printmat(xequil, 'State Equilibrium', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
        printmat(uequil, 'Control Equilibrium', [], 'throttle elevator aileron rudder')

    print "RunF16Sim.py: temporarily initializing GCAS autopilot here (after trim conditions are found)"
    autopilot = GcasAutopilot(xequil, uequil, flightLimits, ctrlLimits)

    # Hard coded LQR gain matrix
    K_lqr = np.zeros((3, 8))

    # Longitudinal Gains
    K_lqr[:1, :3] = np.array([-156.88020, -31.03700, -38.72980], dtype=float)

    # Lateral Gains
    K_lqr[1:, 3:] = np.array([[38.02750, -5.65500, -14.08800, -34.06420, -9.95410], \
                              [17.56400, 1.58390, -41.43510, 6.29550, -53.86020]], dtype=float)

    if printOn:
        printmat(K_lqr, 'Decoupled LQR Controller Gains', 'elevator aileron rudder', \
                 'alpha q Nz_i beta p r ps_i Ny_r_i')

    # Simulate Using ODE Solver
    if printOn:
        print '------------------------------------------------------------'
        print 'Running Nonlinear Simulation using RK45'

    # run the numerical simulation
    der_func = lambda t, y: controlledF16(t, y, xequil, uequil, K_lqr, F16Model, ctrlLimits, autopilot)[0]

    #max_step = None #0.01
    times = [0]
    states = [x0]
    modes = []
    Nz_list = []
    ps_list = []

    start = time.time()

    rk45 = RK45(der_func, times[-1], states[-1], tMax)

    sim_step = 0.01

    # state may advance at time zero
    updated = autopilot.advance_discrete_state(rk45.t, rk45.y)
    modes.append(autopilot.state)

    xd, u, Nz, ps, Ny_r = controlledF16(times[-1], states[-1], xequil, uequil, K_lqr, F16Model, \
                                    ctrlLimits, autopilot)
    pass_fail.advance(rk45.t, rk45.y, autopilot.state, xd, u, Nz, ps, Ny_r)
    Nz_list.append(Nz)
    ps_list.append(ps)

    if updated:
        rk45 = RK45(der_func, times[-1], states[-1], tMax)

    while rk45.status == 'running':
        rk45.step()

        if rk45.t > times[-1] + sim_step:
            dense_output = rk45.dense_output()

            while rk45.t > times[-1] + sim_step:
                t = times[-1] + sim_step
                times.append(t)
                states.append(dense_output(t))

                updated = autopilot.advance_discrete_state(rk45.t, rk45.y)
                modes.append(autopilot.state)

                # re-run dynamics function at current state to get non-state variables
                xd, u, Nz, ps, Ny_r = controlledF16(times[-1], states[-1], xequil, uequil, K_lqr, F16Model, \
                                                ctrlLimits, autopilot)
                pass_fail.advance(rk45.t, rk45.y, autopilot.state, xd, u, Nz, ps, Ny_r)
                Nz_list.append(Nz)
                ps_list.append(ps)

                if updated:
                    rk45 = RK45(der_func, times[-1], states[-1], tMax)
                    break

    # make sure the solver didn't fail
    assert rk45.status == 'finished', "rk.status was {}".format(rk45.status)
    print "Simulation Time: {:.4}s".format(time.time() - start)

    if animFilename is not None:
        plot3d_anim(times, states, modes, ps_list, Nz_list, filename=animFilename)
