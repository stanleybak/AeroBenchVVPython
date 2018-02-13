'''
Compute the Longitudinal Gains using an linearized model and LQR control design

Note: to run this you need to have python-control installed
'''

import numpy as np

from control import lqr # requires python-control be installed

from util import printmat
from trimmerFun import trimmerFun
from jacobFun import jacobFun

def main(printOn=True):
    'runs control design and outputs gain matrix and trim point to stdout'

    if printOn:
        print 'Longitudinal F - 16 Controller for Nz tracking'
        print 'MANUAL INPUTS:'

    # SET THESE VALUES MANUALLY
    altg = 1000 # Altitude guess (ft msl)
    Vtg = 502 # Velocity guess (ft / sec)
    phig = 0 # Roll angle from horizontal guess (deg)
    thetag = 0 # Pitch angle guess (deg)
    # Note: If a gain - scheduled controller is desired, the above values would
    # be varied for each desired trim point.

    model = 'stevens'
    xguess = np.array([Vtg, 0, 0, phig, thetag, 0, 0, 0, 0, 0, 0, altg, 0], dtype=float)

    # u = [throttle elevator aileron rudder]
    uguess = np.array([.2, 0, 0, 0], dtype=float)

    # Orientation for Linearization
    # 1:    Wings Level (gamma = 0)
    # 2:    Wings Level (gamma <> 0)
    # 3:    Constant Altitude Turn
    # 4:    Steady Pull Up
    orient = 4
    inputs = np.array([xguess[0], xguess[11], 0, 0, 0], dtype=float)

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

        printmat(xguess, 'State Guess', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
        printmat(uguess, 'Control Guess', [], 'throttle elevator aileron rudder')

    xequil, uequil = trimmerFun(xguess, uguess, orient, inputs, printOn, model, adjust_cy=False)

    if printOn:
        print '------------------------------------------------------------'
        print 'Equilibrium / Trim Conditions'
        printmat(xequil, 'State Equilibrium', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
        printmat(uequil, 'Control Equilibrium', [], 'throttle elevator aileron rudder')

    # Get Linearized Model
    A, B, C, D = jacobFun(xequil, uequil, printOn, model='stevens', adjust_cy=False)

    ## Decouple Linearized F-16 Model: Isolate Longitudinal States & Actuators
    def subindex_mat(mat, rows, cols):
        'helper function: create a submatrix from a list of rows and columns'

        rv = []

        for row in rows:
            vals = []

            for col in cols:
                vals.append(mat[row, col])

            rv.append(vals)

        return np.array(rv, dtype=float)

    A_long = subindex_mat(A, [1, 7], [1, 7])    # States:   alpha, q
    B_long = subindex_mat(B, [1, 7], [1])       # Inputs:   elevator

    C_long = subindex_mat(C, [2, 1, 0], [1, 7]) # States:   alpha, q
    D_long = subindex_mat(D, [2, 1, 0], [1])    # Inputs:   elevator

    Atilde = np.zeros((3, 3))
    Atilde[:2, :2] = A_long
    Atilde[2, :2] = C_long[2, :]

    Btilde = np.zeros((3, 1))
    Btilde[:2, :1] = B_long
    Btilde[2, 0] = D_long[2, :]

    ## Select State & Control Weights & Penalties
    # Set LQR weights
    # Q: Penalty on State Error
    # These were chosen to try to achieve a natural frequency of 3 rad/sec
    # and a damping ratio (zeta) of 0.707
    # see the matlab code for more analysis of the resultant controller
    q_alpha = 1000
    q_q = 0
    q_Nz = 1500

    # R: Penalty on Control Effort
    r_elevator = 1

    ## Calculate Longitudinal Short Period LQR Gains
    d = np.diag([q_alpha, q_q, q_Nz])

    K_long, _, _ = lqr(Atilde, Btilde, d, r_elevator) # requires python-slycot and python-control be installed

    if printOn:
        printmat(K_long, 'LQR Gains', 'elevator', 'alpha q int_e_Nz')
        print ""

    print "Longitudinal Equilibrium Points:"
    print "xequil_long = np.array({}, dtype=float).transpose()".format([x for x in xequil])
    print "uequil_long = np.array({}, dtype=float).transpose()".format([u for u in uequil])

    print ""
    print "Longitudinal Gain Matrix:"
    print "K_lqr_long = np.array([{}, {}, {}], dtype=float)".format(K_long[0, 0], K_long[0, 1], K_long[0, 2])

if __name__ == '__main__':
    main()
