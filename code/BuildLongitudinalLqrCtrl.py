'''
Compute the Longitudinal and Lateral Gains using an linearized model and LQR control design

Note: to run this you need to have python-control and python-slycot to be installed
'''

import numpy as np

from control import lqr # requires python-control be installed

from util import printmat, Freezable
from trimmerFun import trimmerFun
from jacobFun import jacobFun

# states
X_VT = 0        # air speed, VT    (ft/sec)
X_ALPHA = 1     # angle of attack, alpha  (rad)
X_BETA = 2      # angle of sideslip, beta (rad)
X_PHI = 3       # roll angle, phi  (rad)
X_THETA = 4     # pitch angle, theta  (rad)
X_PSI = 5       # yaw angle, psi  (rad)
X_P = 6         # roll rate, P  (rad/sec)
X_Q = 7         # pitch rate, Q  (rad/sec)
X_R = 8         # yaw rate, R  (rad/sec)
X_PN = 9        # northward horizontal displacement, pn  (feet)
X_PE = 10       # eastward horizontal displacement, pe  (feet)
X_ALTITUDE = 11 # altitude, h  (feet)
X_POW = 12      # engine thrust dynamics lag state, pow
NUM_X = 13

# inputs
U_THROTTLE = 0  # throttle command  0.0 < u(1) < 1.0
U_ELEVATOR = 1  # elevator command in degrees
U_AILERON = 2   # aileron command in degrees
U_RUDDER = 3    # rudder command in degrees
NUM_U = 4

# outputs
Y_AZ = 0        # normal accel
Y_Q = 1         # pitch rate, Q  (rad/sec)
Y_ALPHA = 2     # angle of attack, alpha  (rad)
Y_THETA = 3     # pitch angle, theta  (rad)
Y_VT = 4        # air speed, VT    (ft/sec)
Y_AY = 5        # side accel
Y_P = 6         # roll rate, P  (rad/sec)
Y_R = 7         # yaw rate, R  (rad/sec)
Y_BETA = 8      # angle of sideslip, beta (rad)
Y_PHI = 9       # roll angle, phi  (rad)
NUM_Y = 10

# Orientation for Linearization
ORIENT_WINGS_LEVEL_ZERO_GAMMA = 1 # gamma is the path angle, the angle between the ground and the aircraft
ORIENT_WINGS_LEVEL_NONZERO_GAMMA = 2
ORIENT_CONSTANT_ALTITUDE_TURN = 3
ORIENT_STEADY_PULL_UP = 4

def subindex_mat(mat, rows, cols):
    'helper function: create a submatrix from a list of rows and columns'

    rv = []

    for row in rows:
        vals = []

        for col in cols:
            vals.append(mat[row, col])

        rv.append(vals)

    return np.array(rv, dtype=float)

class DesignData(Freezable):
    'data structure for control gain optimization'

    def __init__(self, printOn, label, trim_alt, trim_Vt, trim_phi, trim_theta, model):
        self.label = label
        self.printOn = printOn

        self.xguess = np.zeros((NUM_X,))
        self.xguess[X_ALTITUDE] = trim_alt
        self.xguess[X_VT] = trim_Vt
        self.xguess[X_PHI] = trim_phi
        self.xguess[X_THETA] = trim_theta

        self.uguess = np.zeros((NUM_U,))
        self.uguess[U_THROTTLE] = 0.2

        self.model = model

        # set these ones after creation of the object
        self.state_indices = None
        self.input_indices = None
        self.output_indices = None
        self.q_list = None
        self.r_list = None

        self.freeze_attrs()

    def compute_trim_states(self, orient):
        'compute the trim states, xequil and uequil'

        # inputs: [Vt, h, gamm, psidot, thetadot]
        inputs = np.zeros((5,))

        inputs[0] = self.xguess[X_VT]
        inputs[1] = self.xguess[X_ALTITUDE]
        #np.array([self.xguess[0], self.xguess[11], 0, 0, 0], dtype=float)

        if self.printOn:
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

            printmat(self.xguess, 'State Guess', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
            printmat(self.uguess, 'Control Guess', [], 'throttle elevator aileron rudder')

        xequil, uequil = trimmerFun(self.xguess, self.uguess, orient, inputs, self.printOn, \
                                    self.model, adjust_cy=False)

        if self.printOn:
            print '------------------------------------------------------------'
            print 'Equilibrium / Trim Conditions'
            printmat(xequil, 'State Equilibrium', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
            printmat(uequil, 'Control Equilibrium', [], 'throttle elevator aileron rudder')

        return xequil, uequil

    def compute_gain_matrix(self, xequil, uequil):
        'compute the control gain K matrix'

        A, B, C, D = jacobFun(xequil, uequil, self.printOn, self.model, adjust_cy=False)

        si = self.state_indices
        ii = self.input_indices
        oi = self.output_indices

        A_con = subindex_mat(A, si, si)
        B_con = subindex_mat(B, si, ii)

        C_con = subindex_mat(C, oi, si)
        D_con = subindex_mat(D, oi, ii)

        Atilde = np.zeros((len(si) + len(oi), len(si) + len(oi))) # stack A and C
        Atilde[:len(si), :len(si)] = A_con
        Atilde[len(si):, :len(si)] = C_con

        Btilde = np.zeros((len(si) + len(oi), len(ii))) # stack B and D
        Btilde[:len(si), :len(ii)] = B_con
        Btilde[len(si):, :len(ii)] = D_con

        # Atilde = np.zeros((len(si) + len(oi), len(si) + len(oi))) # stack A and C
        # Atilde[:2, :2] = A_long
        # Atilde[2, :2] = C_long[2, :]
        #
        # Btilde = np.zeros((3, 1)) # stack B and D
        # Btilde[:2, :1] = B_long
        # Btilde[2, 0] = D_long[2, :]

        q = np.diag(self.q_list)
        r = np.diag(self.r_list)

        K_mat, _, _ = lqr(Atilde, Btilde, q, r) # requires python-slycot and python-control be installed

        return K_mat

def main(printOn=True):
    'runs control design and outputs gain matrix and trim point to stdout'

    if printOn:
        print 'Longitudinal F - 16 Controller for Nz tracking'
        print 'MANUAL INPUTS:'

    # Trim point - manually set
    # Note: If a gain - scheduled controller is desired, the above values would
    # be varied for each desired trim point.
    altg = 1000 # Altitude guess (ft msl)
    Vtg = 502 # Velocity guess (ft / sec)
    phig = 0 # Roll angle from horizontal guess (deg)
    thetag = 0 # Pitch angle guess (deg)
    model = 'stevens'

    long_data = DesignData(printOn, 'longitudinal', altg, Vtg, phig, thetag, model)

    long_data.state_indices = [X_ALPHA, X_Q]
    long_data.input_indices = [U_ELEVATOR]
    long_data.output_indices = [Y_AZ]

    ## Select State & Control Weights & Penalties
    # Set LQR weights
    # Q: Penalty on State Error
    # These were chosen to try to achieve a natural frequency of 3 rad/sec
    # and a damping ratio (zeta) of 0.707
    # see the matlab code for more analysis of the resultant controller
    q_alpha = 1000
    q_q = 0
    q_Nz = 1500
    long_data.q_list = [q_alpha, q_q, q_Nz]

    # R: Penalty on Control Effort
    r_elevator = 1
    long_data.r_list = [r_elevator]

    orient = ORIENT_STEADY_PULL_UP

    xequil, uequil = long_data.compute_trim_states(orient)
    K_long = long_data.compute_gain_matrix(xequil, uequil)

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
