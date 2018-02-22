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

    def __init__(self, printOn, trim_alt, trim_Vt, trim_phi, trim_theta, model):
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
                                    self.model, adjust_cy=True)

        if self.printOn:
            print '------------------------------------------------------------'
            print 'Equilibrium / Trim Conditions'
            printmat(xequil, 'State Equilibrium', [], 'Vt alpha beta phi theta psi p q r pn pe alt pow')
            printmat(uequil, 'Control Equilibrium', [], 'throttle elevator aileron rudder')

        return xequil, uequil

    def compute_gain_matrix(self, xequil, uequil, build_a_b_tilde_func):
        'compute the control gain K matrix'

        A, B, C, D = jacobFun(xequil, uequil, self.printOn, self.model, adjust_cy=True)

        Atilde, Btilde = build_a_b_tilde_func(A, B, C, D, xequil)

        q = np.diag(self.q_list)
        r = np.diag(self.r_list)

        K_mat, _, _ = lqr(Atilde, Btilde, q, r) # requires python-slycot and python-control be installed

        return K_mat

def build_a_b_tilde_longitudinal(A, B, C, D, _):
    'build the A and B tilde matrices for optimization'

    si = [X_ALPHA, X_Q] # state indices
    ii = [U_ELEVATOR] # input indices
    oi = [Y_AZ] # output indices

    A_con = subindex_mat(A, si, si)
    B_con = subindex_mat(B, si, ii)

    C_con = subindex_mat(C, oi, si)
    D_con = subindex_mat(D, oi, ii)

    # stack A and C in a square matrix
    size = A_con.shape[0] + C_con.shape[0]
    Atilde = np.zeros((size, size))
    Atilde[:A_con.shape[0], :A_con.shape[1]] = A_con
    Atilde[A_con.shape[0]:, :C_con.shape[1]] = C_con

    # stack B and D
    Btilde = np.array([row for row in B_con] + [row for row in D_con], dtype=float)

    return Atilde, Btilde

def build_a_b_tilde_lateral(A, B, C, D, xequil):
    'build the A and B tilde matrices for optimization'

    si = [X_BETA, X_P, X_R] # state indices
    ii = [U_AILERON, U_RUDDER] # input indices
    # the two outputs are:
    # ps - stability roll rate, nonlinear formula is ps = p*cos(alpha) + r*sin(alpha), small angle is [p + r*alpha]
    # Ny+r - coupling of term between yaw rate and side accel that pilots don't like which we command to zero

    # Outputs:  beta, p, r, ps, Ny+r

    A_con = subindex_mat(A, si, si)

    B_con = subindex_mat(B, si, ii)

    C_top = subindex_mat(C, [Y_P], si) + subindex_mat(C, [Y_R], si) * xequil[X_ALPHA] # ps ~= p + r*alpha
    C_bottom = subindex_mat(C, [Y_AY], si) + subindex_mat(C, [Y_R], si) # Ny + r
    C_con = np.array([row for row in C_top] + [row for row in C_bottom], dtype=float) # stack C_top and C_bottom

    D_top = subindex_mat(D, [Y_P], ii) + subindex_mat(D, [Y_R], ii) * xequil[X_ALPHA] # ps ~= p + r*alpha

    D_bottom = subindex_mat(D, [Y_AY], ii) + subindex_mat(D, [Y_R], ii) # Ny + r
    D_con = np.array([row for row in D_top] + [row for row in D_bottom], dtype=float) # stack D_top and D_bottom

    # stack A and C in a square matrix
    size = A_con.shape[0] + C_con.shape[0]
    Atilde = np.zeros((size, size))
    Atilde[:A_con.shape[0], :A_con.shape[1]] = A_con
    Atilde[A_con.shape[0]:, :C_con.shape[1]] = C_con

    # stack B and D
    Btilde = np.array([row for row in B_con] + [row for row in D_con], dtype=float)

    return Atilde, Btilde

def main():
    'runs control design and outputs gain matrix and trim point to stdout'

    printOn = False # print the outputs?

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

    # Q: Penalty on State Error in LQR controller
    # These were chosen to try to achieve a natural frequency of 3 rad/sec and a damping ratio (zeta) of 0.707
    # see the matlab code for more analysis of the resultant controllers
    q_alpha = 1000
    q_q = 0
    q_Nz = 1500
    long_q_list = [q_alpha, q_q, q_Nz]

    q_beta = 0
    q_p = 0
    q_r = 0
    q_ps_i = 1200
    q_Ny_r_i = 3000
    lat_q_list = [q_beta, q_p, q_r, q_ps_i, q_Ny_r_i]

    q_lists = [long_q_list, lat_q_list]

    # R: Penalty on Control Effort in LRQ controller
    r_elevator = 1
    long_r_list = [r_elevator]

    r_aileron = 1
    r_rudder = 1
    lat_r_list = [r_aileron, r_rudder]

    r_lists = [long_r_list, lat_r_list]

    orients = [ORIENT_WINGS_LEVEL_NONZERO_GAMMA, ORIENT_WINGS_LEVEL_NONZERO_GAMMA]

    build_tilde_funcs = [build_a_b_tilde_longitudinal, build_a_b_tilde_lateral]

    labels = ['Longitudinal', 'Lateral']
    suffix = ['long', 'lat']

    for i in xrange(len(labels)):
        dd = DesignData(printOn, altg, Vtg, phig, thetag, model)

        dd.q_list = q_lists[i]
        dd.r_list = r_lists[i]

        xequil, uequil = dd.compute_trim_states(orients[i])
        K_mat = dd.compute_gain_matrix(xequil, uequil, build_tilde_funcs[i])

        print "{} Equilibrium Points:".format(labels[i])
        print "xequil_{} = np.array({}, dtype=float).transpose()".format(suffix[i], [x for x in xequil])
        print "uequil_{} = np.array({}, dtype=float).transpose()".format(suffix[i], [u for u in uequil])
        print ""

        print "{} Gain Matrix:".format(labels[i])
        print "K_lqr_{} = np.array({}, dtype=float)".format(suffix[i], [[n for n in row] for row in K_mat])
        print ""

if __name__ == '__main__':
    main()
