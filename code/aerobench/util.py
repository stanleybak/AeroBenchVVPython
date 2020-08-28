'''
Utilities for F-16 GCAS
'''

from math import floor, ceil
import numpy as np

class StateIndex:
    'list of static state indices'

    VT = 0
    VEL = 0 # alias
    
    ALPHA = 1
    BETA = 2
    PHI = 3 # roll angle
    THETA = 4 # pitch angle
    PSI = 5 # yaw angle
    
    P = 6
    Q = 7
    R = 8
    
    POSN = 9
    POS_N = 9
    
    POSE = 10
    POS_E = 10
    
    ALT = 11
    H = 11
    
    POW = 12

class Freezable():
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

class Euler(Freezable):
    '''fixed step euler integration

    loosely based on scipy.integrate.RK45
    '''

    def __init__(self, der_func, tstart, ystart, tend, step=0, time_tol=1e-9):
        assert step > 0, "arg step > 0 required in Euler integrator"
        assert tend > tstart

        self.der_func = der_func # signature (t, x)
        self.tstep = step
        self.t = tstart
        self.y = ystart.copy()
        self.yprev = None
        self.tprev = None
        self.tend = tend

        self.status = 'running'

        self.time_tol = time_tol

        self.freeze_attrs()

    def step(self):
        'take one step'

        if self.status == 'running':
            self.yprev = self.y.copy()
            self.tprev = self.t
            yd = self.der_func(self.t, self.y)

            self.t += self.tstep

            if self.t + self.time_tol >= self.tend:
                self.t = self.tend

            dt = self.t - self.tprev
            self.y += dt * yd

            if self.t == self.tend:
                self.status = 'finished'

    def dense_output(self):
        'return a function of time'

        assert self.tprev is not None

        dy = self.y - self.yprev
        dt = self.t - self.tprev

        dydt = dy / dt

        def fun(t):
            'return state at time t (linear interpolation)'

            deltat = t - self.tprev

            return self.yprev + dydt * deltat

        return fun

def get_state_names():
    'returns a list of state variable names'

    return ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']

def printmat(mat, main_label, row_label_str, col_label_str):
    'print a matrix'

    if isinstance(row_label_str, list) and len(row_label_str) == 0:
        row_label_str = None

    assert isinstance(main_label, str)
    assert row_label_str is None or isinstance(row_label_str, str)
    assert isinstance(col_label_str, str)

    mat = np.array(mat)
    if len(mat.shape) == 1:
        mat.shape = (1, mat.shape[0]) # one-row matrix

    print("{main_label} =")

    row_labels = None if row_label_str is None else row_label_str.split(' ')
    col_labels = col_label_str.split(' ')

    width = 7

    width = max(width, max([len(l) for l in col_labels]))

    if row_labels is not None:
        width = max(width, max([len(l) for l in row_labels]))

    width += 1

    # add blank space for row labels
    if row_labels is not None:
        print("{: <{}}".format('', width), end='')

    # print col lables
    for col_label in col_labels:
        if len(col_label) > width:
            col_label = col_label[:width]

        print("{: >{}}".format(col_label, width), end='')

    print('')

    if row_labels is not None:
        assert len(row_labels) == mat.shape[0], \
            "row labels (len={}) expected one element for each row of the matrix ({})".format( \
            len(row_labels), mat.shape[0])

    for r in range(mat.shape[0]):
        row = mat[r]

        if row_labels is not None:
            label = row_labels[r]

            if len(label) > width:
                label = label[:width]

            print("{:<{}}".format(label, width), end='')

        for num in row:
            #print("{:#<{}}".format(num, width), end='')
            print("{:{}.{}g}".format(num, width, width-3), end='')

        print('')


def fix(ele):
    'round towards zero'

    assert isinstance(ele, float)

    if ele > 0:
        rv = int(floor(ele))
    else:
        rv = int(ceil(ele))

    return rv

def sign(ele):
    'sign of a number'

    if ele < 0:
        rv = -1
    elif ele == 0:
        rv = 0
    else:
        rv = 1

    return rv

def extract_single_result(res, index, llc):
    'extract a res object for a sinlge aircraft from a multi-aircraft simulation'

    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = res['states'][0].size // num_vars

    if num_aircraft == 1:
        assert index == 0
        rv = res
    else:
        rv = {}
        rv['status'] = res['status']
        rv['times'] = res['times']
        rv['modes'] = res['modes']

        full_states = res['states']
        rv['states'] = full_states[:, num_vars*index:num_vars*(index+1)]

        if 'xd_list' in res:
            # extended states
            key_list = ['xd_list', 'ps_list', 'Nz_list', 'Ny_r_list', 'u_list']

            for key in key_list:
                rv[key] = [tup[index] for tup in res[key]]

    return rv
