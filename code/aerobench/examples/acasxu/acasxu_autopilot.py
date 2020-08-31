'''waypoint autopilot

ported from matlab v2
'''

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

import onnxruntime as ort

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex

class AcasXuAutopilot(Autopilot):
    '''AcasXu autopilot'''

    def __init__(self, init, llc):
        'waypoints is a list of 3-tuples'

        init = np.array(init, dtype=float)

        self.nets = load_networks()
        
        self.intruder_waypoint = make_intruder_setpoint(init)

        num_vars = init.size // 2
        self.intruder_airspeed = init[StateIndex.VEL]
        self.ownship_airspeed = init[num_vars + StateIndex.VEL]

        # default control when not waypoint tracking
        self.cfg_u_ol_default = (0, 0, 0, 0.3)

        # control config
        # Gains for speed control
        self.cfg_k_vt = 0.25

        # Gains for altitude tracking
        self.cfg_k_alt = 0.005
        self.cfg_k_h_dot = 0.02

        # Gains for heading tracking
        self.cfg_k_prop_psi = 5
        self.cfg_k_der_psi = 0.5

        # Gains for roll tracking
        self.cfg_k_prop_phi = 0.75
        self.cfg_k_der_phi = 0.5
        self.cfg_max_bank_deg = 65 # maximum bank angle setpoint
        # v2 was 0.5, 0.9

        # Ranges for Nz
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1

        self.nn_update_rate = 2.0
        self.next_nn_update = 0
        self.command = 0 # current ownship command

        self.labels = ['clear', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        self.history = [] # list of 2-tuples: (command, ownship_state)
        
        Autopilot.__init__(self, self.labels[self.command], llc=llc)

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        premode = self.mode

        tol = 1e-6

        if t + tol > self.next_nn_update:
            self.next_nn_update = t + self.nn_update_rate

            num_vars = x_f16.size // 2
            intruder_state = x_f16[:num_vars]
            ownship_state = x_f16[num_vars:]
        
            self.update_nn_command(t, ownship_state, intruder_state)
            self.mode = self.labels[self.command]

            #print(f"{t}: {self.mode}")
            self.history.append((self.command, ownship_state))

        rv = premode != self.mode

        #if rv:
        #    print(f"transition {premode} -> {self.mode} at time {t}")

        return rv

    def update_nn_command(self, t, ownship_state, intruder_state):
        'update self.command based on the neural network output at the current state'

        x1 = ownship_state[StateIndex.POS_E]
        y1 = ownship_state[StateIndex.POS_N]
        heading1 = -ownship_state[StateIndex.PSI] + pi/2
        vel1 = ownship_state[StateIndex.VEL]

        x2 = intruder_state[StateIndex.POS_E]
        y2 = intruder_state[StateIndex.POS_N]
        heading2 = -intruder_state[StateIndex.PSI] + pi/2
        vel2 = intruder_state[StateIndex.VEL]

        heading1 = wrap_to_pi(heading1)
        heading2 = wrap_to_pi(heading2)

        rho = sqrt((x1 - x2)**2 + (y1 - y2)**2)

        dy = y2 - y1
        dx = x2 - x1

        theta = atan2(dy, dx)
        psi = heading2 - heading1
        v_own = vel1
        v_int = vel2

        theta -= heading1

        # ensure angles in range -pi, pi
        theta = wrap_to_pi(theta)
        assert -pi <= theta <= pi

        psi = wrap_to_pi(psi)
        assert -pi <= psi <= pi

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        stdout = False #self.time < 0.5

        if stdout:
            print(f"\nstate at time {t}, x1: {x1}, y1: {y1}, " + \
              f"heading1: {heading1}, x2: {x2}, y2: {y2}, heading2: {heading2}")

            print(f"input (before scaling): rho: {rho}, theta: {theta}, psi: {psi}, v_own: {v_own}, v_int: {v_int}")

        if rho > 60760:
            self.command = 0
        else:
            last_command = self.command

            net = self.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = scale_and_run_network(net, state, stdout)
            self.command = np.argmin(res)

            if stdout:
                print(f"Unscaled network output ({self.labels[self.command]}): {res}")

    def get_u_ref(self, t, x_f16):
        '''get the reference input signals'''

        num_vars = x_f16.size // 2
        intruder_state = x_f16[:num_vars]
        ownship_state = x_f16[num_vars:]

        u_ref_intruder = self.get_u_ref_intruder(intruder_state)
        u_ref_ownship = self.get_u_ref_ownship(ownship_state)

        return u_ref_intruder + u_ref_ownship

    def track_altitude(self, x_f16, h_cmd):
        'get nz to track altitude, taking turning into account'

        h = x_f16[StateIndex.ALT]
        phi = x_f16[StateIndex.PHI]

        # Calculate altitude error (positive => below target alt)
        h_error = h_cmd - h
        nz_alt = self.track_altitude_wings_level(x_f16, h_cmd)
        nz_roll = get_nz_for_level_turn_ol(x_f16)

        if h_error > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)

        return nz

    def get_phi_to_track_heading(self, x_f16, psi_cmd):
        'get phi from psi_cmd'

        # PD Control on heading angle using phi_cmd as control

        # Pull out important variables for ease of use
        psi = wrap_to_pi(x_f16[StateIndex.PSI])
        r = x_f16[StateIndex.R]

        # Calculate PD control
        psi_err = wrap_to_pi(psi_cmd - psi)

        phi_cmd = psi_err * self.cfg_k_prop_psi - r * self.cfg_k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)

        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        return phi_cmd

    def track_roll_angle(self, x_f16, phi_cmd):
        'get roll angle command (ps_cmd)'

        # PD control on roll angle using stability roll rate

        # Pull out important variables for ease of use
        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        # Calculate PD control
        ps = (phi_cmd-phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi

        return ps

    def track_airspeed(self, x_f16, vt_cmd):
        'get throttle command'

        # Proportional control on airspeed using throttle
        throttle = self.cfg_k_vt * (vt_cmd - x_f16[StateIndex.VT])

        return throttle

    def track_altitude_wings_level(self, x_f16, h_cmd):
        'get nz to track altitude'

        vt = x_f16[StateIndex.VT]
        h = x_f16[StateIndex.ALT]

        # Proportional-Derivative Control
        h_error = h_cmd - h
        gamma = get_path_angle(x_f16)
        h_dot = vt * sin(gamma) # Calculated, not differentiated

        # Calculate Nz command
        nz = self.cfg_k_alt*h_error - self.cfg_k_h_dot*h_dot

        return nz

    def get_u_ref_ownship(self, x_f16):
        '''get the reference input for ownship'''

        roll_rate_cmd_list = [0, -1.5, 1.5, -3.0, 3.0] # deg / sec
        roll_rate_cmd_deg = roll_rate_cmd_list[self.command]

        # these bank angle cmds were empirically found to achieve the desired turn rate
        if roll_rate_cmd_deg == 0:
            psi_cmd_deg = 0
        elif roll_rate_cmd_deg == 1.5:
            psi_cmd_deg = 34
        elif roll_rate_cmd_deg == -1.5:
            psi_cmd_deg = -34
        elif roll_rate_cmd_deg == 3.0:
            psi_cmd_deg = 54
        else:
            assert roll_rate_cmd_deg == -3.0, f"unsupported roll rate cmd: {roll_rate_cmd_deg}"
            psi_cmd_deg = -54

        phi_cmd = np.deg2rad(psi_cmd_deg)

        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)
        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

        alt = self.intruder_waypoint[2]
        nz_cmd = self.track_altitude(x_f16, alt)
        throttle = self.track_airspeed(x_f16, self.ownship_airspeed)

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        throttle = max(min(throttle, 1), 0)

        return [nz_cmd, ps_cmd, 0, throttle]

    def get_u_ref_intruder(self, x_f16):
        '''get the reference input for intruder

        intruder always moves towards self.intruder_waypoint
        '''

        psi_cmd = get_waypoint_data(x_f16, self.intruder_waypoint)[0]

        alt = self.intruder_waypoint[2]
        
        # Get desired roll angle given desired heading
        phi_cmd = self.get_phi_to_track_heading(x_f16, psi_cmd)
        ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

        nz_cmd = self.track_altitude(x_f16, alt)
        throttle = self.track_airspeed(x_f16, self.intruder_airspeed)

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        throttle = max(min(throttle, 1), 0)

        return [nz_cmd, ps_cmd, 0, throttle]

def get_waypoint_data(x_f16, waypoint):
    '''returns current waypoint data tuple based on the current waypoint:

    (heading, inclination, horiz_range, vert_range, slant_range)

    heading = heading to tgt, equivalent to psi (rad)
    inclination = polar angle to tgt, equivalent to theta (rad)
    horiz_range = horizontal range to tgt (ft)
    vert_range = vertical range to tgt (ft)
    slant_range = total range to tgt (ft)
    '''

    e_pos = x_f16[StateIndex.POSE]
    n_pos = x_f16[StateIndex.POSN]
    alt = x_f16[StateIndex.ALT]

    delta = [waypoint[i] - [e_pos, n_pos, alt][i] for i in range(3)]

    _, inclination, slant_range = cart2sph(delta)

    heading = wrap_to_pi(pi/2 - atan2(delta[1], delta[0]))

    horiz_range = np.linalg.norm(delta[0:2])
    vert_range = np.linalg.norm(delta[2])

    return heading, inclination, horiz_range, vert_range, slant_range

def get_nz_for_level_turn_ol(x_f16):
    'get nz to do a level turn'

    # Pull g's to maintain altitude during bank based on trig

    # Calculate theta
    phi = x_f16[StateIndex.PHI]

    if abs(phi): # if cos(phi) ~= 0, basically
        nz = 1 / cos(phi) - 1 # Keeps plane at altitude
    else:
        nz = 0

    return nz

def get_path_angle(x_f16):
    'get the path angle gamma'

    alpha = x_f16[StateIndex.ALPHA]       # AoA           (rad)
    beta = x_f16[StateIndex.BETA]         # Sideslip      (rad)
    phi = x_f16[StateIndex.PHI]           # Roll anle     (rad)
    theta = x_f16[StateIndex.THETA]       # Pitch angle   (rad)

    gamma = asin((cos(alpha)*sin(theta)- \
        sin(alpha)*cos(theta)*cos(phi))*cos(beta) - \
        (cos(theta)*sin(phi))*sin(beta))

    return gamma

def wrap_to_pi(psi_rad):
    '''handle angle wrapping

    returns equivelent angle in range [-pi, pi]
    '''

    rv = psi_rad % (2 * pi)

    if rv >= pi:
        rv -= 2 * pi

    return rv

def cart2sph(pt3d):
    '''
    Cartesian to spherical coordinates

    returns az, elev, r
    '''

    x, y, z = pt3d

    h = sqrt(x*x + y*y)
    r = sqrt(h*h + z*z)

    elev = atan2(z, h)
    az = atan2(y, x)

    return az, elev, r

def make_intruder_setpoint(init, hypot=25000):
    '''make the intruder setpoint 3-tuple

    intruder just flies straight
    '''

    alt = init[StateIndex.ALT]
    x = init[StateIndex.POS_E]
    y = init[StateIndex.POS_N]

    psi = init[StateIndex.PSI]

    rad = -psi + pi/2

    new_x = x + hypot * cos(rad)
    new_y = y + hypot * sin(rad)

    return (new_x, new_y, alt)

def predict_with_onnxruntime(sess, input_tensor):
    'run with onnx network with single input and single output'

    names = [i.name for i in sess.get_inputs()]
    assert len(names) == 1

    inp = {names[0]: input_tensor}

    res = sess.run(None, inp)
    assert len(res) == 1

    return res[0]

def scale_and_run_network(sess, data_in, stdout):
    'scale inputs and run network'

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    for i in range(5):
        data_in[i] = (data_in[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"scaled inputs: {data_in}")

    data_in = np.array(data_in, dtype=np.float32)
    data_in.shape = (1, 1, 1, 5)

    return predict_with_onnxruntime(sess, data_in)

def load_networks():
    'load 5 neural networks and return a list'

    nets = []

    for net in range(1, 6):
        filename = f"nn/ACASXU_run2a_{net}_1_batch_2000.onnx"

        #model = onnx.load(filename)
        #onnx.checker.check_model(model)
        #sess = ort.InferenceSession(model.SerializeToString())
        sess = ort.InferenceSession(filename)

        nets.append(sess)

    return nets

if __name__ == '__main__':
    print("Autopulot script not meant to be run directly.")
