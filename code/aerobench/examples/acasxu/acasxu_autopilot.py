'''acasxu autopilot
with support for multiple aircraft
'''

from typing import List, Optional

import os
from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

import onnxruntime as ort

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex, get_state_names, get_script_path

class AcasXuAutopilot(Autopilot):
    '''AcasXu autopilot'''

    def __init__(self, init, llc, num_aircraft_acasxu=1, stop_on_coc=False,
                 hardcoded_u_seq=None, stdout=True):
        'waypoints is a list of 3-tuples'

        init = np.array(init, dtype=float)

        self.nets = load_networks()

        self.stop_on_coc = stop_on_coc
        self.coc_time = None
        self.coc_stop_delay = 10

        # used fixed outputs from acasxu system instead of running neural networks
        self.hardcoded_u_seq = hardcoded_u_seq
        self.hardcoded_cur_step = 0

        self.num_vars = len(get_state_names()) + llc.get_num_integrators()
        assert init.size % self.num_vars == 0
        self.num_aircraft = init.size // self.num_vars
        self.num_aircraft_acasxu = num_aircraft_acasxu

        # waypoints for all airfract
        self.intruder_waypoints = make_intruder_waypoints(init, self.num_vars)

        self.init_airspeed = []

        for a in range(self.num_aircraft):
            self.init_airspeed.append(init[self.num_vars * a + StateIndex.VEL])

        # default control when not running acasxu
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

        # current ownship commands
        self.commands = [0] * self.num_aircraft_acasxu

        # list with one entry for each acasxu aircraft
        # each list entry is a list with one element for every other aircraft (and 0 for self),
        # which is the last acasxu command for that aircraft
        self.all_acasxu_commands = []

        # this is the command used at each step for each acasxu aircraft. you can pass this into hardcoded_command_seq
        self.command_history = []

        for _ in range(self.num_aircraft_acasxu):
            self.all_acasxu_commands.append([0] * self.num_aircraft)

        # closest intruder with no clear of conflict command
        self.closest_intruder_indices: List[Optional[int]] = [None] * self.num_aircraft_acasxu

        self.labels = ['clear', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        self.history = [] # list of 2-tuples: (command, ownship_state)

        # list of 3-tuples: (time, all_acasxu_commands, closest_intruder_indices)
        self.full_history = []

        self.stdout = stdout

        mode = "/".join([self.labels[c] for c in self.commands])

        Autopilot.__init__(self, mode, llc=llc)

    def is_finished(self, t, x_f16):
        'is the maneuver done?'

        rv = False

        if self.stop_on_coc:
            if t > 0:
                all_coc = all([c == 0 for c in self.commands])

                if all_coc:
                    if self.coc_time is None:
                        self.coc_time = t

                    rv = self.coc_time + self.coc_stop_delay < t
                else:
                    self.coc_time = None

        return rv

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        premode = self.mode

        tol = 1e-6

        if t + tol > self.next_nn_update:
            self.next_nn_update = t + self.nn_update_rate

            if self.hardcoded_u_seq:
                # use a hardcoded command rather than running the neural networks
                if self.hardcoded_cur_step >= len(self.hardcoded_u_seq):
                    self.hardcoded_cur_step = len(self.hardcoded_u_seq) - 1

                hardcoded_command = self.hardcoded_u_seq[self.hardcoded_cur_step]

                if isinstance(hardcoded_command, int):
                    hardcoded_command = [hardcoded_command] * self.num_aircraft_acasxu

                for a in range(self.num_aircraft_acasxu):
                    self.commands[a] = hardcoded_command[a]

                self.hardcoded_cur_step += 1
            else:

                #print("--------------------")

                for a in range(self.num_aircraft_acasxu):
                    ownship_state = x_f16[a*self.num_vars:(a+1)*self.num_vars]

                    x1 = ownship_state[StateIndex.POS_E]
                    y1 = ownship_state[StateIndex.POS_N]

                    stdout = False #a in [0, 5]

                    if stdout:
                        print(f"\nUpdating plane {a} at time {t}. State is {x1, y1}")

                    self.commands[a] = 0 # set command to clear of conflict
                    closest_dist_sq = np.inf
                    closest_intruder_index = None

                    # intruder is the closest aircraft in the x/y space
                    for b in range(self.num_aircraft):
                        if a == b:
                            continue

                        intruder_state = x_f16[b*self.num_vars:(b+1)*self.num_vars]

                        # this updates self.all_acasxu_commands[a][b]
                        self.update_nn_command(t, a, ownship_state, b, intruder_state, stdout=stdout)
                        c = self.all_acasxu_commands[a][b]

                        # run acas xu on the intruder

                        x2 = intruder_state[StateIndex.POS_E]
                        y2 = intruder_state[StateIndex.POS_N]

                        dist_sq = (x1-x2)**2 + (y1-y2)**2

                        if stdout:
                            print(f"b={b}. State is {x2, y2}, distSq is {dist_sq}")

                        if dist_sq < closest_dist_sq and c != 0:
                            closest_dist_sq = dist_sq
                            closest_intruder_index = b
                            self.commands[a] = c

                    if stdout:
                        print(f"closest intruder index: {closest_intruder_index}")
                        print(f"command issued: {self.labels[self.commands[a]]} ({self.commands[a]})")

                    if a == 0:
                        self.command_history.append([None] * self.num_aircraft_acasxu)

                    self.command_history[-1][a] = self.commands[a]
                    self.history.append((self.commands[a], ownship_state))
                    self.closest_intruder_indices[a] = closest_intruder_index

                tup = (t, np.array(self.all_acasxu_commands), np.array(self.closest_intruder_indices))
                self.full_history.append(tup)

        self.mode = "/".join([self.labels[c] for c in self.commands])
        rv = premode != self.mode

        if rv and self.stdout:
            print(f"transition {premode} -> {self.mode} at time {t}")

        return rv

    def update_nn_command(self, t, ownship_index, ownship_state, intruder_index, intruder_state, stdout=False):
        '''
        updates self.all_acasxu_commands[ownship_index][intruder_index]

        based on the neural network output at the current state
        '''

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

        #self.time < 0.5

        if stdout:
            print(f"State at time {t}, x1: {x1}, y1: {y1}, " + \
              f"heading1: {heading1}, x2: {x2}, y2: {y2}, heading2: {heading2}")

            print(f"input (before scaling): rho: {rho}, theta: {theta}, psi: {psi}, v_own: {v_own}, v_int: {v_int}")

        if rho > 60760:
            self.all_acasxu_commands[ownship_index][intruder_index] = 0
        else:
            last_command = self.all_acasxu_commands[ownship_index][intruder_index]
            # note using last_command=0 seems to work better for aircraft >= 3

            net = self.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = scale_and_run_network(net, state, stdout)
            c = np.argmin(res)
            self.all_acasxu_commands[ownship_index][intruder_index] = c

            if stdout:
                print(f"Unscaled network output ({self.labels[c]}): {res}")

    def get_u_ref(self, t, x_f16):
        '''get the reference input signals'''

        rv = []
        start = 0

        for a in range(self.num_aircraft):
            end = start + self.num_vars
            state = x_f16[start:end]
            start += self.num_vars
            
            if a < self.num_aircraft_acasxu:
                rv += self.get_u_ref_ownship(state, a)
            else:
                rv += self.get_u_ref_intruder(state, a)

        #print(f".debug {t}, u_ref = {rv}")

        return rv

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

    def get_u_ref_ownship(self, x_f16, index):
        '''get the reference input for ownship'''

        command = self.commands[index]
        assert 0 <= command <= 4, f"invalid command in get u_ref ownship: {command}"

        roll_rate_cmd_list = [0, -1.5, 1.5, -3.0, 3.0] # deg / sec
        roll_rate_cmd_deg = roll_rate_cmd_list[command]

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

        alt = self.intruder_waypoints[index][2]
        nz_cmd = self.track_altitude(x_f16, alt)
        throttle = self.track_airspeed(x_f16, self.init_airspeed[index])

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        throttle = max(min(throttle, 1), 0)

        return [nz_cmd, ps_cmd, 0, throttle]

    def get_u_ref_intruder(self, x_f16, index):
        '''get the reference input for intruder

        intruder always moves towards self.intruder_waypoint
        '''

        psi_cmd = get_waypoint_data(x_f16, self.intruder_waypoints[index])[0]

        alt = self.intruder_waypoints[index][2]

        # Get desired roll angle given desired heading
        phi_cmd = self.get_phi_to_track_heading(x_f16, psi_cmd)
        ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

        nz_cmd = self.track_altitude(x_f16, alt)
        throttle = self.track_airspeed(x_f16, self.init_airspeed[index])

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

def make_intruder_waypoints(init, num_vars, hypot=1e5):
    '''make the intruder waypoints (list of 3-tuples)

    assumes intruder just flies straight
    '''

    rv = []

    start = 0

    while start < init.size:
        end = start + num_vars
        state = init[start:end]
        start += num_vars

        alt = state[StateIndex.ALT]
        x = state[StateIndex.POS_E]
        y = state[StateIndex.POS_N]

        psi = state[StateIndex.PSI]

        rad = -psi + pi/2

        new_x = x + hypot * cos(rad)
        new_y = y + hypot * sin(rad)

        wp = (new_x, new_y, alt)
        rv.append(wp)

    return rv

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

    input_ranges = [[0, 60760], [-pi, pi], [-pi, pi], [100, 1200], [0, 1200]]

    for i, (val, input_range) in enumerate(zip(data_in, input_ranges)):
        assert input_range[0] - 1e-6 <= val <= input_range[1] + 1e-6, \
            f"acasxu neural network input {i} ({val}) not in range {input_range}"

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
        dir_name = get_script_path(__file__)
        filename = os.path.join(dir_name, "nn", f"ACASXU_run2a_{net}_1_batch_2000.onnx")

        sess = ort.InferenceSession(filename)

        nets.append(sess)

    return nets

if __name__ == '__main__':
    print("Autopilot script not meant to be run directly.")
