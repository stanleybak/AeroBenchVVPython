'''
Stanley Bak
run_f16_sim python version
'''

import time

import numpy as np
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex, print_state, Freezable

class F16SimState(Freezable):
    '''object containing simulation state

    With this interface you can run partial simulations, rather than having to simulate for the entire time bound

    if you just want a single run with a fixed time, it may be easier to use the run_f16_sim function
    '''

    def __init__(self, initial_state, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True, keep_intermediate_states=True,
                 custom_stop_func=None):

        self.model_str = model_str = ap.llc.model_str
        self.v2_integrators = v2_integrators
        initial_state = np.array(initial_state, dtype=float)

        self.keep_intermediate_states = keep_intermediate_states
        self.custom_stop_func = custom_stop_func

        self.step = step
        self.ap = ap
        self.print_errors = print_errors

        llc = ap.llc

        num_vars = len(get_state_names()) + llc.get_num_integrators()

        if initial_state.size < num_vars:
            # append integral error states to state vector
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"
        self.x0 = x0
        self.ap = ap

        self.times = None
        self.states = None
        self.modes = None

        self.extended_states = extended_states

        if self.extended_states:
            self.xd_list = None
            self.u_list = None
            self.Nz_list = None
            self.ps_list = None
            self.Ny_r_list = None

        self.cur_sim_time = 0
        self.total_sim_time = 0

        self.der_func = make_der_func(ap, model_str, v2_integrators)

        if integrator_str == 'rk45':
            integrator_class = RK45
            self.integrator_kwargs = {}
        else:
            assert integrator_str == 'euler'
            integrator_class = Euler
            self.integrator_kwargs = {'step': step}

        self.integrator_class = integrator_class
        self.integrator = None

        self.freeze_attrs()

    def init_simulation(self):
        'initial simulation (upon first call to simulate_to)'

        assert self.integrator is None

        self.times = [0]
        self.states = [self.x0]

        # mode can change at time 0
        self.ap.advance_discrete_mode(self.times[-1], self.states[-1])

        self.modes = [self.ap.mode]

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(self.ap, self.times[-1], self.states[-1],
                                                      self.model_str, self.v2_integrators)

            self.xd_list = [xd]
            self.u_list = [u]
            self.Nz_list = [Nz]
            self.ps_list = [ps]
            self.Ny_r_list = [Ny_r]
        
        # note: fixed_step argument is unused by rk45, used with euler
        self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                **self.integrator_kwargs)

    def simulate_to(self, tmax, tol=1e-7, update_mode_at_start=False):
        '''simulate up to the passed in time

        this adds states to self.times, self.states, self.modes, and the other extended state lists if applicable 
        '''

        # underflow errors were occuring if I don't do this
        oldsettings = np.geterr()
        np.seterr(all='raise', under='ignore')

        start = time.perf_counter()

        ap = self.ap
        step = self.step

        if self.integrator is None:
            self.init_simulation()
        elif update_mode_at_start:
            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes[-1] = ap.mode # overwrite last mode

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf, \
                                                        **self.integrator_kwargs)

        assert tmax >= self.cur_sim_time
        self.cur_sim_time = tmax

        assert self.integrator.status == 'running', \
            f"integrator status was {self.integrator.status} in call to simulate_to()"

        assert len(self.modes) == len(self.times), f"modes len was {len(self.modes)}, times len was {len(self.times)}"
        assert len(self.states) == len(self.times)

        while True:
            if not self.keep_intermediate_states and len(self.times) > 1:
                # drop all except last state
                self.times = [self.times[-1]]
                self.states = [self.states[-1]]
                self.modes = [self.modes[-1]]

                if self.extended_states:
                    self.xd_list = [self.xd_list[-1]]
                    self.u_list = [self.u_list[-1]]
                    self.Nz_list = [self.Nz_list[-1]]
                    self.ps_list = [self.ps_list[-1]]
                    self.Ny_r_list = [self.Ny_r_list[-1]]

            next_step_time = self.times[-1] + step

            if abs(self.times[-1] - tmax) > tol and next_step_time > tmax:
                # use a small last step
                next_step_time = tmax

            if next_step_time >= tmax + tol:
                # don't do any more steps
                break

            # goal for rest of the loop: do one more step

            while next_step_time >= self.integrator.t + tol:
                # keep advancing integrator until it goes past the next step time
                assert self.integrator.status == 'running'
                
                self.integrator.step()

                if self.integrator.status != 'running':
                    break

            if self.integrator.status != 'running':
                break

            # get the state at next_step_time
            self.times.append(next_step_time)

            if abs(self.integrator.t - next_step_time) < tol:
                self.states.append(self.integrator.x)
            else:
                dense_output = self.integrator.dense_output()
                self.states.append(dense_output(next_step_time))

            # re-run dynamics function at current state to get non-state variables
            if self.extended_states:
                xd, u, Nz, ps, Ny_r = get_extended_states(ap, self.times[-1], self.states[-1],
                                                          self.model_str, self.v2_integrators)

                self.xd_list.append(xd)
                self.u_list.append(u)

                self.Nz_list.append(Nz)
                self.ps_list.append(ps)
                self.Ny_r_list.append(Ny_r)

            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes.append(ap.mode)

            stop_func = self.custom_stop_func if self.custom_stop_func is not None else ap.is_finished

            if stop_func(self.times[-1], self.states[-1]):
                # this both causes the outer loop to exit and sets res['status'] appropriately
                self.integrator.status = 'autopilot finished'
                break

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                        **self.integrator_kwargs)

        if self.integrator.status == 'failed' and self.print_errors:
            print(f'Warning: integrator status was "{self.integrator.status}"')
        elif self.integrator.status != 'autopilot finished':
            assert abs(self.times[-1] - tmax) < tol, f"tmax was {tmax}, self.times[-1] was {self.times[-1]}"

        self.total_sim_time += time.perf_counter() - start
        np.seterr(**oldsettings)

def run_f16_sim(initial_state, tmax, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True,
                custom_stop_func=None):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    fss = F16SimState(initial_state, ap, step, extended_states,
                integrator_str, v2_integrators, print_errors, custom_stop_func=custom_stop_func)

    fss.simulate_to(tmax)

    # extract states

    res = {}
    res['status'] = fss.integrator.status
    res['times'] = fss.times
    res['states'] = np.array(fss.states, dtype=float)
    res['modes'] = fss.modes

    if extended_states:
        res['xd_list'] = fss.xd_list
        res['ps_list'] = fss.ps_list
        res['Nz_list'] = fss.Nz_list
        res['Ny_r_list'] = fss.Ny_r_list
        res['u_list'] = fss.u_list

    res['runtime'] = fss.total_sim_time

    return res

class SimModelError(RuntimeError):
    'simulation state went outside of what the model is capable of simulating'

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]

            #print(f".called der_func(aircraft={i}, t={t}, state={full_state}")

            alpha = state[StateIndex.ALPHA]
            if not -2 < alpha < 2:
                raise SimModelError(f"alpha ({alpha}) out of bounds")

            vel = state[StateIndex.VEL]
            # even going lower than 300 is probably not a good idea
            if not 200 <= vel <= 3000:
                raise SimModelError(f"velocity ({vel}) out of bounds")

            alt = state[StateIndex.ALT]
            if not -10000 < alt < 100000:
                raise SimModelError(f"altitude ({alt}) out of bounds")

            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r
