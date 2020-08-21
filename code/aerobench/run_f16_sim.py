'''
Stanley Bak
run_f16_sim python version
'''

import time

import numpy as np
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler

def run_f16_sim(initial_state, tmax, ap, step=0.01, extended_states=False, model_str='morelli',
                integrator_str='rk45', v2_integrators=False):
    '''Simulates and analyzes autonomous F-16 maneuvers

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    '''

    start = time.perf_counter()

    llc = ap.llc

    # append integral error states to state vector
    initial_state = np.array(initial_state, dtype=float)

    num_vars = len(get_state_names()) + llc.get_num_integrators() + ap.get_num_integrators()

    x0 = np.zeros(num_vars)
    x0[:initial_state.shape[0]] = initial_state

    # run the numerical simulation
    times = [0]
    states = [x0]
    modes = [ap.state]

    if extended_states:
        xd, u, Nz, ps, Ny_r = controlled_f16(times[-1], states[-1], ap, model_str, v2_integrators)

        xd_list = [xd]
        Nz_list = [Nz]
        ps_list = [ps]
        u_list = [u]
        Ny_r_list = [Ny_r]

    der_func = lambda t, y: controlled_f16(t, y, ap, model_str, v2_integrators)[0]

    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)

    while integrator.status == 'running':
        integrator.step()

        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()

            while integrator.t >= times[-1] + step:
                t = times[-1] + step
                #print(f"{round(t, 2)} / {tmax}")

                times.append(t)
                states.append(dense_output(t))

                updated = ap.advance_discrete_state(times[-1], states[-1])
                modes.append(ap.state)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = controlled_f16(times[-1], states[-1], ap, model_str, v2_integrators)

                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    u_list.append(u)

                    Ny_r_list.append(Ny_r)
                    xd_list.append(xd)

                if updated:
                    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    break

    assert integrator.status == 'finished'
                
    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes

    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        res['u_list'] = u_list

    res['runtime'] = time.perf_counter() - start

    return res
