'''
Stanley Bak
RunF16Sim python version
'''

import numpy as np

from scipy.integrate import RK45
from controlledF16 import controlledF16

def RunF16Sim(initialState, tMax, der_func, F16Model, ap, llc, pass_fail, sim_step=0.01, multipliers=None):
    'Simulates and analyzes autonomous F-16 maneuvers'

    # append integral error states to state vector
    initialState = np.array(initialState, dtype=float)
    x0 = np.zeros((initialState.shape[0] + llc.get_num_integrators() + ap.get_num_integrators(),))
    x0[:initialState.shape[0]] = initialState

    # run the numerical simulation
    times = [0]
    states = [x0]
    modes = [ap.state]

    _, u, Nz, ps, _ = controlledF16(times[-1], states[-1], F16Model, ap, llc, multipliers=multipliers)
    Nz_list = [Nz]
    ps_list = [ps]
    u_list = [u]

    rk45 = RK45(der_func, times[-1], states[-1], tMax)

    while rk45.status == 'running':
        rk45.step()

        if rk45.t > times[-1] + sim_step:
            dense_output = rk45.dense_output()

            while rk45.t > times[-1] + sim_step:
                t = times[-1] + sim_step
                times.append(t)
                states.append(dense_output(t))

                updated = ap.advance_discrete_state(times[-1], states[-1])
                modes.append(ap.state)

                # re-run dynamics function at current state to get non-state variables
                xd, u, Nz, ps, Ny_r = controlledF16(times[-1], states[-1], F16Model, ap, llc, multipliers=multipliers)
                pass_fail.advance(times[-1], states[-1], ap.state, xd, u, Nz, ps, Ny_r)
                Nz_list.append(Nz)
                ps_list.append(ps)
                u_list.append(u)

                if updated:
                    rk45 = RK45(der_func, times[-1], states[-1], tMax)
                    break

                if pass_fail.break_on_error and not pass_fail.result():
                    break

        if pass_fail.break_on_error and not pass_fail.result():
            break

    result = pass_fail.result()

    # make sure the solver didn't fail
    if rk45.status != 'finished':
        result = False # fail

    return result, times, states, modes, ps_list, Nz_list, u_list
