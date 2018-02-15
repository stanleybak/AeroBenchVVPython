'''
Stanley Bak
TrimmerFun in python

This program numerically calculates the equilibrium state and control vectors of an F-16 model given
certain parameters.

states:                                              controls:
	x1 = Vt		x4 = phi	x7 = p	  x10 = pn			u1 = throttle
	x2 = alpha	x5 = theta	x8 = q	  x11 = pe			u2 = elevator
	x3 = beta	x6 = psi    x9 = r	  x12 = alt		    u3 = aileron
                                      x13 = pow         u4 = rudder
'''

from math import sin
import numpy as np

from scipy.optimize import minimize

from clf16 import clf16
from adc import adc

def trimmerFun(Xguess, Uguess, orient, inputs, printOn, model='stevens', adjust_cy=True):
    'calculate equilibrium state'

    assert isinstance(Xguess, np.ndarray)
    assert isinstance(Uguess, np.ndarray)
    assert isinstance(inputs, np.ndarray)

    x = Xguess.copy()
    u = Uguess.copy()

    if printOn:
        print '------------------------------------------------------------'
        print 'Running trimmerFun.m'

    # gamma singam rr  pr   tr  phi cphi sphi thetadot coord stab  orient
    const = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1]
    rtod = 57.29577951

    # orient: 'Wings Level (gamma = 0)','Wings Level (gamma <> 0)','Steady Turn','Steady Pull Up'
    const[11] = orient

    # inputs: [Vt, h, gamm, psidot, thetadot]
    x[0] = inputs[0]
    x[11] = inputs[1]

    if orient == 2:
        gamm = inputs[2]
        const[0] = gamm/rtod
        const[1] = sin(const[0])
    elif orient == 3:
        psidot = inputs[3]
        const[4] = psidot/rtod
    elif orient == 4:
        thetadot = inputs[4]
        const[8] = thetadot/rtod

    if orient == 3:
        s = np.zeros(shape=(7,))
        s[0] = u[0]
        s[1] = u[1]
        s[2] = u[2]
        s[3] = u[3]
        s[4] = x[1]
        s[5] = x[3]
        s[6] = x[4]
    else:
        s = np.zeros(shape=(3,))
        s[0] = u[0]
        s[1] = u[1]
        s[2] = x[1]

    maxiter = 1000
    tol = 1e-7
    minimize_tol = 1e-9

    res = minimize(clf16, s, args=(x, u, const, model, adjust_cy), method='Nelder-Mead', tol=minimize_tol, \
                   options={'maxiter': maxiter})

    cost = res.fun

    if printOn:
        print 'Throttle (percent):            {}'.format(u[0])
        print 'Elevator (deg):                {}'.format(u[1])
        print 'Ailerons (deg):                {}'.format(u[2])
        print 'Rudder (deg):                  {}'.format(u[3])
        print 'Angle of Attack (deg):         {}'.format(rtod*x[1])
        print 'Sideslip Angle (deg):          {}'.format(rtod*x[2])
        print 'Pitch Angle (deg):             {}'.format(rtod*x[4])
        print 'Bank Angle (deg):              {}'.format(rtod*x[3])

        amach, qbar = adc(x[0], x[11])
        print 'Dynamic Pressure (psf):        {}'.format(qbar)
        print 'Mach Number:                   {}'.format(amach)

        print ''
        print 'Cost Function:           {}'.format(cost)

    assert cost < tol, "trimmerFun did not converge"

    return x, u
