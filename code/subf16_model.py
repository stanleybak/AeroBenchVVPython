'''
Stanley Bak
Python F-16 subf16
outputs aircraft state vector deriative
'''

#         x(1) = air speed, VT    (ft/sec)
#         x(2) = angle of attack, alpha  (rad)
#         x(3) = angle of sideslip, beta (rad)
#         x(4) = roll angle, phi  (rad)
#         x(5) = pitch angle, theta  (rad)
#         x(6) = yaw angle, psi  (rad)
#         x(7) = roll rate, P  (rad/sec)
#         x(8) = pitch rate, Q  (rad/sec)
#         x(9) = yaw rate, R  (rad/sec)
#         x(10) = northward horizontal displacement, pn  (feet)
#         x(11) = eastward horizontal displacement, pe  (feet)
#         x(12) = altitude, h  (feet)
#         x(13) = engine thrust dynamics lag state, pow
#
#         u(1) = throttle command  0.0 < u(1) < 1.0
#         u(2) = elevator command in degrees
#         u(3) = aileron command in degrees
#         u(4) = rudder command in degrees
#

from math import sin, cos, pi

from adc import adc
from tgear import tgear
from pdot import pdot
from thrust import thrust
from cx import cx
from cy import cy
from cz import cz
from cl import cl
from dlda import dlda
from dldr import dldr
from cm import cm
from cn import cn
from dnda import dnda
from dndr import dndr
from dampp import dampp

from Morellif16 import Morellif16

def subf16_model(x, u, model, adjust_cy=True):
    '''output aircraft state vector derivative for a given input

    if use_stevens is False, will use Morelli model (polynomials) for part of the computation
    '''

    assert model == 'stevens' or model == 'morelli'
    assert len(x) == 13
    assert len(u) == 4

    thtlc, el, ail, rdr = u

    # The following is the c.g. location which can be modified (nominal is xcg=.35)

    xcg = .35

    s = 300
    b = 30
    cbar = 11.32
    rm = 1.57e-3
    xcgr = .35
    he = 160.0
    c1 = -.770
    c2 = .02755
    c3 = 1.055e-4
    c4 = 1.642e-6
    c5 = .9604
    c6 = 1.759e-2
    c7 = 1.792e-5
    c8 = -.7336
    c9 = 1.587e-5
    rtod = 57.29578
    g = 32.17

    xd = x.copy()
    vt = x[0]
    alpha = x[1]*rtod
    beta = x[2]*rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    power = x[12]

    amach, qbar = adc(vt, alt)
    cpow = tgear(thtlc)

    xd[12] = pdot(power, cpow)

    t = thrust(power, alt, amach)
    dail = ail/20
    drdr = rdr/30

    if model == 'stevens':
        # stevens & lewis (look up table version)
        cxt = cx(alpha, el)
        cyt = cy(beta, ail, rdr)
        czt = cz(alpha, beta, el)

        clt = cl(alpha, beta) + dlda(alpha, beta) * dail + dldr(alpha, beta) * drdr
        cmt = cm(alpha, el)
        cnt = cn(alpha, beta) + dnda(alpha, beta) * dail + dndr(alpha, beta) * drdr
    else:
        # morelli model (polynomial version)
        cxt, cyt, czt, clt, cmt, cnt = Morellif16(alpha*pi/180, beta*pi/180, el*pi/180, ail*pi/180, rdr*pi/180, \
                                                  p, q, r, cbar, b, vt, xcg, xcgr)

    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt

    d = dampp(alpha)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b
    cbta = cos(x[2])
    u = vt * cos(x[1]) * cbta
    v = vt * sin(x[2])
    w = vt * sin(x[1]) * cbta
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    spsi = sin(psi)
    cpsi = cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs
    gcth = g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
    vdot = p * w-r * u + gcth * sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cbta/dum
    xd[3] = p + (sth/cth) * (qsph + r * cph)
    xd[4] = q * cph-r * sph
    xd[5] = (qsph + r * cph)/cth

    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth-cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi-t1
    s8 = cph * cth
    xd[9] = u * s1 + v * s3 + w * s6
    xd[10] = u * s2 + v * s4 + w * s7
    xd[11] = u * sth-v * s5-w * s8
    xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az-xa * xd[7]           # moves normal accel in front of c.g.

    ####################################
    ###### peter additionls below ######
    if adjust_cy:
        ay = ay+xa*xd[8]           # moves side accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
    Ny = ay / g

    return xd, Nz, Ny, az, ay
