'''
Stanley Bak
Python F-16 GCAS
cx
'''

import numpy as np

from aerobench.util import fix, sign

def cx(alpha, el):
    'cx definition'

    a = np.array([[-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166], \
        [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167], \
        [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138], \
        [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091], \
        [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = el / 12
    m = fix(s)
    if m <= -2:
        m = -1

    if m >= 2:
        m = 1

    de = s - m
    n = m + fix(1.1 * sign(de))
    k = k + 3
    l = l + 3
    m = m + 3
    n = n + 3
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    cxx = v + (w - v) * abs(de)

    return cxx
