'''
Stanley Bak
F16 GCAS in Python
cn function
'''

import numpy as np
from aerobench.util import fix, sign

def cn(alpha, beta):
    'cn function'

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033], \
        [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057], \
        [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073], \
        [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041], \
        [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013], \
        [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)

    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k-1, m-1]
    u = a[k-1, n-1]

    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)
