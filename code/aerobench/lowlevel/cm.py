'''
Stanley Bak
F-16 GCAS Python
'''

import numpy as np
from aerobench.util import fix, sign

def cm(alpha, el):
    'cm function'

    a = np.array([[.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192], \
        [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093], \
        [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, .000, -.013, .032], \
        [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006], \
        [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]], dtype=float).T

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

    return v + (w - v) * abs(de)

