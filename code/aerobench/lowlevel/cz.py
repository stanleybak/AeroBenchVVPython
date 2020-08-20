'''
Stanley Bak
Python F-16 GCAS
Cz function
'''

import numpy as np
from aerobench.util import fix, sign

def cz(alpha, beta, el):
    'cz function'

    a = np.array([.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229], \
        dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    l = l + 3
    k = k + 3
    s = a[k-1] + abs(da) * (a[l-1] - a[k-1])

    return s * (1 - (beta / 57.3)**2) - .19 * (el / 25)
