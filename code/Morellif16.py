'''
Stanley Bak
F16 GCAS in Python

Morelli dynamics (Polynomial interpolation)
'''

def Morellif16(alpha, beta, de, da, dr, p, q, r, cbar, b, V, xcg, xcgref):
    'desc'

    #alpha=max(-10*pi/180,min(45*pi/180,alpha)) # bounds alpha between -10 deg and 45 deg
    #beta = max( - 30 * pi / 180, min(30 * pi / 180, beta)) #bounds beta between -30 deg and 30 deg
    #de = max( - 25 * pi / 180, min(25 * pi / 180, de)) #bounds elevator deflection between -25 deg and 25 deg
    #da = max( - 21.5 * pi / 180, min(21.5 * pi / 180, da)) #bounds aileron deflection between -21.5 deg and 21.5 deg
    #dr = max( - 30 * pi / 180, min(30 * pi / 180, dr)) #bounds rudder deflection between -30 deg and 30 deg

    # xcgref = 0.35
    #reference longitudinal cg position in Morelli f16 model


    phat = p * b / (2 * V)
    qhat = q * cbar / (2 * V)
    rhat = r * b / (2 * V)
    ##
    a0 = -1.943367e-2
    a1 = 2.136104e-1
    a2 = -2.903457e-1
    a3 = -3.348641e-3
    a4 = -2.060504e-1
    a5 = 6.988016e-1
    a6 = -9.035381e-1

    b0 = 4.833383e-1
    b1 = 8.644627
    b2 = 1.131098e1
    b3 = -7.422961e1
    b4 = 6.075776e1

    c0 = -1.145916
    c1 = 6.016057e-2
    c2 = 1.642479e-1

    d0 = -1.006733e-1
    d1 = 8.679799e-1
    d2 = 4.260586
    d3 = -6.923267

    e0 = 8.071648e-1
    e1 = 1.189633e-1
    e2 = 4.177702
    e3 = -9.162236

    f0 = -1.378278e-1
    f1 = -4.211369
    f2 = 4.775187
    f3 = -1.026225e1
    f4 = 8.399763
    f5 = -4.354000e-1

    g0 = -3.054956e1
    g1 = -4.132305e1
    g2 = 3.292788e2
    g3 = -6.848038e2
    g4 = 4.080244e2

    h0 = -1.05853e-1
    h1 = -5.776677e-1
    h2 = -1.672435e-2
    h3 = 1.357256e-1
    h4 = 2.172952e-1
    h5 = 3.464156
    h6 = -2.835451
    h7 = -1.098104

    i0 = -4.126806e-1
    i1 = -1.189974e-1
    i2 = 1.247721
    i3 = -7.391132e-1

    j0 = 6.250437e-2
    j1 = 6.067723e-1
    j2 = -1.101964
    j3 = 9.100087
    j4 = -1.192672e1

    k0 = -1.463144e-1
    k1 = -4.07391e-2
    k2 = 3.253159e-2
    k3 = 4.851209e-1
    k4 = 2.978850e-1
    k5 = -3.746393e-1
    k6 = -3.213068e-1

    l0 = 2.635729e-2
    l1 = -2.192910e-2
    l2 = -3.152901e-3
    l3 = -5.817803e-2
    l4 = 4.516159e-1
    l5 = -4.928702e-1
    l6 = -1.579864e-2

    m0 = -2.029370e-2
    m1 = 4.660702e-2
    m2 = -6.012308e-1
    m3 = -8.062977e-2
    m4 = 8.320429e-2
    m5 = 5.018538e-1
    m6 = 6.378864e-1
    m7 = 4.226356e-1

    n0 = -5.19153
    n1 = -3.554716
    n2 = -3.598636e1
    n3 = 2.247355e2
    n4 = -4.120991e2
    n5 = 2.411750e2

    o0 = 2.993363e-1
    o1 = 6.594004e-2
    o2 = -2.003125e-1
    o3 = -6.233977e-2
    o4 = -2.107885
    o5 = 2.141420
    o6 = 8.476901e-1

    p0 = 2.677652e-2
    p1 = -3.298246e-1
    p2 = 1.926178e-1
    p3 = 4.013325
    p4 = -4.404302

    q0 = -3.698756e-1
    q1 = -1.167551e-1
    q2 = -7.641297e-1

    r0 = -3.348717e-2
    r1 = 4.276655e-2
    r2 = 6.573646e-3
    r3 = 3.535831e-1
    r4 = -1.373308
    r5 = 1.237582
    r6 = 2.302543e-1
    r7 = -2.512876e-1
    r8 = 1.588105e-1
    r9 = -5.199526e-1

    s0 = -8.115894e-2
    s1 = -1.156580e-2
    s2 = 2.514167e-2
    s3 = 2.038748e-1
    s4 = -3.337476e-1
    s5 = 1.004297e-1

    ##
    Cx0 = a0 + a1 * alpha + a2 * de**2 + a3 * de + a4 * alpha * de + a5 * alpha**2 + a6 * alpha**3
    Cxq = b0 + b1 * alpha + b2 * alpha**2 + b3 * alpha**3 + b4 * alpha**4
    Cy0 = c0 * beta + c1 * da + c2 * dr
    Cyp = d0 + d1 * alpha + d2 * alpha**2 + d3 * alpha**3
    Cyr = e0 + e1 * alpha + e2 * alpha**2 + e3 * alpha**3
    Cz0 = (f0 + f1 * alpha + f2 * alpha**2 + f3 * alpha**3 + f4 * alpha**4) * (1 - beta**2) + f5 * de
    Czq = g0 + g1 * alpha + g2 * alpha**2 + g3 * alpha**3 + g4 * alpha**4
    Cl0 = h0 * beta + h1 * alpha * beta + h2 * alpha**2 * beta + h3 * beta**2 + h4 * alpha * beta**2 + h5 * \
        alpha**3 * beta + h6 * alpha**4 * beta + h7 * alpha**2 * beta**2
    Clp = i0 + i1 * alpha + i2 * alpha**2 + i3 * alpha**3
    Clr = j0 + j1 * alpha + j2 * alpha**2 + j3 * alpha**3 + j4 * alpha**4
    Clda = k0 + k1 * alpha + k2 * beta + k3 * alpha**2 + k4 * alpha * beta + k5 * alpha**2 * beta + k6 * alpha**3
    Cldr = l0 + l1 * alpha + l2 * beta + l3 * alpha * beta + l4 * alpha**2 * beta + l5 * alpha**3 * beta + l6 * beta**2
    Cm0 = m0 + m1 * alpha + m2 * de + m3 * alpha * de + m4 * de**2 + m5 * alpha**2 * de + m6 * de**3 + m7 * \
        alpha * de**2


    Cmq = n0 + n1 * alpha + n2 * alpha**2 + n3 * alpha**3 + n4 * alpha**4 + n5 * alpha**5
    Cn0 = o0 * beta + o1 * alpha * beta + o2 * beta**2 + o3 * alpha * beta**2 + o4 * alpha**2 * beta + o5 * \
        alpha**2 * beta**2 + o6 * alpha**3 * beta
    Cnp = p0 + p1 * alpha + p2 * alpha**2 + p3 * alpha**3 + p4 * alpha**4
    Cnr = q0 + q1 * alpha + q2 * alpha**2
    Cnda = r0 + r1 * alpha + r2 * beta + r3 * alpha * beta + r4 * alpha**2 * beta + r5 * alpha**3 * beta + r6 * \
        alpha**2 + r7 * alpha**3 + r8 * beta**3 + r9 * alpha * beta**3
    Cndr = s0 + s1 * alpha + s2 * beta + s3 * alpha * beta + s4 * alpha**2 * beta + s5 * alpha**2
    ##

    Cx = Cx0 + Cxq * qhat
    Cy = Cy0 + Cyp * phat + Cyr * rhat
    Cz = Cz0 + Czq * qhat
    Cl = Cl0 + Clp * phat + Clr * rhat + Clda * da + Cldr * dr
    Cm = Cm0 + Cmq * qhat + Cz * (xcgref - xcg)
    Cn = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da + Cndr * dr - Cy * (xcgref - xcg) * (cbar / b)

    return Cx, Cy, Cz, Cl, Cm, Cn
