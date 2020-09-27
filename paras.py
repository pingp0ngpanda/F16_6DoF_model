#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:35:53 2020

@author: johnviljoen
"""

# In[initial conditions] 
x_init, x1dot_init = 0., 100.
y_init, y1dot_init = 0., 0.
z_init, z1dot_init = 0., 0.
phi_init, phi1dot_init = 0., 0.
theta_init, theta1dot_init = 0., 0.
psi_init, psi1dot_init = 0., 0.

# In[aircraft properties]
Ixx, Iyy, Izz, mass = 12874.8472, 75673.623, 85552.1125, 1. #m^2kg, kg
Iyz, Izx, Ixy, Ixz = 0., 1331.41323, 0., 1331.41323
T_max = 12150 #kg, max thrust
length = 14.8 #m
height = 4.8 #m
mean_aerodynamic_chord = 3.450336 #m
wing_area = 27.8709 #m^2
wingspan = 9.144 #m
xcgref = 0.35*mean_aerodynamic_chord #m
xcg = xcgref*1.

# In[simulation parameters]
time_step, time_start, time_end, g, rho = 0.001, 0., 10., 9.81, 1.225

# In[F-16 parameter values for global nonlinear model]

def global_nonlinear_model_parameters():
    a0 = -1.943367e-02
    h0 = -1.058583e-01 
    n0 = -5.159153e+00
    a1 = 2.136104e-01 
    h1 = -5.776677e-01 
    n1 = -3.554716e+00
    a2 = -2.903457e-01 
    h2 = -1.672435e-02 
    n2 = -3.598636e+01
    a3 = -3.348641e-03 
    h3 = 1.357256e-01 
    n3 = 2.247355e+02
    a4 = -2.060504e-01 
    h4 = 2.172952e-01 
    n4 = -4.120991e+02
    a5 = 6.988016e-01 
    h5 = 3.464156e+00 
    n5 = 2.411750e+02
    a6 = -9.035381e-01 
    h6 = -2.835451e+00 
    o0 = 2.993363e-01
    b0 = 4.833383e-01 
    h7 = -1.098104e+00 
    o1 = 6.594004e-02
    b1 = 8.644627e+00 
    i0 = -4.126806e-01 
    o2 = -2.003125e-01
    b2 = 1.131098e+01 
    i1 = -1.189974e-01 
    o3 = -6.233977e-02
    b3 = -7.422961e+01 
    i2 = 1.247721e+00 
    o4 = -2.107885e+00
    b4 = 6.075776e+01 
    i3 = -7.391132e-01 
    o5 = 2.141420e+00
    c0 = -1.145916e+00 
    j0 = 6.250437e-02 
    o6 = 8.476901e-01
    c1 = 6.016057e-02 
    j1 = 6.067723e-01 
    p0 = 2.677652e-02
    c2 = 1.642479e-01 
    j2 = -1.101964e+00 
    p1 = -3.298246e-01
    d0 = -1.006733e-01 
    j3 = 9.100087e+00 
    p2 = 1.926178e-01
    d1 = 8.679799e-01 
    j4 = -1.192672e+01 
    p3 = 4.013325e+00
    d2 = 4.260586e+00 
    k0 = -1.463144e-01 
    p4 = -4.404302e+00
    d3 = -6.923267e+00 
    k1 = -4.073901e-02 
    q0 = -3.698756e-01
    e0 = 8.071648e-01 
    k2 = 3.253159e-02 
    q1 = -1.167551e-01
    e1 = 1.189633e-01 
    k3 = 4.851209e-01 
    q2 = -7.641297e-01
    e2 = 4.177702e+00 
    k4 = 2.978850e-01 
    r0 = -3.348717e-02
    e3 = -9.162236e+00 
    k5 = -3.746393e-01 
    r1 = 4.276655e-02
    f0 = -1.378278e-01 
    k6 = -3.213068e-01 
    r2 = 6.573646e-03
    f1 = -4.211369e+00 
    l0 = 2.635729e-02 
    r3 = 3.535831e-01
    f2 = 4.775187e+00 
    l1 = -2.192910e-02 
    r4 = -1.373308e+00
    f3 = -1.026225e+01 
    l2 = -3.152901e-03 
    r5 = 1.237582e+00
    f4 = 8.399763e+00 
    l3 = -5.817803e-02 
    r6 = 2.302543e-01
    f5 = -4.354000e-01 
    l4 = 4.516159e-01 
    r7 = -2.512876e-01
    g0 = -3.054956e+01 
    l5 = -4.928702e-01 
    r8 = 1.588105e-01
    g1 = -4.132305e+01 
    l6 = -1.579864e-02 
    r9 = -5.199526e-01
    g2 = 3.292788e+02 
    m0 = -2.029370e-02 
    s0 = -8.115894e-02
    g3 = -6.848038e+02 
    m1 = 4.660702e-02 
    s1 = -1.156580e-02
    g4 = 4.080244e+02 
    m2 = -6.012308e-01 
    s2 = 2.514167e-02
    m3 = -8.062977e-02 
    s3 = 2.038748e-01
    m4 = 8.320429e-02 
    s4 = -3.337476e-01
    m5 = 5.018538e-01 
    s5 = 1.004297e-01
    m6 = 6.378864e-01
    m7 = 4.226356e-01
    
    a = [a0, a1, a2, a3, a4, a5, a6]
    b = [b0, b1, b2, b3, b4]
    c = [c0, c1, c2]
    d = [d0, d1, d2, d3]
    e = [e0, e1, e2, e3]
    f = [f0, f1, f2, f3, f4, f5]
    g = [g0, g1, g2, g3, g4]
    h = [h0, h1, h2, h3, h4, h5, h6, h7]
    i = [i0, i1, i2, i3]
    j = [j0, j1, j2, j3, j4]
    k = [k0, k1, k2, k3, k4, k5, k6]
    l = [l0, l1, l2, l3, l4, l5, l6]
    m = [m0, m1, m2, m3, m4, m5, m6, m7]
    n = [n0, n1, n2, n3, n4, n5]
    o = [o0, o1, o2, o3, o4, o5, o6]
    p = [p0, p1, p2, p3, p4]
    q = [q0, q1, q2]
    r = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]
    s = [s0, s1, s2, s3, s4, s5]
    
    return(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)

# In[wrap for input]
nonlinear_parameters = global_nonlinear_model_parameters()
initial_state_vector = [x_init, x1dot_init, y_init, y1dot_init, z_init, z1dot_init, phi_init, phi1dot_init, theta_init, theta1dot_init, psi_init, psi1dot_init]
aircraft_properties = [Ixx, Iyy, Izz, mass, Iyz, Izx, Ixy, Ixz, mean_aerodynamic_chord, wingspan, wing_area, mean_aerodynamic_chord, wing_area, wingspan, xcgref, xcg, T_max]
simulation_parameters = [time_step, time_start, time_end, g, rho]