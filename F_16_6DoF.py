# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:59:00 2020

@author: johnv
"""

import pandas as pd
import numpy as np
from numpy import sin, cos, pi
from math import nan
from sys import exit

# In[F-16 parametrisation]

# hardcoded values
g, rho, S, cbar = 9.81, 1.225, 1., 2.

# initialise states
U_init, V_init, W_init = 1., 1., 1.
p_init, q_init, r_init = 0., 0., 0.
psi_init, theta_init, phi_init = 0., 0., 0.
xe_init, ye_init, ze_init = 0., 0., 0.

# initial conditions
theta, phi, psi, x1dot, y1dot, z1dot, x2dot, y2dot, z2dot = 0.05, 0., 0., 10., 0., 0., 0., 0., 0.

# inertial
Ixx, Iyy, Izz, mass = 1., 1., 1., 1.
Iyz, Izx, Ixy, Ixz = 1., 1., 1., 1.

# simulation
time_step, time_start, time_end, g = 0.001, 0., 10., 9.81

# wrap for input
initial_state_vector = [U_init, V_init, W_init, p_init, q_init, r_init, psi_init, theta_init, phi_init, xe_init, ye_init, ze_init]
initial_conditions = [theta, phi, psi, x1dot, y1dot, z1dot, x2dot, y2dot, z2dot]
aircraft_properties = [Ixx, Iyy, Izz, mass, Iyz, Izx, Ixy, Ixz]
simulation_parameters = [time_step, time_start, time_end, g]

### F-16 parameter values for global nonlinear model ###

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
    
    
    # # undefined ones set to 0 to allow for pandas dataframe
    # a7, a8, a9 = nan, nan, nan
    # b5, b6, b7, b8, b9 = nan, nan, nan, nan, nan
    # c3, c4, c5, c6, c7, c8, c9 = nan, nan, nan, nan, nan, nan, nan
    # d4, d5, d6, d7, d8, d9 = nan, nan, nan, nan, nan, nan
    # e4, e5, e6, e7, e8, e9 = nan, nan, nan, nan, nan, nan
    # f6, f7, f8, f9 = nan, nan, nan, nan
    # g5, g6, g7, g8, g9 = nan, nan, nan, nan, nan
    # h8, h9 = nan, nan
    # i4, i5, i6, i7, i8, i9 = nan, nan, nan, nan, nan, nan
    # j5, j6, j7, j8, j9 = nan, nan, nan, nan, nan
    # k7, k8, k9 = nan, nan, nan
    # l7, l8, l9 = nan, nan, nan
    # n6, n7, n8, n9 = nan, nan, nan, nan
    # m8, m9 = nan, nan
    # o7, o8, o9 = nan, nan, nan
    # p5, p6, p7, p8, p9 = nan, nan, nan, nan, nan
    # q3, q4, q5, q6, q7, q8, q9 = nan, nan, nan, nan, nan, nan, nan
    # s6, s7, s8, s9 = nan, nan, nan, nan
    
    
    
    
    # # initialise data of lists
    # data = {'a':[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9],
    #         'b':[b0, b1, b2, b3, b4, b5, b6, b7, b8, b9],
    #         'c':[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9],
    #         'd':[d0, d1, d2, d3, d4, d5, d6, d7, d8, d9],
    #         'e':[e0, e1, e2, e3, e4, e5, e6, e7, e8, e9],
    #         'f':[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
    #         'g':[g0, g1, g2, g3, g4, g5, g6, g7, g8, g9],
    #         'h':[h0, h1, h2, h3, h4, h5, h6, h7, h8, h9],
    #         'i':[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9],
    #         'j':[j0, j1, j2, j3, j4, j5, j6, j7, j8, j9],
    #         'k':[k0, k1, k2, k3, k4, k5, k6, k7, k8, k9],
    #         'l':[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9],
    #         'm':[m0, m1, m2, m3, m4, m5, m6, m7, m8, m9],
    #         'n':[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9],
    #         'o':[o0, o1, o2, o3, o4, o5, o6, o7, o8, o9],
    #         'p':[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9],
    #         'q':[q0, q1, q2, q3, q4, q5, q6, q7, q8, q9],
    #         'r':[r0, r1, r2, r3, r4, r5, r6, r7, r8, r9],
    #         's':[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]}
    
    # df = pd.DataFrame(data, index=[0,1,2,3,4,5,6,7,8,9])
    
    
    
    return(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)

nonlinear_parameters = global_nonlinear_model_parameters()

# In[Coefficient functions]

#a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = nonlinear_parameters

def Cx_calc(alpha, de, a):
    Cx = a[0] + a[1]*alpha + a[2]*de**2 + a[3]*de + a[4]*alpha*de + a[5]*alpha**2 + a[6]*alpha**3
    return(Cx)
    
def Cxq_calc(alpha, b):
    Cxq = b[0] + b[1]*alpha + b[2]*alpha**2 + b[3]*alpha**3 + b[4]*alpha**4
    return(Cxq)
    
def Cy_calc(beta, da, dr, c):
    Cy = c[0]*beta + c[1]*da + c[2]*dr
    return(Cy)
    
def Cyp_calc(alpha, d):
    Cyp = d[0] + d[1]*alpha + d[2]*alpha**2 + d[3]*alpha**3
    return(Cyp)
    
def Cyr_calc(alpha, e):
    Cyr = e[0] + e[1]*alpha + e[2]*alpha**2 + e[3]*alpha**3
    return(Cyr)
    
def Cz_calc(alpha, beta, de, f):
    Cz = (f[0] + f[1]*alpha + f[2]*alpha**2 + f[3]*alpha**3 + f[4]*alpha**4)*(1-beta**2)
    return(Cz)
    
def Czq_calc(alpha, g):
    Czq = g[0] + g[1]*alpha + g[2]*alpha**2 + g[3]*alpha**3 + g[4]*alpha**4
    return(Czq)
    
def Cl_calc(alpha, beta, h):
    Cl = h[0]*beta + h[1]*alpha*beta + h[2]*beta*alpha**2 + h[3]*beta**2 + h[4]*alpha*beta**2 + h[5]*beta*alpha**3 + h[6]*beta*alpha**4 + h[7]*(alpha**2)*(beta**2)
    return(Cl)
        
def Clp_calc(alpha, i):
    Clp = i[0] + i[1]*alpha + i[2]*alpha**2 + i[3]*alpha**3
    return(Clp)
    
def Clr_calc(alpha, j):
    Clr = j[0] + j[1]*alpha + j[2]*alpha**2 + j[3]*alpha**3 + j[4]*alpha**4
    return(Clr)
    
def Clda_calc(alpha, beta, k):
    Clda = k[0] + k[1]*alpha + k[2]*beta + k[3]*alpha**2 + k[4]*alpha*beta + k[5]*beta*alpha**2 + k[6]*alpha**3
    return(Clda)
    
def Cldr_calc(alpha, beta, l):
    Cldr = l[0] + l[1]*alpha + l[2]*beta + l[3]*alpha*beta + l[4]*beta*alpha**2 + l[5]*beta*alpha**3 + l[6]*beta**2
    return(Cldr)
    
def Cm_calc(alpha, de, m):
    Cm = m[0] + m[1]*alpha + m[2]*de + m[3]*alpha*de + m[4]*de**2 + m[5]*de*alpha**2 + m[6]*de**3 + m[7]*alpha*de**2
    return(Cm)
    
def Cmq_calc(alpha, n):
    Cmq = n[0] + n[1]*alpha + n[2]*alpha**2 + n[3]*alpha**3 + n[4]*alpha**4 + n[5]*alpha**5
    return(Cmq)
    
def Cn_calc(alpha, beta, o):
    Cn = o[0]*beta + o[1]*alpha*beta + o[2]*beta**2 + o[3]*alpha*beta**2 + o[4]*beta*alpha**2 + o[5]*(alpha**2)*(beta**2) + o[6]*beta*alpha**3
    return(Cn)
    
def Cnp_calc(alpha, p):
    Cnp = p[0] + p[1]*alpha + p[2]*alpha**2 + p[3]*alpha**3 + p[4]*alpha**4
    return(Cnp)
    
def Cnr_calc(alpha, q):
    Cnr = q[0] + q[1]*alpha + q[2]*alpha**2
    return(Cnr)
    
def Cnda_calc(alpha, beta, r):
    Cnda = r[0] + r[1]*alpha + r[2]*beta + r[3]*alpha*beta + r[4]*beta*alpha**2 + r[5]*beta*alpha**3 + r[6]*alpha**2 + r[7]*alpha**3 + r[8]*beta**3 + r[9]*alpha*beta**3
    return Cnda
    
def Cndr_calc(alpha, beta, s):
    Cndr = s[0] + s[1]*alpha + s[2]*beta + s[3]*alpha*beta + s[4]*beta*alpha**2 + s[5]*alpha**2
    return(Cndr)



# In[F-16 Class creation and initialisation]

class F_16:
    
    def __init__(self, initial_state_vector):
        
        self.U = initial_state_vector[0]
        self.V = initial_state_vector[1]
        self.W = initial_state_vector[2]
        self.p = initial_state_vector[3]
        self.q = initial_state_vector[4]
        self.r = initial_state_vector[5]
        self.psi = initial_state_vector[6]
        self.theta = initial_state_vector[7]
        self.phi = initial_state_vector[8]
        self.xe = initial_state_vector[9]
        self.ye = initial_state_vector[10]
        self.ze = initial_state_vector[11]
    
    def sim(self, aircraft_properties, initial_conditions, nonlinear_parameters, simulation_parameters):
        
        Ixx, Iyy, Izz, mass, Iyz, Izx, Ixy, Ixz = aircraft_properties
        time_step, time_start, time_end, grav = simulation_parameters
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = nonlinear_parameters
        
        rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))
        X, Y, Z, L, M, N, Cx, Cy, Cz, Cl, Cm, Cn = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        x, x1dot, x2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        y, y1dot, y2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        z, z1dot, z2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        theta, theta1dot, theta2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        phi, phi1dot, phi2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        psi, psi1dot, psi2dot = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        
        theta[0], phi[0], psi[0], x1dot[0], y1dot[0], z1dot[0], x2dot[0], y2dot[0], z2dot[0] = initial_conditions
        
        # test values
        span = 10.
        V = 100.
        ref_chord = 4.
        xcgref = 10.
        xcg = 9.
        de, da, dr = 0., 0., 0.
        
        for idx, t in enumerate(rng):
            #print(idx)
            
            pbar = self.p*span/(2*V)
            qbar = self.q*ref_chord/(2*V)
            rbar = self.r*span/(2*V)
            
            # self.q =/= q      
            ''' Coefficients '''
            Cx[idx] = Cx_calc(theta[idx], de, a) + Cxq_calc(theta[idx], b)*qbar
            Cy[idx] = Cy_calc(psi[idx], da, dr, c) + Cyp_calc(theta[idx], d)*pbar + Cyr_calc(theta[idx], e)*rbar
            Cz[idx] = Cz_calc(theta[idx], psi[idx], de, f) + Czq_calc(theta[idx], g)*qbar
            Cl[idx] = Cl_calc(theta[idx], psi[idx], h) + Clp_calc(theta[idx], i)*pbar + Clr_calc(theta[idx], j)*rbar + Clda_calc(theta[idx], psi[idx], k)*da + Cldr_calc(theta[idx], psi[idx], l)*dr
            Cm[idx] = Cm_calc(theta[idx], de, m) + Cmq_calc(theta[idx], n)*qbar + Cz_calc(theta[idx], psi[idx], de, f)*(xcgref - xcg)
            Cn[idx] = Cn_calc(theta[idx], psi[idx], o) + Cnp_calc(theta[idx], p)*pbar + Cnr_calc(theta[idx], q)*rbar + Cnda_calc(theta[idx], psi[idx], r)*da + Cndr_calc(theta[idx], psi[idx], s)*dr - Cy_calc(psi[idx], da, dr, c)*(xcgref-xcg)
            
            print(Cx[idx])
            
            ''' Forces/Moments '''
            # Fore/Aft (likely correct)
            X[idx] = 0.5*rho*x1dot[idx]**2*span*Cx[idx]
            Y[idx] = 0.5*rho*x1dot[idx]**2*span*Cy[idx]
            Z[idx] = 0.5*rho*x1dot[idx]**2*span*Cz[idx]
            
            
            
            # Rolls (not sure)
            M[idx] = 0.5*rho*x1dot[idx]**2*span*cbar*Cm[idx]
            N[idx] = 0.5*rho*x1dot[idx]**2*span*Cn[idx]
            L[idx] = 0.5*rho*x1dot[idx]**2*span*Cl[idx]
            
            ''' Accelerations linear/Rotational '''
            x2dot[idx] = X[idx]/mass
            y2dot[idx] = Y[idx]/mass
            z2dot[idx] = Z[idx]/mass
            theta2dot[idx] = M[idx]/Iyy
            phi2dot[idx] = L[idx]/Ixx
            psi2dot[idx] = N[idx]/Izz
            
            
            
            if idx < 9999:
                ''' Velocities '''
                x1dot[idx+1] = x1dot[idx] + x2dot[idx]*time_step
                y1dot[idx+1] = y1dot[idx] + y2dot[idx]*time_step
                z1dot[idx+1] = z1dot[idx] + z2dot[idx]*time_step
                theta1dot[idx+1] = theta1dot[idx] + theta2dot[idx]*time_step
                phi1dot[idx+1] = phi1dot[idx] + phi2dot[idx]*time_step
                psi1dot[idx+1] = psi1dot[idx] + psi2dot[idx]*time_step
            
                ''' Translations'''
                # body axes linear translation, must be translated to earth axes
                x[idx+1] = x[idx] + x1dot[idx]*time_step + 0.5*x2dot[idx]*time_step**2
                y[idx+1] = y[idx] + y1dot[idx]*time_step + 0.5*y2dot[idx]*time_step**2
                z[idx+1] = z[idx] + z1dot[idx]*time_step + 0.5*z2dot[idx]*time_step**2
            
                # angular translation
                theta[idx+1] = theta[idx] + theta1dot[idx]*time_step + 0.5*theta2dot[idx]*time_step**2
                phi[idx+1] = phi[idx] + phi1dot[idx]*time_step + 0.5*phi2dot[idx]*time_step**2
                psi[idx+1] = psi[idx] + psi1dot[idx]*time_step + 0.5*psi2dot[idx]*time_step**2
            
            else:
                pass
            
        return(x, y, z, theta, phi, psi)
        
# initialise instance of F_16 class w/initial conditions
f16 = F_16(initial_state_vector)

# run simulation with inertial values of a true f16
x, y, z, theta, phi, psi = f16.sim(aircraft_properties, initial_conditions, nonlinear_parameters, simulation_parameters)
# In[]       
''' Testing 

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))
X, Y, Z, L, M, N, Cx, Cy, Cz, Cl, Cm, Cn = np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))

        
# test values
span = 10.
V = 100.
ref_chord = 4.
xcgref = 10.
xcg = 9.
de, da, dr = 0., 0., 0.

ptest, qtest, rtest = 0., 0., 0.
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = nonlinear_parameters
test_list = []

for idx, t in enumerate(rng):
    #print(idx)
            
    pbar = ptest*span/(2*V)
    qbar = qtest*ref_chord/(2*V)
    rbar = rtest*span/(2*V)
            
    # self.q =/= q
    #print(Cx_calc(alpha, de, a))
            

    Cx[idx] = Cx_calc(alpha, de, a) + Cxq_calc(alpha, b)*qbar
    print(Cx[idx])
    test_list.append(Cx[idx])
    print(idx)
    Cy[idx] = Cy_calc(beta, da, dr, c) + Cyp_calc(alpha, d)*pbar + Cyr_calc(alpha, e)*rbar
    print(Cx[idx])
    print(idx)
    Cz[idx] = Cz_calc(alpha, beta, de, f) + Czq_calc(alpha, g)*qbar
    Cl[idx] = Cl_calc(alpha, beta, h) + Clp_calc(alpha, i)*pbar + Clr_calc(alpha, j)*rbar + Clda_calc(alpha, beta, k)*da + Cldr_calc(alpha, beta, l)*dr
    Cm[idx] = Cm_calc(alpha, de, m) + Cmq_calc(alpha, n)*qbar + Cz_calc(alpha, beta, de, f)*(xcgref - xcg)
    Cn[idx] = Cn_calc(alpha, beta, o) + Cnp_calc(alpha, p)*pbar + Cnr_calc(alpha, q)*rbar + Cnda_calc(alpha, beta, r)*da + Cndr_calc(alpha, beta, s)*dr - Cy_calc(beta, da, dr, c)*(xcgref-xcg)
    #exit()
            
    # Fore/Aft
    X[idx] = 0.5*rho*V**2*span*Cx[idx]
    Y[idx] = 0.5*rho*V**2*span*Cy[idx]
    Z[idx] = 0.5*rho*V**2*span*Cz[idx]
            
    M[idx] = 0.5*rho*V**2*span*cbar*Cm[idx]
    N[idx] = 0.5*rho*V**2*span*Cn[idx]
    L[idx] = 0.5*rho*V**2*span*Cl[idx]
    
'''