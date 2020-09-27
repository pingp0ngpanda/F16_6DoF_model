#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:47:23 2020

@author: johnviljoen
"""

''' The acf.py stands for aerodynamic coefficient functions, as described in the Morelli paper '''

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