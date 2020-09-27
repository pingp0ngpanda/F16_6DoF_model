# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:59:00 2020

@author: johnv
"""

import numpy as np
from sys import exit
import tqdm #for progress bars
import progressbar
import time

### Import aerodynamic coefficient functions (acf's) as described in the Morelli paper
import acf

# In[F-16 Class creation and initialisation]

class F_16_class:
    
    def __init__(self, initial_state_vector):
        
        self.x = initial_state_vector[0]
        self.x1dot = initial_state_vector[1]
        self.y = initial_state_vector[2]
        self.y1dot = initial_state_vector[3]
        self.z = initial_state_vector[4]
        self.z1dot = initial_state_vector[5]
        self.phi = initial_state_vector[6]
        self.phi1dot = initial_state_vector[7]
        self.theta = initial_state_vector[8]
        self.theta1dot = initial_state_vector[9]
        self.psi = initial_state_vector[10]
        self.psi1dot = initial_state_vector[11]
    
    def sim(self, aircraft_properties, nonlinear_parameters, simulation_parameters):
        
        ''' Extract parameters '''
        Ixx, Iyy, Izz, mass, Iyz, Izx, Ixy, Ixz, mean_aerodynamic_chord,\
            wingspan, wing_area, mean_aerodynamic_chord, wing_area, wingspan,\
            xcgref, xcg, T_max = aircraft_properties
        time_step, time_start, time_end, grav, rho = simulation_parameters
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = nonlinear_parameters
        
        ''' Calculate rng from simulation parameters '''
        rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))
        
        ''' Create variables used for calculations '''
        X, Y, Z, L, M, N, Cx, Cy, Cz, Cl, Cm, Cn = np.zeros(len(rng)),\
            np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)),\
            np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng)),\
            np.zeros(len(rng)), np.zeros(len(rng)), np.zeros(len(rng))
        x2dot = np.zeros(len(rng))
        y2dot = np.zeros(len(rng))
        z2dot = np.zeros(len(rng))
        theta2dot = np.zeros(len(rng))
        phi2dot = np.zeros(len(rng))
        psi2dot = np.zeros(len(rng))
        V = np.zeros(len(rng))
        de = np.zeros(len(rng))
        da = np.zeros(len(rng))
        dr = np.zeros(len(rng))
        
        ''' Create storage for class attributes '''
        x = np.zeros(len(rng))
        y = np.zeros(len(rng))
        z = np.zeros(len(rng))
        phi = np.zeros(len(rng))
        theta = np.zeros(len(rng))
        psi = np.zeros(len(rng))
        
        bar = progressbar.ProgressBar(maxval=len(rng)).start()
        for idx, t in enumerate(rng):
            #print(idx)
            
            ''' Total velocity (ignores wind currently)'''
            V[idx] = np.sqrt((self.x1dot)**2 + (self.y1dot)**2 + (self.z1dot)**2)
            
            pbar = self.phi1dot*wing_area/(2*V[idx])
            qbar = self.theta1dot*mean_aerodynamic_chord/(2*V[idx])
            rbar = self.psi1dot*wing_area/(2*V[idx])
            
            # self.q =/= q      
            ''' Coefficients '''
            Cx[idx] = acf.Cx_calc(self.theta, de[idx], a) + acf.Cxq_calc(self.theta, b)*qbar
            Cy[idx] = acf.Cy_calc(self.psi, da[idx], dr[idx], c) + acf.Cyp_calc(self.theta, d)*pbar\
                + acf.Cyr_calc(self.theta, e)*rbar
            Cz[idx] = acf.Cz_calc(self.theta, self.psi, de[idx], f) + acf.Czq_calc(self.theta, g)*qbar
            Cl[idx] = acf.Cl_calc(self.theta, self.psi, h) + acf.Clp_calc(self.theta, i)*pbar\
                + acf.Clr_calc(self.theta, j)*rbar + acf.Clda_calc(self.theta, self.psi, k)*da[idx]\
                + acf.Cldr_calc(self.theta, self.psi, l)*dr[idx]
            Cm[idx] = acf.Cm_calc(self.theta, de[idx], m) + acf.Cmq_calc(self.theta, n)*qbar\
                + acf.Cz_calc(self.theta, self.psi, de[idx], f)*(xcgref - xcg)
            Cn[idx] = acf.Cn_calc(self.theta, self.psi, o) + acf.Cnp_calc(self.theta, p)*pbar\
                + acf.Cnr_calc(self.theta, q)*rbar + acf.Cnda_calc(self.theta, self.psi, r)*da[idx]\
                + acf.Cndr_calc(self.theta, self.psi, s)*dr[idx] - acf.Cy_calc(self.psi, da[idx],\
                  dr[idx], c)*(xcgref-xcg)
            
            #print(Cx[idx])
            
            ''' Forces/Moments '''
            # Fore/Aft (likely correct)
            X[idx] = 0.5*rho*self.x1dot**2*wing_area*Cx[idx]
            Y[idx] = 0.5*rho*self.x1dot**2*wing_area*Cy[idx]
            Z[idx] = 0.5*rho*self.x1dot**2*wing_area*Cz[idx]
            
            
            # Rolls (not sure)
            M[idx] = 0.5*rho*self.x1dot**2*wing_area*mean_aerodynamic_chord*Cm[idx]
            N[idx] = 0.5*rho*self.x1dot**2*wing_area*wingspan*Cn[idx]
            L[idx] = 0.5*rho*self.x1dot**2*wing_area*wingspan*Cl[idx]
            
            ''' Accelerations linear/Rotational '''
            x2dot[idx] = X[idx]/mass
            y2dot[idx] = Y[idx]/mass
            z2dot[idx] = Z[idx]/mass
            theta2dot[idx] = M[idx]/Iyy
            phi2dot[idx] = L[idx]/Ixx
            psi2dot[idx] = N[idx]/Izz
            
            if idx < len(rng)-1:
            
                ''' Translations'''
                # body axes linear translation, must be translated to earth axes
                self.x = self.x + self.x1dot*time_step + 0.5*x2dot[idx]*time_step**2
                self.y = self.y + self.y1dot*time_step + 0.5*y2dot[idx]*time_step**2
                self.z = self.z + self.z1dot*time_step + 0.5*z2dot[idx]*time_step**2
            
                # angular translation
                self.theta = self.theta + self.theta1dot*time_step + 0.5*theta2dot[idx]*time_step**2
                self.phi = self.phi + self.phi1dot*time_step + 0.5*phi2dot[idx]*time_step**2
                self.psi = self.psi + self.psi1dot*time_step + 0.5*psi2dot[idx]*time_step**2
            
                ''' Velocities '''
                self.x1dot = self.x1dot + x2dot[idx]*time_step
                self.y1dot = self.y1dot + y2dot[idx]*time_step
                self.z1dot = self.z1dot + z2dot[idx]*time_step
                self.theta1dot = self.theta1dot + theta2dot[idx]*time_step
                self.phi1dot = self.phi1dot + phi2dot[idx]*time_step
                self.psi1dot = self.psi1dot + psi2dot[idx]*time_step
            
            ''' Store class attributes '''
            x[idx], y[idx], z[idx] = self.x, self.y, self.z
            phi[idx], theta[idx], psi[idx] = self.phi, self.theta, self.psi
            
            bar.update(idx)
            
        return(x, y, z, phi, theta, psi)