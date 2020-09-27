#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:24:54 2020

@author: johnviljoen
"""

''' F-16 parameter values, initial_conditions, simulation parameters for global nonlinear model '''
from paras import nonlinear_parameters, initial_state_vector, aircraft_properties, simulation_parameters

''' Import the F16 class created in 'aircraft.py' '''
from simulation import F_16_class

''' Initialise an instance of the F16 class '''
f16 = F_16_class(initial_state_vector)

''' run simulation with predefined parameters '''
x, y, z, phi, theta, psi = f16.sim(aircraft_properties, nonlinear_parameters, simulation_parameters)

''' postprocess sim output '''
from post_process import rot_vec
u, v, w = rot_vec(phi, theta, psi)

''' pass processed simulation output to visualiser '''
import vis
#vis.traj_plt(x, y, z)
vis.traj_vec_fld_plt(x, y, z, u, v, w)
