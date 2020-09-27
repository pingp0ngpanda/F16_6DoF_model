#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:24:54 2020

@author: johnviljoen
"""

''' F-16 parameter values, initial_conditions, simulation parameters for global nonlinear model '''
from paras import nonlinear_parameters, initial_state_vector, aircraft_properties, simulation_parameters

''' Import the F16 class created in 'aircraft.py' '''
from aircraft import F_16_class

''' Initialise an instance of the F16 class '''
f16 = F_16_class(initial_state_vector)

''' run simulation with predefined parameters '''
x, y, z, phi, theta, psi = f16.sim(aircraft_properties, nonlinear_parameters, simulation_parameters)


# In[Visualise]

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z)