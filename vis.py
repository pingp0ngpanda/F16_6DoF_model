#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:42:58 2020

@author: johnviljoen
"""

import matplotlib.pyplot as plt

def traj_plt(x, y, z): #trajectory plot

    fig = plt.figure() # same as MATLAB figure,
    ax = fig.add_subplot(111, projection='3d') # adds a 3D subplot to the figure 
    ax.plot(x, y, z) # plots on this 3D subplot
    
def traj_vec_fld_plt(x, y, z, u, v, w): #trajectory + vectors at each point plot

    fig = plt.figure() # same as MATLAB figure,
    ax = fig.add_subplot(111, projection='3d') # adds a 3D subplot to the figure 
    ax.quiver(x, y, z, u, v, w, length=100, normalize=True) # plots on this 3D subplot