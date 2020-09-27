#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:14:06 2020

@author: johnviljoen
"""

''' This is to rotate angles in 3D around 3 axes like euler angles I think,
equations are taken from wikipedia https://en.wikipedia.org/wiki/Rotation_matrix '''

import numpy as np
from numpy import cos, sin

def rot_vec(phi, theta, psi): # rotate a single set of values

    unit_vec = np.matrix([[1],
                          [0],
                          [0]])
    
    if isinstance(phi, np.float64): # detects if a series or single value has been passed to it
        
        uvw = np.zeros([3])
        #print(uvw[1])
        yaw_mat = np.matrix([[cos(psi), -sin(psi), 0.],
                             [sin(psi), cos(psi), 0.],
                             [0., 0., 1.]])

        pitch_mat = np.matrix([[cos(theta), 0, sin(theta)],
                               [0., 1., 0.],
                               [-sin(theta), 0., cos(theta)]])

        roll_mat = np.matrix([[1., 0., 0.],
                              [0, cos(phi), -sin(phi)],
                              [0., sin(phi), cos(phi)]])
        
        uvw[:] = unit_vec.transpose() * yaw_mat*pitch_mat*roll_mat
        
        u = uvw[0]
        v = uvw[1]
        w = uvw[2]
        
        return(u, v, w)
    
    else:
    
        rng = range(len(phi))
        u = np.zeros(len(rng))
        v = np.zeros(len(rng))
        w = np.zeros(len(rng))
        uvw = np.zeros([3, len(rng)])
        
    
        for idx in rng:

            yaw_mat = np.matrix([[cos(psi[idx]), -sin(psi[idx]), 0.],
                                 [sin(psi[idx]), cos(psi[idx]), 0.],
                                 [0., 0., 1.]])

            pitch_mat = np.matrix([[cos(theta[idx]), 0, sin(theta[idx])],
                                   [0., 1., 0.],
                                   [-sin(theta[idx]), 0., cos(theta[idx])]])

            roll_mat = np.matrix([[1., 0., 0.],
                                  [0, cos(phi[idx]), -sin(phi[idx])],
                                  [0., sin(phi[idx]), cos(phi[idx])]])

            uvw[:,idx] = unit_vec.transpose() * yaw_mat*pitch_mat*roll_mat
        
        u = uvw[0,:]
        v = uvw[1,:]
        w = uvw[2,:]
        
        return(u, v, w)
        



