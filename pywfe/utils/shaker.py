# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:24:28 2023

@author: Austen
"""
import numpy as np
# k = 42992  # spring constant
# m = 1.21  # moving mass
# zeta = 0.05  # damping ratio


# BL = 5.6  # voice coil coefficient
# R = 3.5 + 1.7  # coil + internal resistance
# L = 0.298e-3  # coil inductance


class Shaker:

    def __init__(self, k=42992, m=1.21, zeta=0.05, Bl=5.6, R=5.2, L=0.298e-3, V_ext=1):

        c_c = 2*np.sqrt(k*m)
        self.c_viscous = c_c*zeta  # damping ratio

        self.k = k
        self.m = m
        self.Bl = Bl
        self.R = R
        self.L = L

        self.V_ext = V_ext

    def newtons_per_amp(self, f):
        omega = 2*np.pi*f

        return -1j*omega*self.m*(self.Bl)/(1j*omega*self.m + self.c_viscous + self.k/(1j*omega))

    def volts_per_amp(self, f):
        omega = 2*np.pi*f
        return (self.R + 1j*omega*self.L) + ((self.Bl)**2)/(1j*omega*self.m + self.c_viscous + (self.k/(1j*omega)))

    def newtons_per_volt(self, f):

        return self.newtons_per_amp(f)/self.volts_per_amp(f)

    def accelerance(self, f):

        omega = 2*np.pi*f

        a = (-omega**2)/((-omega**2)*self.m + 1j*omega*self.c_viscous + self.k)

        return a

    def force(self, f, q0):

        accelerance_wfe = (2*np.pi*1j*f)**2 * q0
        accelerance_shaker = self.accelerance(f)

        force_per_volt = self.newtons_per_volt(f)

        fwfe_per_fshaker = accelerance_shaker / \
            (accelerance_shaker + accelerance_wfe)

        force_pipe = fwfe_per_fshaker*force_per_volt*self.V_ext

        return force_pipe
