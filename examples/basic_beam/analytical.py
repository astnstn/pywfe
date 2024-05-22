import numpy as np
import matplotlib.pyplot as plt

E = 2.1e11  # young mod
rho = 7850  # density
h = 0.1  # cube length
A = h**2  # beam cross sectional area
I = h**4 / 12  # second moment of area
nu = 0.3  # poisson ratio
G = E/(2*(1 + nu))  # shear mod

a = np.sqrt(E*I/(rho*A))


def euler_an(f): return np.sqrt(2*np.pi*f/a)


def long_an(f): return 2*np.pi*f / np.sqrt(E/7850)


def shear_an(f): return 2*np.pi*f / np.sqrt(G/7850)


def mobility(f, x):

    k = euler_an(f)
    omega = 2*np.pi*f

    return -omega/(4*E*I*k**3) * (1j*np.exp(-k*x) - np.exp(-1j*k*x))
