import numpy as np
import matplotlib.pyplot as plt
import analytical
import pywfe
from pywfe.core.classify_modes import classify_wavemode

from wavepype.wfemodel import WFEModel
A = analytical.A  # area of rod
J = analytical.I  # second moment of area

rho = analytical.rho  # density of steel
E = analytical.E  # youngs modulus of steel

mu = rho*A  # linear mass density

f_max = 1e3  # maximum frequency
lambda_min = 2*np.pi/analytical.euler_an(f_max)  # mimimum wavelength
l_max = lambda_min / 10  # unit cell length max

l = np.round(l_max, decimals=1)  # rounded unit cell length chosen

m = mu*l  # mass of unit cell

# stiffness matrix
K = E*J/(l**3) * np.array([

    [12,    6*l,    -12,   6*l],
    [6*l, 4*l**2, -6*l, 2*l**2],
    [-12,   -6*l,    12,  -6*l],
    [6*l, 2*l**2, -6*l, 4*l**2]

])

# mass matrix
M = m/420 * np.array([

    [156,   22*l,      54,    -13*l],
    [22*l,  4*l**2,  13*l,  -3*l**2],
    [54,    13*l,     156,    -22*l],
    [-13*l, -3*l**2,  -22*l, 4*l**2]

])

dof = {'node': [0, 0, 1, 1],
       'fieldvar': ['y', 'phi']*2,
       'coord': [
                [0, 0, l, l],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
]
}

x_r = np.linspace(0, 50, 3000)

f_arr = np.linspace(10, f_max, 2)

# model = pywfe.Model(K, M, dof, logging_level=20, solver = 'polynomial')
model = pywfe.load_comsol("data_block")

forcer = pywfe.Forcer(model)
# forcer.select_nodes()
nodes = [0, 2, 3, 6]
[forcer.add_nodal_force(node, {'v': 1}) for node in [0, 2, 3, 6]]

output = model.frequency_sweep(f_arr, x_r=x_r, quantities=[
    'displacements', 'forces'])
# %%
q = output['displacements'][1, :, :]
f = output['forces'][1, :, :]

v = 2*np.pi*1j*f_arr[-1, None]*q

P = 0.5 * np.real(np.conj(f) * v)

P = np.sum(P, axis=-1)


plt.plot(x_r, P[:])
