import numpy as np
import matplotlib.pyplot as plt
import pywfe


def euler_wavenumber(f):
    # wavenumber of euler bernoulli beam
    return np.sqrt(2*np.pi*f/a)


def transfer_velocity(f, x):
    # transfer velocity for beam x > 0
    k = euler_wavenumber(f)
    omega = 2*np.pi*f

    return -omega/(4*E*I*k**3) * (1j*np.exp(-k*x) - np.exp(-1j*k*x))

# %%


E = 2.1e11  # young mod
rho = 7850  # density
h = 0.1  # bean cross section side length length
A = h**2  # beam cross sectional area
I = h**4 / 12  # second moment of area

a = np.sqrt(E*I/(rho*A))  # factor in dispersion relation


f_max = 1e3  # maximum frequency
lambda_min = 2*np.pi/euler_wavenumber(f_max)  # mimimum wavelength
l_max = lambda_min / 10  # unit cell length max - 10 unit cells per wavelength

l = np.round(l_max, decimals=1)  # rounded unit cell length chosen


# stiffness matrix
K = E*I/(l**3) * np.array([

    [12,    6*l,    -12,   6*l],
    [6*l, 4*l**2, -6*l, 2*l**2],
    [-12,   -6*l,    12,  -6*l],
    [6*l, 2*l**2, -6*l, 4*l**2]

])

# mass matrix
M = rho*A*l/420 * np.array([

    [156,   22*l,      54,    -13*l],
    [22*l,  4*l**2,  13*l,  -3*l**2],
    [54,    13*l,     156,    -22*l],
    [-13*l, -3*l**2,  -22*l, 4*l**2]

])


dof = {'node': [0, 0, 1, 1],
       'fieldvar': ['y', 'phi']*2,
       'coord': [
                [0, 0, l, l]
]
}

# crea
beam_model = pywfe.Model(K, M, dof)

# create frequency array
f_arr = np.linspace(10, f_max, 100)

# calculate the wfe wavenumbers
k_wfe = beam_model.dispersion_relation(f_arr)

plt.plot(f_arr, euler_wavenumber(f_arr), '.', color='red', label='analytical')
plt.plot(f_arr, k_wfe, color='black')

plt.legend(loc='best')
plt.xlabel("frequency (Hz)")
plt.ylabel("wavenumber (1/m)")


# %%

beam_model.force[0] = 1

x_r = 0

w = beam_model.transfer_function(f_arr, x_r=x_r, dofs=[0], derivative=1)

plt.semilogy(f_arr, abs(transfer_velocity(f_arr, x_r)),
             '.', color='red', label='analytical')
plt.semilogy(f_arr, abs(w), color='black', label='WFE')

plt.legend(loc='best')
plt.xlabel("frequency (Hz)")
plt.ylabel("abs(mobility) (m/(Ns)")
