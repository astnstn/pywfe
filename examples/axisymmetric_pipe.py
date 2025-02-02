import matplotlib.pyplot as plt
import numpy as np
import pywfe
# import wavepype.toolbox as tb
# from scipy.fft import rfft, irfft
# import scienceplots
# from wavepype import plottools
# from study import *

# plt.style.use(['science', 'bright'])


# def remove_jumps(arr, threshold):
#     diff_arr = np.diff(arr, axis=0)
#     jumps = np.abs(diff_arr) > threshold
#     arr[:-1][jumps] = np.nan
#     arr[1:][jumps] = np.nan
#     return arr


# %%
# load in axisymmetric pipe
# steel, water filled
model = pywfe.load("AXISYM_thin_1pc_damping", source='database')

# %% model description (custom metadata)

print(model.description)


# %% show unique x axis variables

print(set(model.dof['coord'][0]))


# %% show unique fieldvariables

print(set(model.dof['fieldvar']))

# %% look at the model

model.see()

# %% dispersion relation

f_arr = np.linspace(10, 10e3, 300)

k = model.dispersion_relation(f_arr)

# %%
plt.subplot(2, 1, 1)
plt.plot(f_arr, k.real, '.')
plt.ylabel('Re(k)')
plt.ylim(0, 50)
plt.subplot(2, 1, 2)
plt.plot(f_arr, k.imag, '.')
plt.ylabel('Im(k)')

plt.ylim(0, -50)


# %%

sweep_result = model.frequency_sweep(
    f_arr, quantities=['wavenumbers', 'phi_plus'])


# %%

phi = sweep_result['phi_plus']

print(phi.shape)

phi_max_f = phi[-1]

print(phi_max_f.shape)

# first axis represents the degrees of freedom.
# the first half of this axis is displacement, the second half is force

# second axis represents the mode number

q_max_f = phi_max_f[:model.N//2]

# %% selecting dof regions by field variable

struc_dof = model.select_dofs(fieldvar=['u', 'w'])
fluid_dof = model.select_dofs(fieldvar='p')

# %%
radial_coord = model.dof['coord'][1]
radial_outer_wall_dof = model.select_dofs(
    fieldvar=['u', 'v'], where=radial_coord == 0.2)


# %%

fluid_dof_indices = model.selection_index(fluid_dof)

p_max_f = q_max_f[fluid_dof_indices]

# %%

plt.plot(fluid_dof['coord'][1], p_max_f[:, [0, 1, 2, 3, 4]])


# %%

sweep_result = model.frequency_sweep(
    f_arr, quantities=['wavenumbers', 'phi_plus'], mac=True)


# %%


# thick = pywfe.load("AXISYM_thick_1pc_damping", source='database')

# f_arr = np.linspace(20, 20e3, 500)

# k_thin = thin.frequency_sweep(f_arr, quantities=['wavenumbers'], mac=True)[
#     'wavenumbers']

# k_thick = thick.frequency_sweep(f_arr, quantities=['wavenumbers'], mac=True)[
#     'wavenumbers']

# # %%

# k_thin = k_thin[:, pywfe.sort_wavenumbers(k_thin)]
# k_thick = k_thick[:, pywfe.sort_wavenumbers(k_thick)]

# k_thin[:, [0, 1]] = k_thin[:, [1, 0]]

# # %%

# k_thin[abs(k_thin.imag) > 0.6] = np.nan
# k_thick[abs(k_thick.imag) > 0.6] = np.nan

# # %%

# a_thin = -20*np.log10(np.e)*k_thin.imag
# a_thick = -20*np.log10(np.e)*k_thick.imag

# # %%
# threshold = 0.3
# for i in range(10):
#     a_thin[:, i] = remove_jumps(a_thin[:, i], threshold)
#     a_thick[:, i] = remove_jumps(a_thick[:, i], threshold)
# # %%
# plt.plot(f_arr/1e3, a_thick[:, :10],
#          color='grey', alpha=0.5, linewidth=0.6)
# plt.plot(f_arr/1e3, a_thin[:, :10])


# plt.xlabel("Frequency (kHz)")
# plt.ylabel("Attenuation (dB/m)")

# plottools.frac_max_width(0.8)
# plottools.limits()
# plottools.trim_width()

# plt.savefig("attenuation_thin.pdf")

# # %%
# plt.figure()

# plt.plot(f_arr/1e3, a_thin[:, :10])
# plt.gca().set_prop_cycle(None)
# plt.plot(f_arr/1e3, a_thick[:, :10],
#          alpha=0.6, linewidth=1, linestyle='--')


# plt.xlabel("Frequency (kHz)")
# plt.ylabel("Attenuation (dB/m)")

# plottools.frac_max_width(0.8)
# plottools.limits()
# plottools.trim_width()

# plt.savefig("attenuation.pdf")
