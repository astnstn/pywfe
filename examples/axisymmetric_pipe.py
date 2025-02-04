import matplotlib.pyplot as plt
import numpy as np
import pywfe

# %%
# load in axisymmetric pipe
# steel, water filled
model = pywfe.load("AXISYM_thin_0pt1pc_damping", source='database')

# %% model description (custom metadata)

print(model.description)

# %% show unique fieldvariables

print(set(model.dof['fieldvar']))

# %% look at the model

model.see()

# %% dispersion relation, unsorted wavenumber solutions

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

plt.xlabel('Frequency (Hz)')

plt.ylim(0, -50)

# %%

k_prop = np.copy(k)
k_prop[abs(k.imag) > 0.5] = np.nan

c_p = 2*np.pi*f_arr[:, None]/k_prop

plt.plot(f_arr, c_p, '.')
plt.ylim(0, 8e3)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase velocity (m/s)')

# %% frequency sweep, obtaining multiple parameters

sweep_result = model.frequency_sweep(
    f_arr, quantities=['wavenumbers', 'phi_plus'], mac=True)

# %%

plt.subplot(2, 1, 1)
plt.plot(f_arr, sweep_result['wavenumbers'].real)
plt.ylabel('Re(k)')
plt.ylim(0, 50)
plt.subplot(2, 1, 2)
plt.plot(f_arr, sweep_result['wavenumbers'].imag)
plt.ylabel('Im(k)')

plt.ylim(0, -50)

# %%


# %%

phi = np.copy(sweep_result['phi_plus'])

# (n_freqs, n_dofs, n_modes)
# (first half of n_dofs is displacements, second is forces)
print(phi.shape)


# get just the displacement component of the mode shapes
phi_q = phi[:, :model.N//2, :]
phi_f = phi[:, model.N//2:, :]

# %% selecting dof regions by field variable

struc_dof = model.select_dofs(fieldvar=['u', 'w'])
fluid_dof = model.select_dofs(fieldvar='p')

# %%

fluid_dof_indices = model.dofs_to_indices(fluid_dof)

phi_p = phi_q[:, fluid_dof_indices, :]


# %%

sorted_mode_indices = pywfe.sort_wavenumbers(sweep_result['wavenumbers'])

k_sorted = np.copy(sweep_result['wavenumbers'])[..., sorted_mode_indices]
phi_p_sorted = np.copy(phi_p)[..., sorted_mode_indices]


# %%

radial_coord = fluid_dof['coord'][1]
frequency_index = 10

for mode_index in [0, 1]:

    plt.subplot(2, 1, 1)

    plt.plot(radial_coord, phi_p_sorted[frequency_index, :, mode_index])
    plt.axhline(y=0, color='black', linestyle=':')
    plt.xlabel('radial coordinate (m)')
    plt.ylabel('pressure (arb)')

    plt.subplot(2, 1, 2)

    plt.plot(f_arr, 2*np.pi*f_arr /
             k_sorted[..., mode_index], label=f'{mode_index + 1}')
    plt.axvline(x=f_arr[frequency_index], color='black')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Phase velocity (m/s)')

plt.legend(loc='best')
plt.suptitle(f'Frequency: {f_arr[frequency_index]:.0f} Hz')
plt.tight_layout()
plt.title()

# %%

frequency_index = -1

k_sorted_propagating = np.copy(k_sorted)
k_sorted_propagating[abs(k_sorted.imag) > 0.5] = np.nan

for mode_index in [0, 1, 2, 3, 4]:

    plt.subplot(2, 1, 1)

    plt.plot(radial_coord, phi_p_sorted[frequency_index, :, mode_index])
    plt.axhline(y=0, color='black', linestyle=':')
    plt.xlabel('radial coordinate (m)')
    plt.ylabel('pressure (arb)')

    plt.subplot(2, 1, 2)

    plt.plot(f_arr, 2*np.pi*f_arr /
             k_sorted_propagating[..., mode_index], label=f'{mode_index + 1}')

    plt.axvline(x=f_arr[frequency_index], color='black')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Phase velocity (m/s)')

plt.subplot(2, 1, 2)
plt.ylim(0, 10e3)
plt.legend(loc='best', ncols=5)

# %%

# add a 1 newton radial force to the outer pipe wall
model.force[45] = 1


# %%

# plot the pressure across the the radial coordinate at x=0
excitation_frequency = 15e3

p0 = model.displacements(f=excitation_frequency, x_r=0, dofs=fluid_dof)

plt.plot(radial_coord, p0)
plt.xlabel('radial coordinate (m)')
plt.ylabel('pressure (Pa)')
plt.title(f'frequency: {excitation_frequency} Hz')

# %%

excitation_frequency = 1000

x_arr = np.linspace(0, 100, 1000)

u_x = model.displacements(f=excitation_frequency, x_r=x_arr, dofs=[45])

plt.plot(x_arr, u_x)
plt.xlabel('axial coordinate (m)')
plt.ylabel('displacement (m)')
plt.title(f'frequency: {excitation_frequency} Hz')


# %%

input_mobility = model.transfer_function(f_arr, x_r=0, dofs=[45], derivative=1)

# %%

plt.semilogy(f_arr, abs(input_mobility))
plt.xlabel('Frequency (Hz)')
plt.ylabel('input mobility (m/Ns)')

# %%

excitation_frequency = 4e3
x_arr = np.linspace(0, 2, 400)

p_x = model.displacements(f=excitation_frequency, x_r=x_arr, dofs=fluid_dof)

pywfe.save_as_vtk('pressure field', p_x, x_arr, fluid_dof)
