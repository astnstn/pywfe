"""
Pipe fluid structure energy distribution
----------------------------------------

For axisymmetric model
"""
import numpy as np


def calculate_power_distribution(model, phi, k, f_arr, rho_f=1000, fluid_fieldvar=["p"], struc_fieldvar=["u", "v", "w"], threshold=0.5):

    # split phi into acoustic and structural dofs

    fluid_dofs = model.select_dofs(fieldvar=fluid_fieldvar)
    struc_dofs = model.select_dofs(fieldvar=struc_fieldvar)

    fluid_inds = model.selection_index(fluid_dofs)
    struc_inds = model.selection_index(struc_dofs)

    # displacement and force are the top and bottom half of the eigenvector
    disp = phi[:, :model.N//2, :]
    force = phi[:, model.N//2:, :]

    # # split the eigenvectors into fluid and structural parts
    disp_f = disp[:, fluid_inds]
    disp_s = disp[:, struc_inds]

    # print(force.shape)
    # print(disp.shape)

    force_f = force[:, fluid_inds]
    force_s = force[:, struc_inds]

    # calculate structural power

    v = 1j*2*np.pi*f_arr[:, None, None]*disp_s

    struc_power_dof = 0.5*np.real(np.conj(force_s) * v)
    struc_power = np.sum(struc_power_dof, axis=1)

    # calculate acoustic power

    dp_dx = -1j*k[:, None, :]*disp_f

    rho = rho_f
    phi_vx = (-1/(1j*2*np.pi*f_arr[:, None, None]*rho))*dp_dx

    phi_Ix = 0.5*np.real(disp_f * np.conj(phi_vx))

    r_coord = model.select_dofs(fieldvar='p')['coord'][1]

    fluid_power = np.trapz(
        phi_Ix*2*np.pi*r_coord[None, :, None], x=r_coord, axis=1)

    fluid_power[abs(k.imag) > threshold] = 0
    struc_power[abs(k.imag) > threshold] = 0

    return fluid_power, struc_power


# def calculate_power_distribution2(model, f_arr, x=0)
