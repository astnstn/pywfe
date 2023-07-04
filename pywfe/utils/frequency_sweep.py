"""
Frequency Sweep
------

This module contains the  fucntion for calculating various quantities over
an array of frequencies for a pywfe.Model object.

"""
import numpy as np
from pywfe.utils.modal_assurance import sorting_indices
from tqdm import tqdm


def frequency_sweep(model, f_arr, quantities, x_r=0, mac=False,
                    imag_threshold=None, dofs='all'):

    if dofs == "all":
        dofs = slice(0, model.N//2)

    else:
        dofs = np.array(dofs)

    output = {quantity: [] for quantity in quantities}

    if mac:
        phi_previous = model.generate_eigensolution(f_arr[0]).phi_plus
    else:
        inds = np.arange(model.N//2)

    for i in tqdm(range(len(f_arr))):

        if mac:
            phi_next = model.generate_eigensolution(f_arr[i]).phi_plus
            inds = sorting_indices(phi_previous, phi_next)
            phi_previous = phi_next[:, inds]

        # print(inds)
        for quantity in quantities:

            if quantity == "excited_amplitudes":

                e_plus = np.array(
                    model.excited_amplitudes(f_arr[i]))[..., 0, inds]

                output["excited_amplitudes"].append(e_plus)

            if quantity == "propagated_amplitudes":

                b_plus = np.array(model.propagated_amplitudes(
                    x_r, f_arr[i]))[..., 0, inds]

                output["propagated_amplitudes"].append(b_plus)

            if quantity == "modal_displacements":

                q_j_plus = np.array(model.modal_displacements(
                    x_r, f_arr[i]))[..., 0, :, :]

                dims = len(q_j_plus.shape) - 1
                index_expansion = [np.newaxis] * dims + [slice(None)]

                q_j_plus = np.take_along_axis(q_j_plus,
                                              inds[tuple(index_expansion)],
                                              axis=-1)

                q_j_plus = q_j_plus[..., dofs, :]

                output["modal_displacements"].append(q_j_plus)

            if quantity == "wavenumbers":

                k_plus = model.wavenumbers(f_arr[i])[inds]

                output["wavenumbers"].append(k_plus)

            if quantity == "displacements":

                q = model.displacements(x_r, f_arr[i])[..., dofs]

                output["displacements"].append(q)

            if quantity == "forces":

                D = model.form_dsm(f_arr[i])
                D_LL = D[:model.N//2, :model.N//2]

                # print(D_LL.shape, q.shape)

                if q is not None:
                    # Ensure v is a 2D array
                    q = np.atleast_2d(q)

                    # Perform matrix multiplication
                    F = D_LL @ q.T
                    F = F.T  # Transpose back to original shape

                    output["forces"].append(F)

    for key in output.keys():
        output[key] = np.squeeze(np.array(output[key]))

    return output
