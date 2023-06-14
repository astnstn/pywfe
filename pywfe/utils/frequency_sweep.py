"""
Frequency Sweep
------

This module contains the  fucntion for calculating various quantities over
an array of frequencies for a pywfe.Model object.

"""
import numpy as np
from pywfe.utils.modal_assurance import sorting_indices


def frequency_sweep(model, f_arr, quantities, x_r=0, mac=False):

    output = {quantity: [] for quantity in quantities}

    if mac:
        phi_previous = model.generate_eigensolution(f_arr[0]).phi_plus
    else:
        inds = np.arange(model.N//2)

    for i in range(len(f_arr)):

        if mac:
            phi_next = model.generate_eigensolution(f_arr[i]).phi_plus
            inds = sorting_indices(phi_previous, phi_next)
            phi_previous = phi_next[:, inds]

        # print(inds)
        for quantity in quantities:

            if quantity == "excited_amplitudes":

                # this doesn't work
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

                output["modal_displacements"].append(q_j_plus)

            if quantity == "wavenumbers":

                # this works
                k_plus = model.wavenumbers(f_arr[i])[inds]

                output["wavenumbers"].append(k_plus)

    for key in output.keys():
        output[key] = np.array(output[key])

    return output
