"""
frequency_sweep
---------------

This module contains the  fucntion for calculating various quantities over
an array of frequencies for a pywfe.Model object.

"""
import numpy as np
from pywfe.utils.modal_assurance import sorting_indices
from tqdm import tqdm


def frequency_sweep(model, f_arr, quantities, x_r=0, mac=False, dofs='all'):
    """
    Perform a sweep over frequency array, extracting specified quatities
    at each step. Modal assurance criterion can be used to track modes through
    frequency by modeshape similarity

    Parameters
    ----------
    model : pywfe.Model
        The model to perform the sweep with.
    f_arr : np.ndarray float
        frequency array.
    quantities : list of str type
        a list of strings specifying the quantities to be calculated.
        These are:
        - phi_plus: the (positive going) eigenvectors
        - excited_amplitudes: see `pywfe.Model.excited_amplitudes`
        - propagated_amplitudes: see `pywfe.Model.propagated_amplitudes`
        - modal_displacements: see `pywfe.Model.modal_displacements`
        - wavenumbers: see `pywfe.Model.wavenumbers`
        - displacements: see `pywfe.Model.displacements`
        - forces: see `pywfe.Model.forces`
    x_r : float, np.ndarray, optional
        response distance(s). The default is 0.
    mac : bool, optional
        Use the modal assurance criterion to sort waves. The default is False.
    dofs : dofs, optional
        The selected degrees of freedom. See `pywfe.Model.dofs_to_inds`.
        The default is 'all'.

    Returns
    -------
    output : dict
        Dictionary of outputs for specified quantities.

    """

    if dofs == "all":
        dofs = slice(0, model.N//2)

    else:
        dofs = np.array(dofs)

    output = {quantity: [] for quantity in quantities}

    # if modally sorting
    if mac:
        # set the 'previous' positive eigenvector to the first solution
        phi_previous = model.generate_eigensolution(f_arr[0]).phi_plus
    else:
        # else no sorting
        inds = np.arange(model.N//2)

    for i in tqdm(range(len(f_arr))):

        # if modal assurance
        if mac:
            # generate the 'next' eigensolution (current frequency)
            phi_next = model.generate_eigensolution(f_arr[i]).phi_plus
            # get the sorting indices from pywfe.modal_assurance.sorting_indices
            inds = sorting_indices(phi_previous, phi_next)
            # cache the sorted current (to be previous) frequency
            phi_previous = phi_next[:, inds]
        for quantity in quantities:

            if quantity == "phi_plus":

                output['phi_plus'].append(phi_previous)

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

                F = model.forces(x_r, f_arr[i])[..., dofs]

                output['forces'].append(F)

    for key in output.keys():
        output[key] = np.squeeze(np.array(output[key]))

    return output
