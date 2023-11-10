"""
modal_assurance
---------------

This module contains functions for sorting frequency sweept data by mode index

"""
import numpy as np


def mac_matrix(modes_prev, modes_next):
    """Compute the Modal Assurance Criterion (MAC) matrix."""
    nmodes = modes_prev.shape[1]
    assert modes_next.shape[1] == nmodes, "The two mode sets must have the same number of modes"

    mac_matrix = np.empty((nmodes, nmodes), dtype='complex')

    for i in range(nmodes):
        for j in range(nmodes):
            numerator = np.abs(np.vdot(modes_prev[:, i], modes_next[:, j]))**2
            denominator = (np.vdot(modes_prev[:, i], modes_prev[:, i]) *
                           np.vdot(modes_next[:, j], modes_next[:, j]))
            mac_matrix[i, j] = numerator / denominator

    return mac_matrix


def sorting_indices(modes_prev, modes_next):

    matrix = mac_matrix(modes_prev, modes_next)

    # Find sorting order
    sorting_order = np.argmax(matrix, axis=1)

    return sorting_order


def sort_wavenumbers(wavenumbers):

    avg_imag = np.mean(np.abs(wavenumbers.imag), axis=0)
    sorted_indices = np.argsort(avg_imag)

    return sorted_indices
