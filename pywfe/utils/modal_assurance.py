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
    sorting_order = np.argmax(matrix, axis=1)

    # Check for duplicates in the sorting order
    unique, counts = np.unique(sorting_order, return_counts=True)
    duplicates = unique[counts > 1]

    for dup in duplicates:
        # Get the indices for the modes that are duplicated
        duplicated_indices = np.where(sorting_order == dup)[0]
        for i, dup_index in enumerate(duplicated_indices):
            if i == 0:
                continue  # Skip the first one, keep it assigned
            # Set the MAC value for the already used mode to zero
            matrix[:, dup] = 0
            # Find the next highest MAC value for the current mode
            sorting_order[dup_index] = np.argmax(matrix[dup_index, :])

    return sorting_order


def sort_wavenumbers(wavenumbers, imag_threshold=None):
    if imag_threshold is not None:
        # Apply threshold to find propagating wavenumbers (non-NaN)
        # and ensure the real part is non-negative
        is_propagating = (np.abs(wavenumbers.imag) <=
                          imag_threshold) & (wavenumbers.real >= 0)

        # Initialize an array to hold the first propagating index for each mode
        first_propagating_index = np.full(wavenumbers.shape[1], np.inf)

        # For each mode, find the first frequency index where it is considered propagating
        for mode_index in range(wavenumbers.shape[1]):
            propagating_indices = np.where(is_propagating[:, mode_index])[0]
            if propagating_indices.size > 0:
                first_propagating_index[mode_index] = propagating_indices[0]

        # Sort modes by the first propagating index
        sorted_indices = np.argsort(first_propagating_index)

        return sorted_indices

    else:
        # If no threshold is provided, sort by the mean of the absolute imaginary part
        avg_imag = np.mean(np.abs(wavenumbers.imag), axis=0)
        sorted_indices = np.argsort(avg_imag)

        return sorted_indices
