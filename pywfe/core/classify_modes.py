"""
classify_modes
---------------------

This module contains the functionality needed to sort eigensolutions of the
WFE method into positive and negative going waves.
"""
import logging
import numpy as np
from pywfe.types import Eigensolution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def classify_wavemode(f, eigenvalue, eigenvector, threshold):

    logger.debug(f"eigenvalue size {abs(eigenvalue)}")

    _k = -np.log(eigenvalue)/(1j)
    _k = np.sign(_k.real)

    # if the eigenvalue is above 1 + threshold
    if abs(eigenvalue) > 1 + threshold:

        direction = "left"

    # if the eigenvalue is above 1 - threshold
    elif abs(eigenvalue) < 1 - threshold:

        direction = "right"

    # otherwise evaluate the power flow
    else:
        n = len(eigenvector)

        displacement = eigenvector[:n//2]
        force = eigenvector[n//2:]

        power_flow = np.real(1j * (2*np.pi*f) *
                             np.conj(force.T) @ displacement)

        if power_flow > 0:

            direction = "right"

        elif power_flow < 0:

            direction = "left"

    # print(f"size: {abs(eigenvalue)}, k sign: {_k}, eval: {direction}")
    return direction


def sort_eigensolution(f, eigenvalues, right_eigenvectors, left_eigenvectors):

    # the total counted positive and negative going waves
    positive_count = 0
    negative_count = 0

    N = len(eigenvalues)

    # somewhat arbitrarily chosen inital threshold
    # it finds values close to one and takes their standard deviation
    values_close_to_one = eigenvalues[abs(1 - abs(eigenvalues)) < 0.01]

    mean_offset = np.mean(abs(1 - abs(values_close_to_one)))
    mean_offset = mean_offset/5
    threshold = mean_offset + np.std(abs(1 - abs(values_close_to_one)))/5
    # print(mean_offset, threshold)
    threshold = 1e-8

    # set up empty arrays for the output
    positive_eigenvalues = np.zeros((N//2), dtype='complex')
    negative_eigenvalues = np.zeros_like(positive_eigenvalues)

    positive_right_eigenvectors = np.zeros((N, N//2), dtype='complex')
    negative_right_eigenvectors = np.zeros_like(positive_right_eigenvectors)
    positive_left_eigenvectors = np.zeros_like(positive_right_eigenvectors)
    negative_left_eigenvectors = np.zeros_like(positive_right_eigenvectors)

    for i in range(len(eigenvalues)):

        # classify each wave and add to positive and negative going arrays
        if classify_wavemode(f, eigenvalues[i],
                             right_eigenvectors[:, i], threshold) == "right":

            positive_eigenvalues[positive_count] = eigenvalues[i]
            positive_right_eigenvectors[:,
                                        positive_count] = right_eigenvectors[:, i]
            positive_left_eigenvectors[:,
                                       positive_count] = left_eigenvectors[:, i]

            positive_count += 1

        if classify_wavemode(f, eigenvalues[i],
                             right_eigenvectors[:, i], threshold) == "left":

            negative_eigenvalues[negative_count] = eigenvalues[i]
            negative_right_eigenvectors[:,
                                        negative_count] = right_eigenvectors[:, i]
            negative_left_eigenvectors[:,
                                       negative_count] = left_eigenvectors[:, i]

            negative_count += 1

    return Eigensolution(positive_eigenvalues, negative_eigenvalues,
                         positive_right_eigenvectors, negative_right_eigenvectors,
                         positive_left_eigenvectors, negative_left_eigenvectors)
