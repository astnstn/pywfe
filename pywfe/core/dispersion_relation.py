"""
Dispersion Relation
-------------------

This module contains the functionality needed to solve the dispersion relation
of the a WFE model.
"""
import numpy as np
from pywfe.core import eigensolvers
from pywfe.core import classify_modes


def wavenumber(f, DSM, delta, direction="both", solver="transfer_matrix"):

    (eigenvalues,
     right_eigenvectors,
     left_eigenvectors) = eigensolvers.solver[solver](DSM)

    if direction == "both":

        k = -np.log(eigenvalues)/(1j*delta)

        return k

    else:

        (positive_eigenvalues, negative_eigenvalues,
         positive_right_eigenvectors,
         negative_right_eigenvectors,
         positive_left_eigenvectors,
         negative_left_eigenvectors) = classify_modes.sort_eigensolution(f,
                                                                         eigenvalues,
                                                                         right_eigenvectors,
                                                                         left_eigenvectors)
        if direction == "positive":

            k = -np.log(positive_eigenvalues)/(1j*delta)

            return k

        elif direction == "negative":

            k = -np.log(negative_eigenvalues)/(1j*delta)

            return k

        else:
            raise Exception(
                "invalid direction, choose positive, negative or both")
