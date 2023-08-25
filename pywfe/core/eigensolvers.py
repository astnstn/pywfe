"""
eigensolvers
------------

This module contains different solvers for the WFE eigenproblem.
"""

import numpy as np
import scipy


def transfer_matrix(DSM):
    """
    Classical transfer matrix formulation of the WFE eigenproblem.

    The transfer function is defined as

    .. math::

       \\mathbf{T} = \\begin{bmatrix}
       -D_{LR}^{-1} D_{LL} & D_{LR}^{-1} \\\\
       -D_{RL}+D_{RR} D_{LR}^{-1} D_{LL} & -D_{RR} D_{LR}^{-1}
       \\end{bmatrix}


    which leads to the eigenvalue problem

    .. math::

       T \mathbf{\Phi} = \lambda \mathbf{\Phi}

    The left eigenvectors can be found by considering :math:`\mathbf{T}^{T}`

    Parameters
    ----------
    DSM : (N,N) ndarray (float or complex)
        The dynamic stiffness matrix of the system. 
        NxN array of type float or complex.

    Returns
    -------
    vals : ndarray
        1-D array of length N type complex.
    left_eigenvectors : ndarray
        NxN array of type float or complex.
        Column i is vector corresponding to vals[i]
    right_eigenvectors : ndarray
        NxN array of type float or complex.
        Column i is vector corresponding to vals[i]

    """

    n = len(DSM)

    A = DSM[:n//2, :n//2]
    B = DSM[:n//2, n//2:]
    C = DSM[n//2:, :n//2]
    D = DSM[n//2:, n//2:]

    T = np.zeros_like(DSM)

    T[:n//2, :n//2] = -np.linalg.inv(B)@A
    T[:n//2, n//2:] = np.linalg.inv(B)
    T[n//2:, :n//2] = -C + D@np.linalg.inv(B)@A
    T[n//2:, n//2:] = -D@np.linalg.inv(B)

    eigenvalues, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(
        T, left=True)

    return eigenvalues, right_eigenvectors, left_eigenvectors


def polynomial(DSM):
    """
    [unfinished] Polynomial form of the eigenproblem

    Parameters
    ----------
    DSM : (N,N) ndarray (float or complex)
        The dynamic stiffness matrix of the system. 
        NxN array of type float or complex.

    Returns
    -------
    vals : ndarray
        1-D array of length N type complex.
    left_eigenvectors : ndarray
        NxN array of type float or complex.
        Column i is vector corresponding to vals[i]
    right_eigenvectors : ndarray
        NxN array of type float or complex.
        Column i is vector corresponding to vals[i]

    """
    n = len(DSM)

    D_LL = DSM[:n//2, :n//2]
    D_LR = DSM[:n//2, n//2:]
    D_RL = DSM[n//2:, :n//2]
    D_RR = DSM[n//2:, n//2:]

    B = np.zeros_like(DSM)

    # B[:n//2, :n//2] =
    B[:n//2, n//2:] = np.identity(n//2)
    B[n//2:, :n//2] = -np.linalg.inv(D_LR) @ D_RL
    B[n//2:, n//2:] = -np.linalg.inv(D_LR) @ (D_LL + D_RR)

    # vals, vecs, left_vecs = scipy.linalg.eig(B, left=True)
    vals, vecs = np.linalg.eig(B)

    for i in range(vecs.shape[-1]):

        vecs[n//2:, i] = vecs[n//2:, i]/vals[i]

        vecs[n//2:, i] = (D_LL + vals[i]*D_LR) @ vecs[n//2:, i]
        vecs[:, i] = vecs[:, i]/np.linalg.norm(vecs[:, i])

    return vals, vecs, vecs


solver = {"transfer_matrix": transfer_matrix,
          "polynomial": polynomial}
