"""
forced_problem
--------------

This module contains the functionality needed to apply forces to a WFE model.
"""

import numpy as np
from pywfe.types import Eigensolution, Boundaries


def calculate_excited_amplitudes(eigensolution, force):
    """
    Calculates the directly excited amplitudes subject to a given force
    and modal solution.

    Parameters
    ----------
    eigensolution : namedtuple
        eigensolution.
    force : np.ndarray
        force vector.

    Returns
    -------
    e_plus : np.ndarray
        directly excited modal amplitudes (positive).
    e_minus : np.ndarray
        directly excited modal amplitudes (negative).

    """

    n = eigensolution.phi_plus.shape[-1]

    force = applied_force = np.hstack(
        (np.zeros(len(force), dtype='complex'), force))

    modal_matrix = np.hstack(
        (eigensolution.phi_plus, - eigensolution.phi_minus))

    amplitudes = np.linalg.inv(modal_matrix) @ force

    e_plus = amplitudes[:n]
    e_minus = amplitudes[n:]

    return e_plus, e_minus


def generate_reflection_matrices(eigensolution, A_right, B_right, A_left, B_left):
    """
    Calculates the reflection matrices from boundary matrices.

    Parameters
    ----------
    eigensolution : TYPE
        DESCRIPTION.
    A_right : np.ndarray
        A matrix on the right boundary.
    B_right : np.ndarray
        B matrix on the right boundary.
    A_left : np.ndarray
        A natrix on the left boundary.
    B_left : np.ndarray
        B matrix on the left boundary.

    Returns
    -------
    R_right : np.ndarray
        Right reflection matrix.
    R_left : np.ndarray
        Left reflection matrix.

    """

    sol = eigensolution
    ndof = sol.phi_plus.shape[0]//2

    A_left = A_left if A_left is not None else np.zeros((ndof, ndof))
    B_left = B_left if B_left is not None else np.zeros((ndof, ndof))

    A_right = A_right if A_right is not None else np.zeros((ndof, ndof))
    B_right = B_right if B_right is not None else np.zeros((ndof, ndof))

    try:

        X = -np.linalg.inv((A_right @
                            sol.phi_minus[ndof:] + B_right @ sol.phi_minus[:ndof]))

        Y = A_right @ sol.phi_plus[ndof:] + B_right @ sol.phi_plus[:ndof]

        R_right = X@Y

    except:

        R_right = np.zeros_like(A_right)

    try:
        X = -np.linalg.inv((A_left @
                            sol.phi_minus[ndof:] + B_left @ sol.phi_minus[:ndof]))

        Y = A_left @ sol.phi_plus[ndof:] + B_left @ sol.phi_plus[:ndof]

        R_left = X@Y

    except:

        R_left = np.zeros_like(A_left)

    return R_right, R_left


def calculate_propagated_amplitudes(e_plus, e_minus, k_plus,
                                    L, R_right, R_left, x_r, x_e=0):
    """
    Calculates the ampltiudes of waves after propagation to response point

    Parameters
    ----------
    e_plus : np.ndarray
        positive directly excited amplitudes.
    e_minus : np.ndarray
        negative directly excited amplitudes.
    k_plus : np.ndarray
        wavenumber array.
    L : float
        Length of waveguide.
    R_right : np.ndarray
        Right reflection matrix.
    R_left : np.ndarray
        Left reflection matrix.
    x_r : float, np.ndarray
        Response distance.
    x_e : float, 
        Excitation distance. The default is 0.

    Returns
    -------
    b_plus : np.ndarray
        positive propagated amplitudes.
    b_minus : np.ndarray
        negative propagated amplitudes.

    """

    ndof = len(k_plus)

    def tau(arg): return np.diag(np.exp(-1j*k_plus*arg))

    # X = np.linalg.inv(np.eye(ndof) - tau(x_e) @ R_left @ tau(L - x_e))
    # Y = e_plus + tau(x_e) @ R_left @  tau(x_e) @ e_minus

    # a_plus = X@Y
    # a_minus = tau(L - x_e) @ R_right @ tau(L - x_e) @ a_plus

    a_plus = e_plus

    b_plus = tau(x_r - x_e) @ a_plus
    # b_minus = tau(L - x_r) @ R_right @ tau(L - x_r) @ b_plus

    # the above is temporary, I am saving time not inverting matrices
    # by only considering the infintie case (reflections unfinished)
    # b_plus = e_plus
    b_minus = np.zeros_like(e_plus)

    b_plus[np.isnan(b_plus)] = 0
    b_minus[np.isnan(b_minus)] = 0

    return b_plus, b_minus


def calculate_modal_displacements(eigensolution, b_plus, b_minus):
    """
    Calculates the displacement of each mode (last axis is modal)

    Parameters
    ----------
    eigensolution : namedtuple
        eigensolution.
    b_plus : np.ndarray
        positive propagated amplitudes.
    b_minus : np.ndarray
        negative propagated amplitudes.


    Returns
    -------
    q_j_plus : np.ndarray
        positive going modal displacements.
    q_j_minus : np.ndarray
        negative going modal displacements.

    """

    sol = eigensolution
    ndof = sol.phi_plus.shape[0]//2

    q_j_plus = b_plus[None, :]*eigensolution.phi_plus[:ndof]
    q_j_minus = b_minus[None, :]*eigensolution.phi_minus[:ndof]

    return q_j_plus, q_j_minus


def calculate_modal_forces(eigensolution, b_plus, b_minus):
    """
    Calculates the internal forces of each mode (last axis is modal)

    Parameters
    ----------
    eigensolution : namedtuple
        eigensolution.
    b_plus : np.ndarray
        positive propagated amplitudes.
    b_minus : np.ndarray
        negative propagated amplitudes.

    Returns
    -------
    f_j_plus : np.ndarray
        Positive going modal forces.
    f_j_minus : np.ndarray
        Negative going modal forces.

    """

    ndof = eigensolution.phi_plus.shape[0]//2

    f_j_plus = b_plus[None, :]*eigensolution.phi_plus[ndof:]
    f_j_minus = b_minus[None, :]*eigensolution.phi_minus[ndof:]

    return f_j_plus, f_j_minus
