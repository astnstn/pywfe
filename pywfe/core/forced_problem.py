"""
Forced Problem
--------------

This module contains the functionality needed to apply forces to a WFE model.
"""

import numpy as np
from pywfe.types import Eigensolution, Boundaries


def calculate_excited_amplitudes(eigensolution, force):

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


def calculate_propagated_amplitudes(eigensolution, delta, L, force, boundaries, x_r, x_e=0):

    A_right, B_right, A_left, B_left = boundaries

    sol = eigensolution
    ndof = sol.phi_plus.shape[0]//2

    # Use zeros_like to create zero arrays of the same shape

    e_plus, e_minus = calculate_excited_amplitudes(eigensolution, force)

    R_right, R_left = generate_reflection_matrices(
        eigensolution, A_right, B_right, A_left, B_left)

    # print(R_right, R_left)

    k_plus = -np.log(sol.lambda_plus)/(1j*delta)

    def tau(arg): return np.diag(np.exp(-1j*k_plus*arg))

    X = np.linalg.inv(np.eye(ndof) - tau(x_e) @ R_left @ tau(L - x_e))
    Y = e_plus + tau(x_e) @ R_left @  tau(x_e) @ e_minus

    a_plus = X@Y
    a_minus = tau(L - x_e) @ R_right @ tau(L - x_e) @ a_plus

    b_plus = tau(x_r - x_e) @ a_plus
    b_minus = tau(L - x_r) @ R_right @ tau(L - x_r) @ b_plus

    b_plus[np.isnan(b_plus)] = 0
    b_minus[np.isnan(b_minus)] = 0

    return b_plus, b_minus


def calculate_modal_displacements(eigensolution, delta, L, force, boundaries, x_r, x_e=0):

    sol = eigensolution
    ndof = sol.phi_plus.shape[0]//2

    b_plus, b_minus = calculate_propagated_amplitudes(
        eigensolution, delta, L, force, boundaries, x_r, x_e=x_e)

    q_j_plus = b_plus[None, :]*eigensolution.phi_plus[:ndof]
    q_j_minus = b_minus[None, :]*eigensolution.phi_minus[:ndof]

    return q_j_plus, q_j_minus
