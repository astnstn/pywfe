"""
Transfer Function Interpolator
------

This module contains the functionality to interpolate transfer functions
from pywfe.Model object.

"""

import numpy as np


def evaluate_error(q_c, q_i):

    error = np.abs(q_i - q_c)/np.abs(q_c)
    mean_error = np.mean(error[error > 0])*100

    return mean_error


def interpolate_vectors(q_c, f_c):

    df = f_c[-1] - f_c[-2]
    f_i = (f_c[1:] + f_c[:-1])/2

    q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df)*(q_c[..., 1:] - q_c[..., :-1])

    return q_i


def interpolated_transfer_function(model, f_c, q_c, n):

    n_passes = 0

    # interpolate frequencies
    df = f_c[-1] - f_c[-2]
    f_i = (f_c[1:] + f_c[:-1])/2

    # interpolate vectors
    q_i = interpolate_vectors(q_c, f_c)

    pass


def _interpolated_transfer_function(model, x_r, force, frequencies, dofs='all', ns=8, tolerance=5):

    # frequencies to be calculated
    f_c = frequencies[::ns]

    # frequencies to be interpolated
    f_i = (f_c[1:] + f_c[:-1])/2

    # frequency spacing
    df = f_c[-1] - f_c[-2]

    # calculate first actual and interpolated solutions
    q_c = model.transfer_function(x_r, force, f_c, dofs=dofs)
    q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df)*(q_c[..., 1:] - q_c[..., :-1])
    q_ci = model.transfer_function(x_r, force, f_i, dofs=dofs)

    f_c = np.concatenate((f_c, f_i))
    q_c = np.concatenate((q_c, q_ci), axis=-1)

    sort_index = np.argsort(f_c)
    f_c = f_c[sort_index]
    q_c = q_c[..., sort_index]

    # calculate error
    error = np.abs(q_i - q_ci)/np.abs(q_ci)
    mean_error = np.mean(error[error > 0])*100

    print(f"initial error: {mean_error:.2f}")

    while len(f_c) != len(frequencies) and mean_error > tolerance:

        f_i = (f_c[1:] + f_c[:-1])/2
        df = f_c[-1] - f_c[-2]

        q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df) * \
            (q_c[..., 1:] - q_c[..., :-1])
        q_ci = model.transfer_function(
            x_r, force, f_i, dofs=dofs)

        f_c = np.concatenate((f_c, f_i))
        q_c = np.concatenate((q_c, q_ci), axis=-1)

        sort_index = np.argsort(f_c)
        f_c = f_c[sort_index]
        q_c = q_c[..., sort_index]

        error = np.abs(q_i - q_ci)/np.abs(q_ci)
        mean_error = np.mean(error[error > 0])*100

        print(f"error: {mean_error:.2f}")

        if mean_error < tolerance:
            print("converged!")
            break

    if mean_error < tolerance:

        N_interpolated = len(frequencies) - len(f_c)

        print(f"no. of interpolated solutions = {N_interpolated}")

        while len(f_c) != len(frequencies):

            print("interpolating remaining")

            print(len(f_c), len(frequencies))

            f_i = (f_c[1:] + f_c[:-1])/2
            df = f_c[-1] - f_c[-2]

            q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df) * \
                (q_c[..., 1:] - q_c[..., :-1])

            f_c = np.concatenate((f_c, f_i))
            q_c = np.concatenate((q_c, q_i), axis=-1)

            sort_index = np.argsort(f_c)
            f_c = f_c[sort_index]
            q_c = q_c[..., sort_index]

        print("finished")

        return q_c

    print("finished")

    return q_c
