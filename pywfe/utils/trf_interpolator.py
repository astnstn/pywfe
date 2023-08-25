"""
Transfer Function Interpolator
------

This module contains the functionality to interpolate transfer functions
from pywfe.Model object.

"""

import numpy as np


def passed_function(f_arr, func, *args, **kwargs):

    q = func(f_arr, *args, **kwargs)

    return q


def interpolator(f_arr, q0, func, n_pass=4, tolerance=5, *args, **kwargs):

    print("!!")

    def interp():

        dim_array = np.ones((1, q_c.ndim), int).ravel()
        dim_array[0] = -1

        f_i_c = (f_i - f_c[:-1]).reshape(dim_array)

        q_i = q_c[:-1, ...] + ((f_i_c)/df) * \
            (q_c[1:, ...] - q_c[:-1, ...])

        return q_i

    # expected length of frequency array after n passes
    target_len_f = (len(f_arr) - 1) * 2**n_pass + 1
    mean_error = 100

    f_c = f_arr
    q_c = q0

    print("entered interpolator")

    # main loop:

    for n in range(n_pass):

        print(f"=====pass {n}=====")

        print(f"len f_c: {len(f_c)}")

        # create interpolation frequency array
        f_i = (f_c[1:] + f_c[: -1])/2

        print(f"len f_i: {len(f_i)}")

        # find df
        df = f_c[-1] - f_c[-2]
        # find q_i
        print("interpolating")
        q_i = interp()

        print(f"mean q_i: {np.mean(q_i)}")

        # find q_ci
        print("calculating interpolated points")
        print(f"args: {args}, kwargs: {kwargs}")
        print(f"func: {func}")

        if mean_error > tolerance:
            print('solving via calculation')
            q_ci = func(f_i, *args, **kwargs)

        else:
            print('solving via interpolation')
            q_ci = interp()

        print(f"mean q_ci: {np.mean(q_ci)}")

        # evaluate error
        error = np.abs(q_i - q_ci)/np.abs(q_ci)
        mean_error = np.mean(error[error > 0])*100
        
        if np.isnan(mean_error):
            mean_error = 0

        print(f"mean error {mean_error}")

        # ---if error less than tolerance
        # ------interpolation loop

        # remake f_c and q_c
        # Concatenate frequency arrays
        f_c = np.concatenate((f_c, f_i))

        # Concatenate corresponding arrays along the zeroth axis
        q_c = np.concatenate((q_c, q_ci), axis=0)

        # Get the sorting indices for the frequency array
        sort_index = np.argsort(f_c)

        # Sort the frequency array
        f_c = f_c[sort_index]

        # Sort the corresponding array along the same indices
        q_c = q_c[sort_index]

    return f_c, q_c


def _interpolated_transfer_function(model, x_r, force, frequencies, dofs='all', ns=8, tolerance=5):

    # frequencies to be calculated
    f_c = frequencies[:: ns]

    # frequencies to be interpolated
    f_i = (f_c[1:] + f_c[: -1])/2

    # frequency spacing
    df = f_c[-1] - f_c[-2]

    # calculate first actual and interpolated solutions
    q_c = model.transfer_function(
        x_r, force, f_c, dofs=dofs)  # actual solution 1
    # interpolated between q_c
    q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df)*(q_c[..., 1:] - q_c[..., :-1])
    # actual solution to interpolated points
    q_ci = model.transfer_function(x_r, force, f_i, dofs=dofs)

    # merging frequency and displacement arrays
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

            # q_i = q_c[..., :-1] + ((f_i - f_c[:-1])/df) *
            # (q_c[..., 1:] - q_c[..., :-1])

            f_c = np.concatenate((f_c, f_i))
            q_c = np.concatenate((q_c, q_i), axis=-1)

            sort_index = np.argsort(f_c)
            f_c = f_c[sort_index]
            q_c = q_c[..., sort_index]

        print("finished")

        return q_c

    print("finished")

    return q_c
