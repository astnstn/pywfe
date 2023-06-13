"""
Frequency Sweep
------

This module contains the  fucntion for calculating various quantities over
an array of frequencies for a pywfe.Model object.

"""
import numpy as np
from pywfe.utils.modal_assurance import mac_matrix, sorting_indices


def frequency_sweep(model, f_arr, quantities, x_r=0, mac=False):

    def function_dictionary(f_arr, x_r):

        functions = {
            # shape (len(f_arr), 2, nmodes)
            'excited_amplitudes': {
                'function': model.excited_amplitudes,
                'args': [f],
            },
            # shape (len(f_arr), 2, nmodes)
            'propagated_amplitudes': {
                'function': model.propagated_amplitudes,
                'args': [x_r, f],
            },
            # shape (len(f_arr), 2, ndofs, nmodes)
            'modal_displacements': {
                'function': model.modal_displacements,
                'args': [x_r, f],
            },
            # shape (len(f_arr), ndofs)
            'displacements': {
                'function': model.displacements,
                'args': [x_r, f]
            },
            # shape (len(f_arr), nmodes)
            'wavenumbers': {
                'function': model.wavenumbers,
                'args': [f]
            },
            'eigensolution': {
                'function': model.generate_eigensolution,
                'args': [f]
            }
        }
        return functions

    output = {quantity: [] for quantity in quantities}

    if mac:
        phi_previous = model.generate_eigensolution(f_arr[0])

    for f in f_arr:

        print(f"solving {f:.2f}")

        for quantity in quantities:

            functions = function_dictionary(f, x_r)

            function = functions[quantity]['function']
            args = functions[quantity]['args']

            calculated_quantity = function(*args)

            output[quantity].append(function(*args))

    # print(output['eigensolution'])

    for key in output.keys():
        if key != 'eigensolution':
            output[key] = np.array(output[key])

    return output
