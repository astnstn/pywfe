"""
model
------------

This module contains the Model class which serves an API for most required
WFE functionality.
"""

import logging
import numpy as np
from tqdm import tqdm
from pywfe.types import Boundaries
from pywfe.core import model_setup
from pywfe.core import eigensolvers
from pywfe.core import boundary_conditions
from pywfe.core.classify_modes import sort_eigensolution
from pywfe.core.forced_problem import calculate_excited_amplitudes
from pywfe.core.forced_problem import calculate_propagated_amplitudes
from pywfe.core.forced_problem import calculate_modal_displacements
from pywfe.core.forced_problem import calculate_modal_forces
from pywfe.core.forced_problem import generate_reflection_matrices
from pywfe.utils.frequency_sweep import frequency_sweep
from pywfe.utils.forcer import Forcer
import copy

# solver dictionary which contains all the forms of the eigenproblem
solver = eigensolvers.solver
conditions = boundary_conditions.conditions


def handle_iterable_xr(func):
    """
    This wrapper is intended to allow the calculation of forced response
    quantities at a range of distances. Handles single x_r or iterable x_r.

    Parameters
    ----------
    func : func
        Forced response function.

    Returns
    -------
    wrapper : func
        Wrapped forced response function.

    """

    def wrapper(self, x_r, f=None):
        if hasattr(x_r, '__iter__') and not isinstance(x_r, str):
            return np.array([func(self, _x_r, f) for _x_r in x_r])
        else:
            return func(self, x_r, f)
    return wrapper


class Model:

    def __init__(self, K, M, dof,
                 null=None, nullf=None,
                 axis=0,
                 logging_level=20, solver="transfer_matrix"):

        K, M = K.astype('complex'), M.astype('complex')

        # chooses what form of the eigenproblem to solve
        self.solver = solver

        # Set up logging

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format=('%(asctime)s %(levelname)-8s'
                                    '[%(filename)s:%(lineno)d] %(message)s'),
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            level=logging_level)

        logging.info("Initalising WFE model")

        unconstrained_dofs = len(K)

        # generate full dof dictionary, set waveguide axis
        dof = model_setup.generate_dof_info(dof, axis)

        # apply boundary coniditions if needed
        if null is not None and nullf is not None:

            logging.info("Applying boundary conditions")

            K, M, dof = model_setup.apply_boundary_conditions(
                K, M, dof, null, nullf)

            logging.info(f"dofs reduced from {unconstrained_dofs} to {len(K)}")

        # order the dofs into left and right faces
        K, M, dof = model_setup.order_system_faces(K, M, dof)

        self.K, self.M, self.dof = K, M, dof

        # substructure the matrices into LL, LR etc
        self.K_sub, self.M_sub = model_setup.substructure_matrices(K, M, dof)

        # node dictionary
        self.node = model_setup.create_node_dict(dof)

        # length of the waveguide
        self.delta = np.max(self.dof['coord'][0])

        # number of left/right dofs
        self.N = int(np.sum(self.dof['face'] == 'L')*2)

        # do some logging for debug purposes
        init_debug = ["Model initialised",
                      f"Segment length: {self.delta}",
                      f"Total ndof: {len(K)}",
                      f"Condensed ndof {self.N}"]

        [logging.info(line) for line in init_debug]

        self.frequency = -1
        self.eigensolution = ()
        self.solution = {}
        self.force = np.zeros((self.N//2), dtype='complex')
        self._previous_force = None  # for recalculating e+/e-
        self.x_e = 0
        self.L = 1

        zero_boundary = np.zeros((self.N//2, self.N//2))
        self.boundaries = Boundaries(*[zero_boundary]*4)

        logging.debug("debugging...")

    def is_same_frequency(self, f):
        """
        Check if this frequency has already been calculated

        Parameters
        ----------
        f : float
            frequency.

        Returns
        -------
        bool
            Whether the given frequency has already been calculated.

        """
        if f is None or f == self.frequency:
            return True
        else:
            return False

    def set_boundary(self, which, condition):

        if which == "right":
            self.boundaries = Boundaries(*conditions[condition](self.N//2),
                                         self.boundaries[2], self.boundaries[3])

        if which == "left":
            self.boundaries = Boundaries(
                self.boundaries[0], self.boundaries[1], *conditions[condition](self.N//2))

    def form_dsm(self, f):
        """
        Forms the DSM of the model at a given frequency

        Parameters
        ----------
        f : float
            frequency at which to form the DSM.

        Returns
        -------
        DSM : ndarray
            (ndof, ndof) sized array of type complex. The condensed DSM.
        """

        DSM = {}

        for key in self.K_sub.keys():
            DSM[key] = self.K_sub[key] - (2*np.pi*f)**2 * self.M_sub[key]

        return DSM['EE'] - DSM['EI'] @ np.linalg.inv(DSM['II']) @ DSM['IE']

    def generate_eigensolution(self, f):
        """
        Generates the sorted eigensolution at a given frequency.
        If frequency is None or the presently calculated frequency,
        then reuse the previously calculated eigensolution.

        Parameters
        ----------
        f : float
            The frequency at which to calculate the eigensolution.

        Returns
        -------
        eigensolution : Eigensolution (namedtuple)
            The sorted eigensolution. The named tuple fields are:
                - lambda_[plus]/[minus] : +/- going eigenvalues
                - phi_[plus]/[minus] : +/- going right eigenvectors
                - psi_[plus]/[minus] : +/- going left eigenvectors

        """
        # determine if the frequency has already been calculated
        if self.is_same_frequency(f):

            return self.eigensolution

        # otherwise calculated the eigensolution
        else:
            self.frequency = f

            DSM = self.form_dsm(f)

            unsorted_solution = solver[self.solver](DSM)

            self.eigensolution = sort_eigensolution(f, *unsorted_solution)

            return self.eigensolution

    def wavenumbers(self, f=None, direction="plus", imag_threshold=None):
        """
        Calculates the wavenumbers of the system at a given frequency

        Parameters
        ----------
        f : float, optional
            Frequency at which to calculated wavenumbers. The default is None.
        direction : str, optional
            Choose positive going or negative going waves. The default is "plus".
        imag_threshold : float, optional
            Imaginary part of wavenumber above which will be set to np.nan.
            The default is None.

        Returns
        -------
        k : ndarray
            The array of wavenumbers at this frequency.
        """
        sol = self.generate_eigensolution(f=f)

        if direction == "plus":
            k = -np.log(sol.lambda_plus)/(1j*self.delta)
        elif direction == "minus":
            k = -np.log(sol.lambda_minus)/(1j*self.delta)
        elif direction == "both" or "all":
            lambdas = np.concatenate((sol.lambda_plus, sol.lambda_minus))
            k = -np.log(lambdas)/(1j*self.delta)

        if imag_threshold:
            k[abs(k.imag) > imag_threshold] = np.nan
        return k

    def dispersion_relation(self, frequency_array, direction='plus',
                            imag_threshold=None):
        """
        Calculate frequency-wavenumber relation

        Parameters
        ----------
        frequency_array : ndarray
            Frequencies to calculate.
        direction : str, optional
            Choose positive going or negative going waves. The default is "plus".
        imag_threshold : float, optional
            Imaginary part of wavenumber above which will be set to np.nan.
            The default is None.

        Returns
        -------
        wavenumbers : ndarray
            (nfreq, n_waves) sized array of type complex.

        """

        k = [self.wavenumbers(f=f,
                              direction=direction,
                              imag_threshold=imag_threshold)
             for f in frequency_array]

        return np.array(k)

    def phase_velocity(self, frequency_array, direction='plus',
                       imag_threshold=None):

        k = [self.wavenumbers(f=f,
                              direction=direction,
                              imag_threshold=imag_threshold)
             for f in frequency_array]

        k = np.array(k)

        return 2*np.pi*frequency_array[:, None]/k

    def excited_amplitudes(self, f=None):
        """
        Find the excited amplitudes subject to a given force and frequency.
        If the solution has already been calculated for the same inputs,
        reuse the old solution.

        Parameters
        ----------
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        e_plus : ndarray
            Positive excited wave amplitudes.
        e_minus : ndarray
            Negative excited wave amplitudes.
        """
        # print("entering super (base) excited amplitudes")
        # print(f"super force: {self.force[82]}")

        # if the force has been unchanged from prior calculations
        is_same_force = np.all(self.force == self._previous_force)

        # if same force and frequency, then we can reuse e_plus/e_minus
        if is_same_force and self.is_same_frequency(f):

            # print("same force and freq")

            # if it hasn't been calculated yet, do so
            if (self.solution.get('e', None) is None or
                    self.solution.get('e_freq', None) != f):

                # print("freshly calculating")

                (e_plus,
                 e_minus) = calculate_excited_amplitudes(self.generate_eigensolution(f),
                                                         self.force)
            # otherwise reuse the solution

            else:
                # print("reusing solution")
                (e_plus, e_minus) = self.solution['e']

        # if there is a different force or frequency involved, needs recalculating
        else:
            # print("freshly recalculating")
            (e_plus,
             e_minus) = calculate_excited_amplitudes(self.generate_eigensolution(f),
                                                     self.force)

        # store result for later use in computations
        # or for analysis
        self.solution['e'] = (e_plus, e_minus)
        self.solution['e_freq'] = f
        self._previous_force = np.copy(self.force)

        return e_plus, e_minus

    def reflection_matrices(self, f=None):
        """
        Generate the reflection matrices of the system for a given frequency.
        If the solution was calculated before, reuse.

        Parameters
        ----------
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        R_right : ndarray
            (ndof, ndof) sized array for the right reflection matrix. Complex.
        R_left : ndarray
            (ndof, ndof) sized array for the left reflection matrix. Complex.
        """
        if self.is_same_frequency(f):

            if self.solution.get('R', None) is None:
                R_right, R_left = generate_reflection_matrices(self.generate_eigensolution(f),
                                                               *self.boundaries)
            else:
                R_right, R_left = self.solution['R']
        else:
            R_right, R_left = generate_reflection_matrices(self.generate_eigensolution(f),
                                                           *self.boundaries)
        self.solution['R'] = R_right, R_left

        return R_right, R_left

    @ handle_iterable_xr
    def propagated_amplitudes(self, x_r, f=None):
        """
        Calculate the propagated and superimposed amplitudes
        for a given distance and frequency.

        Parameters
        ----------
        x_r : float
            Axial response distance.
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        b_plus, b_minus : ndarray
            Positive and negative amplitudes.

        """

        e_plus, e_minus = self.excited_amplitudes(f)
        k_plus = self.wavenumbers(f)
        (R_right,
         R_left) = self.reflection_matrices(f)

        return calculate_propagated_amplitudes(e_plus, e_minus, k_plus,
                                               self.L, R_right, R_left,
                                               x_r, x_e=self.x_e)

    @ handle_iterable_xr
    def modal_displacements(self, x_r, f=None):
        """
        Calculate the modal displacements at a given distance and frequency.
        Each column corresponds to a different wavemode, each row is
        a different degree of freedom.

        Parameters
        ----------
        x_r : float
            Axial response distance.
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        q_j_plus, q_j_minus : ndarray
            The modal displacements for positive and negative going waves.
        """
        b_plus, b_minus = self.propagated_amplitudes(x_r, f)

        return calculate_modal_displacements(self.generate_eigensolution(f),
                                             b_plus, b_minus)

    @ handle_iterable_xr
    def displacements(self, x_r, f=None):
        """
        Calculate the generalised displacements at a given force and distance.

        Parameters
        ----------
        x_r : float
            Axial response distance.
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        displacements : ndarray
            The displacement for each degree of freedom at x_r
        """

        # just sum up the modal displacements over the last axis
        # which means superimposing the modal displacements of each wave
        q_j_plus, q_j_minus = self.modal_displacements(x_r, f=f)

        q_j = q_j_plus + q_j_minus

        return np.sum(q_j, axis=-1)

    @ handle_iterable_xr
    def modal_forces(self, x_r, f=None):
        b_plus, b_minus = self.propagated_amplitudes(x_r, f)

        return calculate_modal_forces(self.generate_eigensolution(f),
                                      b_plus, b_minus)

    @ handle_iterable_xr
    def forces(self, x_r, f=None):

        f_j_plus, f_j_minus = self.modal_forces(x_r, f=f)

        f_j = f_j_plus + f_j_minus

        return np.sum(f_j, axis=-1)

    def frequency_sweep(self, f_arr,
                        x_r=0, quantities=['displacements'], mac=False, dofs='all'):

        return frequency_sweep(self, f_arr, quantities, x_r=x_r, mac=mac, dofs=dofs)

    def transfer_function(self, f_arr, x_r, dofs="all"):

        if dofs == "all":
            dofs = slice(0, self.N//2)

        else:
            dofs = np.array(dofs)

        displacements = []

        # displacements = [self.displacements(x_r, f)[dofs] for f in f_arr]

        for f in tqdm(f_arr):

            displacements.append(self.displacements(x_r, f)[..., dofs])

        return np.squeeze(np.array(displacements))

    def select_dofs(self, fieldvar=None):

        # select the dofs on the left face (the only important ones)

        dofs = self.dof.copy()

        selected_dofs = (dofs['face'] == 'L')

        dofs['coord'] = dofs['coord'][:, selected_dofs]

        for key in ['face', 'fieldvar', 'index', 'node']:
            dofs[key] = dofs[key][selected_dofs]

        if fieldvar is not None:

            selected_dofs = np.isin(dofs['fieldvar'], fieldvar)

            dofs['coord'] = dofs['coord'][:, selected_dofs]

            for key in ['face', 'fieldvar', 'index', 'node']:
                dofs[key] = dofs[key][selected_dofs]

        return dofs

    def selection_index(self, dof):

        return np.where(np.isin(self.dof['index'], dof['index']))[0]

    def see(self):

        self.forcer = Forcer(self)
        self.forcer.select_nodes()

    def couple(self, coupler, coupling_dof):

        return CoupledModel(self, coupler, coupling_dof)


# class CoupledModel(Model):

#     def __init__(self, original_model, coupler, coupling_dof):
#         # Copy attributes from the original model
#         self.__dict__ = copy.deepcopy(original_model.__dict__)

#         # Add or modify any attributes specific to the coupled model
#         self.coupler = coupler
#         self.coupling_dof = coupling_dof

#     def excited_amplitudes(self, f=None):

#         # print("entering modified excited_amplitudes")
#         # print(f"initial inherit force: {self.force[82]}")

#         # put in unit force to get the transfer displacement q0
#         self.force[self.coupling_dof] = 1
#         # print("set self force to unity")
#         # print(f"inherit force: {self.force[82]}")

#         e_plus, e_minus = super().excited_amplitudes(f=f)
#         q_j_plus, q_j_minus = calculate_modal_displacements(self.generate_eigensolution(f),
#                                                             e_plus, e_minus)
#         q_j = q_j_plus + q_j_minus
#         q0 = np.sum(q_j, axis=-1)[self.coupling_dof]

#         # get force from coupler
#         # input force into specified coupling_dof
#         output_force = self.coupler.force(f, q0)
#         # print(f"output_force aquired: {output_force}")

#         self.force[self.coupling_dof] = output_force
#         # print(f"inherit force before modifying previous: {self.force[82]}")
#         # self._previous_force[self.coupling_dof] = 1

#         # print(f"inherit force after: {self.force[82]}")

#         # print(self.force[self.coupling_dof])
#         return super().excited_amplitudes(f=f)

class CoupledModel(Model):

    def __init__(self, original_model, coupler, coupling_dof):
        # Copy attributes from the original model
        self.__dict__ = copy.deepcopy(original_model.__dict__)

        # Add or modify any attributes specific to the coupled model
        self.coupler = coupler
        self.coupling_dof = coupling_dof

    # def excited_amplitudes(self, f=None):

    #     # put in unit force to get the transfer displacement q0
    #     self.force[self.coupling_dof] = 1

    #     e_plus, e_minus = super().excited_amplitudes(f=f)
    #     q_j_plus, q_j_minus = calculate_modal_displacements(self.generate_eigensolution(f),
    #                                                         e_plus, e_minus)
    #     q_j = q_j_plus + q_j_minus
    #     q0 = np.sum(q_j, axis=-1)[self.coupling_dof]

    #     # get force from coupler
    #     # input force into specified coupling_dof
    #     output_force = self.coupler.force(f, q0)

    #     self.force[self.coupling_dof] = output_force

    #     return super().excited_amplitudes(f=f)

    def excited_amplitudes(self, f=None):

        # put in unit force to get the transfer displacement q0
        self.force[self.coupling_dof] = 1

        (e_plus,
         e_minus) = calculate_excited_amplitudes(self.generate_eigensolution(f),
                                                 self.force)

        k_plus = self.wavenumbers(f)
        (R_right,
         R_left) = self.reflection_matrices(f)

        b_plus, b_minus = calculate_propagated_amplitudes(e_plus, e_minus, k_plus,
                                                          self.L, R_right, R_left,
                                                          0, x_e=self.x_e)

        q_j_plus, q_j_minus = calculate_modal_displacements(self.generate_eigensolution(f),
                                                            b_plus, b_minus)

        q_j = q_j_plus + q_j_minus
        q0 = np.sum(q_j, axis=-1)[self.coupling_dof]

        output_force = self.coupler.force(f, q0)
        self.force[self.coupling_dof] = output_force

        (e_plus,
         e_minus) = calculate_excited_amplitudes(self.generate_eigensolution(f),
                                                 self.force)

        return e_plus, e_minus
