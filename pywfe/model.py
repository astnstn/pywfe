"""
model
------------

This module contains the Model class which serves an API for most required
WFE functionality.
"""

import logging
from collections.abc import Iterable
import numpy as np
from pywfe.types import Eigensolution, Boundaries
from pywfe.core import model_setup
from pywfe.core import eigensolvers
from pywfe.core import boundary_conditions
from pywfe.core.classify_modes import sort_eigensolution
from pywfe.core import dispersion_relation
from pywfe.core.forced_problem import calculate_excited_amplitudes
from pywfe.core.forced_problem import calculate_propagated_amplitudes
from pywfe.core.forced_problem import calculate_modal_displacements

# solver dictionary which contains all the forms of the eigenproblem
solver = eigensolvers.solver
conditions = boundary_conditions.conditions


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
                            level=logging.DEBUG)

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
        self.force = np.zeros((self.N//2))
        self.x_e = 0
        self.L = 1

        zero_boundary = np.zeros((self.N//2, self.N//2))
        self.boundaries = Boundaries(*[zero_boundary]*4)

        logging.debug("debugging...")

    def set_boundary(self, which, condition):

        null = np.zeros((self.N//2, self.N//2))

        if which == "right":
            self.boundaries = Boundaries(*conditions[condition](self.N//2),
                                         null, null)

        if which == "left":
            self.boundaries = Boundaries(
                null, null, *conditions[condition](self.N//2))

    def form_dsm(self, f):
        """
        Forms the DSM of the model at a given frequency

        Parameters
        ----------
        f : float
            frequency at which to form the DSM.

        Returns
        -------
        ndarray
            (ndof, ndof) sized array of type complex.

        """

        DSM = {}

        for key in self.K_sub.keys():
            DSM[key] = self.K_sub[key] - (2*np.pi*f)**2 * self.M_sub[key]

        return DSM['EE'] - DSM['EI'] @ np.linalg.inv(DSM['II']) @ DSM['IE']

    def generate_eigensolution(self, f):

        if f == None or f == self.frequency:

            return self.eigensolution

        else:
            self.frequency = f

            DSM = self.form_dsm(f)

            unsorted_solution = solver[self.solver](DSM)

            sorted_solution = sort_eigensolution(f, *unsorted_solution)

            self.eigensolution = sorted_solution

            return sorted_solution

    def excited_amplitudes(self, f=None):

        return calculate_excited_amplitudes(self.generate_eigensolution(f),
                                            self.force)

    def propagated_amplitudes(self, x_r, f=None):

        sol = self.generate_eigensolution(f=f)

        return calculate_propagated_amplitudes(sol,
                                               self.delta,
                                               self.L,
                                               self.force,
                                               self.boundaries,
                                               x_r,
                                               x_e=self.x_e)

    def modal_displacements(self, x_r, f=None):

        sol = self.generate_eigensolution(f=f)

        return calculate_modal_displacements(sol,
                                             self.delta,
                                             self.L,
                                             self.force,
                                             self.boundaries,
                                             x_r,
                                             x_e=self.x_e)

    def displacements(self, x_r, f=None):

        q_j_plus, q_j_minus = self.modal_displacements(x_r, f=f)

        q_j = q_j_plus + q_j_minus

        return np.sum(q_j, axis=-1)

    def wavenumbers(self, f, direction="both", imag_threshold=1):

        if not isinstance(f, Iterable):

            DSM = self.form_dsm(f)

            k = dispersion_relation.wavenumber(f, DSM, self.delta,
                                               direction=direction,
                                               solver=self.solver)

        else:
            k = []
            for freq in f:

                logging.debug(f"solving frequency {freq}")

                DSM = self.form_dsm(freq)

                k.append(dispersion_relation.wavenumber(freq, DSM, self.delta,
                                                        direction=direction,
                                                        solver=self.solver))

            k = np.array(k)

        k[abs(k.imag) > imag_threshold] = np.nan

        return k