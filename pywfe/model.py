"""
model
------------

This module contains the Model class which serves an API for most required
WFE functionality.
"""

import logging
from collections.abc import Iterable
import numpy as np
from pywfe.core import model_setup
from pywfe.core import eigensolvers
from pywfe.core.classify_modes import sort_eigensolution
from pywfe.core import dispersion_relation

# solver dictionary which contains all the forms of the eigenproblem
solver = eigensolvers.solver


class Model:

    def __init__(self, K, M, dof,
                 null=None, nullf=None,
                 axis=0,
                 logging_level=20, solver="transfer_matrix"):

        # chooses what form of the eigenproblem to solve
        self.solver = solver

        # Set up logging
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

    def eigensolve(self, f):
        """
        Solves the eigenproblem for the model at a given frequency

        Parameters
        ----------
        f : float
            frequency.

        Returns
        -------
        eigenvalues : ndarray
            The unsorted eigenvalues len(ndof) type complex.
        right_eigenvectors : ndarray
            Unsorted right eigenvectors (ndof, ndof) type complex.
        left_eigenvectors : ndarray
            Unsorted left eigenvectors (ndof, ndof) type complex.

        """

        DSM = self.form_dsm(f)

        (eigenvalues,
         right_eigenvectors,
         left_eigenvectors) = solver[self.solver](DSM)

        return eigenvalues, right_eigenvectors, left_eigenvectors

    def eigensort(self, f):

        eigenvalues, right_eigenvectors, left_eigenvectors = self.eigensolve(f)

        (positive_eigenvalues,
         negative_eigenvalues,
         positive_right_eigenvectors,
         negative_right_eigenvectors,
         positive_left_eigenvectors,
         negative_left_eigenvectors) = sort_eigensolution(f, eigenvalues,
                                                          right_eigenvectors,
                                                          left_eigenvectors)

        return (positive_eigenvalues, negative_eigenvalues,
                positive_right_eigenvectors, negative_right_eigenvectors,
                positive_left_eigenvectors, negative_left_eigenvectors)

    def wavenumbers(self, f, direction="both", imag_threshold=1):

        if not isinstance(f, Iterable):

            DSM = self.form_dsm(f)

            k = dispersion_relation.wavenumber(f, DSM, self.delta,
                                               direction=direction,
                                               solver=self.solver)

        else:
            k = []
            for freq in f:

                DSM = self.form_dsm(freq)

                k.append(dispersion_relation.wavenumber(freq, DSM, self.delta,
                                                        direction=direction,
                                                        solver=self.solver))

            k = np.array(k)

        k[abs(k.imag) > imag_threshold] = np.nan

        return k
