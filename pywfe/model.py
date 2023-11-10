import copy
from pywfe.utils.io_utils import save
from pywfe.utils.forcer import Forcer
from pywfe.utils.frequency_sweep import frequency_sweep
from pywfe.core.forced_problem import generate_reflection_matrices
from pywfe.core.forced_problem import calculate_modal_forces
from pywfe.core.forced_problem import calculate_modal_displacements
from pywfe.core.forced_problem import calculate_propagated_amplitudes
from pywfe.core.forced_problem import calculate_excited_amplitudes
from pywfe.core.classify_modes import sort_eigensolution
from pywfe.core import boundary_conditions
from pywfe.core import eigensolvers
from pywfe.core import model_setup
from pywfe.types import Boundaries
from tqdm import tqdm
from functools import wraps
import numpy as np
import logging
"""
model
-----

This module contains the Model class which serves an API for most required
WFE functionality.
"""


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
    @wraps(func)
    # Add **kwargs to accept any number of keyword arguments
    def wrapper(self, x_r, f=None, **kwargs):
        if hasattr(x_r, '__iter__') and not isinstance(x_r, str):
            # Pass **kwargs to the function inside the list comprehension
            return np.array([func(self, _x_r, f, **kwargs) for _x_r in x_r])
        else:
            # Pass **kwargs to the function call
            return func(self, x_r, f, **kwargs)
    return wrapper


class Model:
    """
    The main high level api in the pywfe package.
    """

    def __init__(self, K, M, dof,
                 null=None, nullf=None,
                 axis=0,
                 logging_level=20, solver="transfer_matrix"):
        """
        initialise a Model object

        Parameters
        ----------
        K : np.ndarray
            Stiffness matrix :math:`\mathbf{K}` shape :math:`(N, N)`.
        M : np.ndarray
            Mass matrix :math:`\mathbf{M}` shape :math:`(N, N)`.
        dof : dict
            A dictionary containing the following keys:

            - ``'coord'`` : array of shape :math:`(n_{\\text{{dim}}}, N)`
                Coordinates of the degrees of freedom, where :math:`n_{\\text{{dim}}}` is the number of spatial dimensions and :math:`N` is the total number of degrees of freedom in the initial total mesh.
            - ``'node'`` : array of shape :math:`(N,)`
                Node number that the degree of freedom sits on.
            - ``'fieldvar'`` : array of shape :math:`(N,)`
                Field variable for the degree of freedom (e.g., pressure, displacement in x, displacement in y).
            - ``'index'`` : array of shape :math:`(N,)`
                Index of the degree of freedom, used to keep track of the degrees of freedom when sorted.

        null : ndarray, optional
            Null space constraint matrix (for boundary conditions) shape :math:`(N, N)`. The default is None.
        nullf : ndarray, optional
            Force null space constraint matrix (for boundary conditions) shape :math:`(N, N)`. The default is None.
        axis : int, optional
            The waveguide axis. Moves ``dof['coord'][axis]`` to ``dof['coord'][0]``. The default is 0.
        logging_level : int, optional
            logging level. The default is 20.
        solver : str, optional
            The form of the eigenvalue to use. The default is "transfer_matrix".
            Options are currently ``'transfer_matrix'`` or ``'polynomial'``.

        Returns
        -------
        None.

        """
        self.description = None

        K, M = K.astype('complex'), M.astype('complex')

        # chooses what form of the eigenproblem to solve
        self.solver = solver
        """Description of solver."""

        # Set up logging
        self.logger = logging.getLogger('pywfe')
        self.logger.info("Initalising WFE model")

        unconstrained_dofs = len(K)

        # generate full dof dictionary, set waveguide axis
        dof = model_setup.generate_dof_info(dof, axis)

        # apply boundary coniditions if needed
        if null is not None and nullf is not None:

            self.logger.info("Applying boundary conditions")

            K, M, dof = model_setup.apply_boundary_conditions(
                K, M, dof, null, nullf)

            self.logger.info(
                f"dofs reduced from {unconstrained_dofs} to {len(K)}")

        # order the dofs into left and right faces
        K, M, dof = model_setup.order_system_faces(K, M, dof)

        self.K = K
        """Sorted stiffness matrix"""
        self.M = M
        """Sorted mass matrix"""
        self.dof = dof
        """Sorted dof dictionary"""

        # substructure the matrices into LL, LR etc
        K_sub, M_sub = model_setup.substructure_matrices(K, M, dof)

        self.K_sub = K_sub
        """dictionary containing substructured stiffness matrices ``'LL', 'LR, 'RL', 'RR', 'LI', 'IL', 'RI', 'IR', 'II'``"""
        self.M_sub = M_sub
        """dictionary containing substructured mass matrices."""

        # node dictionary
        self.node = model_setup.create_node_dict(dof)
        """dictionary of node information"""

        # length of the waveguide
        self.delta = np.max(self.dof['coord'][0])
        """Waveguide segment length"""

        # number of left/right dofs
        self.N = int(np.sum(self.dof['face'] == 'L')*2)
        """Number of dofs on both left and right faces combined"""

        # do some self.logger for debug purposes
        init_debug = ["Model initialised",
                      f"Segment length: {self.delta}",
                      f"Total ndof: {len(K)}",
                      f"Condensed ndof {self.N}"]

        [self.logger.info(line) for line in init_debug]

        self.frequency = -1
        self.eigensolution = ()
        """The eigensolution at a given frequency. Gives values and vectors corresponding to propagation constants and mode shapes"""
        self.solution = {}
        self.force = np.zeros((self.N//2), dtype='complex')
        """The force vector corresponding to forces at each dof"""
        self._previous_force = None  # for recalculating e+/e-
        self.x_e = 0
        self.L = 1

        zero_boundary = np.zeros((self.N//2, self.N//2))
        self.boundaries = Boundaries(*[zero_boundary]*4)

    def __repr__(self):

        return f"pywfe.Model(N = {self.N})"

    def dofs_to_indices(self, dofs):
        """


        Parameters
        ----------
        dofs : str or list or dict
            'all' specifies all dofs.
            A list of integers is interpreted as the dof indices.
            A dof dictionary, created with `model.select_dofs()`

        Returns
        -------
        inds : np.ndarray
            array of dof indices.

        """

        if dofs == "all":
            inds = slice(0, self.N//2)

        elif isinstance(dofs, dict):

            inds = self.selection_index(dofs)

        elif isinstance(dofs, list):

            if isinstance(dofs[0], str):
                dofs = self.select_dofs(fieldvar=dofs)
                inds = self.selection_index(dofs)

            else:
                inds = np.array(dofs)

        elif isinstance(dofs, str):
            dofs = self.select_dofs(fieldvar=dofs)
            inds = self.selection_index(dofs)

        return inds

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
        """
        Sets the boundary conditions

        Parameters
        ----------
        which : string
            left or right.
        condition : string
            fixed or free.

        Returns
        -------
        None.

        """

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
        k = []

        for f in tqdm(frequency_array):

            k.append(self.wavenumbers(f=f,
                                      direction=direction,
                                      imag_threshold=imag_threshold))

        return np.array(k)

    def phase_velocity(self, frequency_array, direction='plus',
                       imag_threshold=None):
        """
        gets the phase velocity curves for a given frequency array

        Parameters
        ----------
        frequency_array : np.ndarray
            DESCRIPTION.
        direction : str, optional
            Direction of the waves. The default is 'plus'.
        imag_threshold : float, optional
            Imaginary threshold above which set to np.nan. The default is None.

        Returns
        -------
        ndarray
            phase velocity.

        """

        k = []

        for f in tqdm(frequency_array):

            k.append(self.wavenumbers(f=f,
                                      direction=direction,
                                      imag_threshold=imag_threshold))

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
        # if the force has been unchanged from prior calculations
        is_same_force = np.all(self.force == self._previous_force)

        # if same force and frequency, then we can reuse e_plus/e_minus
        if is_same_force and self.is_same_frequency(f):

            # if it hasn't been calculated yet, do so
            if (self.solution.get('e', None) is None or
                    self.solution.get('e_freq', None) != f):

                (e_plus,
                 e_minus) = calculate_excited_amplitudes(self.generate_eigensolution(f),
                                                         self.force)

            # otherwise reuse the solution
            else:
                (e_plus, e_minus) = self.solution['e']

        # if there is a different force or frequency involved, needs recalculating
        else:
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
    def modal_displacements(self, x_r, f=None, dofs='all'):
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
        dofs = self.dofs_to_indices(dofs)

        b_plus, b_minus = self.propagated_amplitudes(x_r, f)

        q_j_plus, q_j_minus = calculate_modal_displacements(self.generate_eigensolution(f),
                                                            b_plus, b_minus)

        return q_j_plus[dofs], q_j_minus[dofs]

    @ handle_iterable_xr
    def displacements(self, x_r, f=None, dofs='all'):
        """
        gets the displacements for all degrees of freedom at specified x and f.

        Parameters
        ----------
        x_r : float
            response distance (can be array like).
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        ndarray
            displacements for each degree of freedom.

        """

        # just sum up the modal displacements over the last axis
        # which means superimposing the modal displacements of each wave
        q_j_plus, q_j_minus = self.modal_displacements(x_r, f=f, dofs=dofs)

        q_j = q_j_plus + q_j_minus

        return np.sum(q_j, axis=-1)

    @ handle_iterable_xr
    def modal_forces(self, x_r, f=None, dofs='all'):
        """
        Generates the modal forces at given distance and frequency

        Parameters
        ----------
        x_r : float
            Response distance.
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        np.ndarray
            modal force array.

        """
        dofs = self.dofs_to_indices(dofs)

        b_plus, b_minus = self.propagated_amplitudes(x_r, f)

        q_j_plus, q_j_minus = calculate_modal_forces(self.generate_eigensolution(f),
                                                     b_plus, b_minus)

        return q_j_plus[dofs], q_j_minus[dofs]

    @ handle_iterable_xr
    def forces(self, x_r, f=None, dofs='all'):
        """
        Gets the total force on each degree of freedom.

        Parameters
        ----------
        x_r : float
            Response distance.
        f : float, optional
            Frequency. The default is None.

        Returns
        -------
        np.ndarray
            forces.

        """

        f_j_plus, f_j_minus = self.modal_forces(x_r, f=f, dofs=dofs)

        f_j = f_j_plus + f_j_minus

        return np.sum(f_j, axis=-1)

    def frequency_sweep(self, f_arr,
                        x_r=0, quantities=['displacements'], mac=False, dofs='all'):
        """
        Solves various quantities over specified frequency and response range.
        Includes Modal Assurance Critereon (MAC) sorting.

        Parameters
        ----------
        f_arr : np.ndarray
            Array of frequencies.
        x_r : float or np.ndarray, optional
            Response distance. The default is 0.
        quantities : list, optional
            Quantities to solve for. The default is ['displacements'].
        mac : bool, optional
            Whether to sort modal quantities according to MAC. The default is False.
        dofs : list, optional
            Select specific degrees of freedom. The default is 'all'.

        Returns
        -------
        dict
            Dictionary of output for specified quantities.

        """
        return frequency_sweep(self, f_arr, quantities, x_r=x_r, mac=mac, dofs=dofs)

    def transfer_function(self, f_arr, x_r, dofs="all", derivative=0):
        """
        Gets the displacement over frequency at specified distance and dofs.

        Parameters
        ----------
        f_arr : np.ndarray
            Frequency array.
        x_r : float or np.ndarray
            Response distance.
        dofs : list, optional
            List of dofs to return. The default is "all".

        Returns
        -------
        ndarray
            Displacements over frequency and distance.

        """

        dofs = self.dofs_to_indices(dofs)

        displacements = []

        for f in tqdm(f_arr):

            output = ((1j*2*np.pi*f)**derivative) * \
                self.displacements(x_r, f)[..., dofs]

            displacements.append(output)

        return np.squeeze(np.array(displacements))

    def select_dofs(self, fieldvar=None):
        """
        select the model degrees of freedom that correspond to specified 
        field variable.

        Parameters
        ----------
        fieldvar : str or list, optional
            The fieldvariable or list thereof to select for. The default is None.

        Returns
        -------
        dofs : dict
            Reduced dof dictionary.

        """
        dofs = self.left_dofs()

        if fieldvar is not None:

            selected_dofs = np.isin(dofs['fieldvar'], fieldvar)

            dofs['coord'] = dofs['coord'][:, selected_dofs]

            for key in ['face', 'fieldvar', 'index', 'node']:
                dofs[key] = dofs[key][selected_dofs]

        return dofs

    def left_dofs(self):

        dofs = self.dof.copy()

        selected_dofs = (dofs['face'] == 'L')

        dofs['coord'] = dofs['coord'][:, selected_dofs]

        for key in ['face', 'fieldvar', 'index', 'node']:
            dofs[key] = dofs[key][selected_dofs]

        return dofs

    def selection_index(self, dof):
        """
        Get the dof indices for a given selection.

        Parameters
        ----------
        dof : dict
            dof dictionary.

        Returns
        -------
        np.ndarray
            1D array of indices for selected dofs.
        """

        return np.where(np.isin(self.dof['index'], dof['index']))[0]

    def see(self):
        """
        Creates interactive matplotlib widget to visualise mesh and 
        inspect degrees of freedom.

        Returns
        -------
        None.

        """

        # this essentially generates an interactive matplotlib of the mesh
        # used for seeing what dofs are where so you can add specific forcing
        self.forcer = Forcer(self)
        self.forcer.select_nodes()

    def save(self, folder, source='local'):
        """
        Save the model to a folder

        Parameters
        ----------
        folder : str
            folder name.
        source : str, optional
            Save to ``'local'`` or ``'database'``. The default is 'local'.

        Returns
        -------
        None.

        """

        save(folder, self, source=source)
