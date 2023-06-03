"""
model_setup
------------

This module contains the functions for setting up the WFE model.

This includes:
    - Creating the relevant `dof` dict data
    - Applying the boundary conditions
    - Sorting M, K and dofs to left and right faces
    - Creating `node` dict data
"""
import itertools
import numpy as np


def generate_dof_info(dof: dict, axis=0):
    """
    Generates the `dof` dictionary, including which face each dof is on.
    Also rolls sets the waveguide axis and created index array if none given.

    Parameters
    ----------
    dof : dict
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    dof : TYPE
        DESCRIPTION.

    """
    # make sure it's an array
    dof['coord'] = np.array(dof['coord'])
    dof['node'] = np.array(dof['node'])
    dof['fieldvar'] = np.array(dof['fieldvar'])

    # roll coord axes for the chosen waveguide axis to appear first
    dof['coord'] = np.roll(dof['coord'], axis, axis=0)

    # find dofs on left and right face
    L_inds = (dof['coord'][0] == min(dof['coord'][0]))
    R_inds = (dof['coord'][0] == max(dof['coord'][0]))

    # face array len ndof, sorting face of each dof
    dof_face = np.array(['I'] * len(dof['coord'][0]))
    dof_face[L_inds] = 'L'
    dof_face[R_inds] = 'R'

    dof['face'] = dof_face

    # set index if none
    if "index" not in dof.keys():
        dof["index"] = np.arange(len(dof['coord'][0]))
    else:
        dof['index'] = np.array(dof['index'])

    return dof


def apply_boundary_conditions(K, M, dof, null, nullf):
    """
    Applies boundary conditions according to null constraint matrices.
    Resorts and removes degrees of freedom as needed. (NOT FINISHED)

    Parameters
    ----------
    K : ndarray
        (ndof, ndof) sized array of type float or complex.
    M : ndarray
        (ndof, ndof) sized array of type float or complex.
    dof : dict
        dof dictionary.
    null : ndarray
       (ndof, ndof) sized array of type float.
    nullf : ndarray
        (ndof, ndof) sized array of type float.

    Returns
    -------
    K : ndarray
        (ndof, ndof) sized array of type float or complex.
    M : ndarray
        (ndof, ndof) sized array of type float or complex.
    dof : dict
        dof dictionary.

    """
    # dimensions of model
    ndim = dof['coord'].shape[0]

    # create empty index sorting array
    sorted_inds = np.zeros(len(null.T), dtype='int')

    # need to iterate through the rows of Null.T
    # if row i is all zero - > eliminate corresponding dof i
    # if row i has a single 1 at column j, dof j moves to index i
    # if row i has multiple non-zero columns,
    # then their elements are condensed, and moved position i
    # create empty index sorting array

    # go through the rows (i index)
    for i in range(len(null.T)):

        # go through the elements of each Null.T row
        for j in range(len(null)):

            # Once a non-zero element is found,
            # that dof index j is moved to the position i
            if null.T[i, j] != 0:

                sorted_inds[i] = j

    # go through dof info and resort
    for key in dof.keys():

        # special case since coord array is multidimensional
        if key == 'coord':

            new_coords = np.array([dof[key][i, sorted_inds]
                                  for i in range(ndim)])
            dof[key] = new_coords

        else:
            dof[key] = dof[key][sorted_inds]

    # apply the boundary conditions to M and K
    K = nullf.T @ K @ null
    M = null.T @ M @ null

    return K, M, dof


def order_system_faces(K, M, dof):
    """

    Parameters
    ----------
    K : ndarray
        (ndof, ndof) sized array of type float or complex.
    M : ndarray
        (ndof, ndof) sized array of type float or complex.
    dof : dict
        dof dictionary.

    Returns
    -------
    K : ndarray
        (ndof, ndof) sized array of type float or complex.
    M : ndarray
        (ndof, ndof) sized array of type float or complex.
    dof : dict
        dof dictionary.

    """

    # get the sorted indices
    reversed_coord_arrays = [_ for _ in dof['coord'][::-1]]
    sorted_inds = np.lexsort(reversed_coord_arrays)

    # go through keys of dof and sort,
    for key in dof.keys():

        if key != 'coord':
            dof[key] = dof[key][sorted_inds]

        # coordinate arrays are special case, sort each dimension
        else:
            for i in range(dof['coord'].shape[0]):
                dof['coord'][i] = dof['coord'][i][sorted_inds]

    # sort M and K
    K = K[:, sorted_inds][sorted_inds, :]
    M = M[:, sorted_inds][sorted_inds, :]

    return K, M, dof


def substructure_matrices(K, M, dof):
    """
    Creates dictionaries for the submatrices of K and M

    Parameters
    ----------
    K : ndarray
        (ndof, ndof) sized array of type float or complex.
    M : ndarray
        (ndof, ndof) sized array of type float or complex.
    dof : dict
        dof dictionary.

    Returns
    -------
    K_sub : dict
        dictionary of substructured stiffness matrices.
    M_sub : dict
        dictionary of substructured mass matrices.

    """

    # all possible face type
    face_types = ("L", "R", "I")

    # create empty substructure dict
    K_sub = {}
    M_sub = {}

    # go through each face combination
    for face_type in itertools.product(face_types, face_types):

        face_type = "".join(face_type)

        # create the row and column masks to select face types
        row_mask = dof['face'] == face_type[0]
        col_mask = dof['face'] == face_type[1]

        # assign substructured matrices
        K_sub[face_type] = K[row_mask][:, col_mask]
        M_sub[face_type] = M[row_mask][:, col_mask]

    # create the partitions needed for condensing
    K_sub['EE'] = np.vstack((
        np.hstack((K_sub['LL'], K_sub['LR'])),
        np.hstack((K_sub['RL'], K_sub['RR']))))

    M_sub['EE'] = np.vstack((
        np.hstack((M_sub['LL'], M_sub['LR'])),
        np.hstack((M_sub['RL'], M_sub['RR']))))

    K_sub['EI'] = np.vstack((K_sub['LI'], K_sub['RI']))
    M_sub['EI'] = np.vstack((M_sub['LI'], M_sub['RI']))
    K_sub['IE'] = np.hstack((K_sub['IL'], K_sub['IR']))
    M_sub['IE'] = np.hstack((M_sub['IL'], M_sub['IR']))

    return K_sub, M_sub


def create_node_dict(dof):
    """
    Creates node dictionary for nodes on the left face of the model

    Parameters
    ----------
    dof : dict
        dof dictionary.

    Returns
    -------
    node : dict
        node dictionary.

    """

    # CREATING NODE DICTIONARY
    node = {'number': None,
            'coord': [],
            'fieldvar': [],
            'dof': []}

    # get the nodes of the left face
    where_left = (dof['face'] == 'L')
    left_nodes = dof['node'][where_left]

    numbers = np.unique(left_nodes)
    node['number'] = numbers

    # find the coordinates and field variables for each node
    for current_node in node['number']:
        current_node_indices = dof['node'] == current_node

        node['coord'].append(
            dof['coord'][:, current_node_indices][:, 0])
        node['fieldvar'].append(
            dof['fieldvar'][current_node_indices].tolist())
        node['dof'].append(np.where(current_node_indices)[0].tolist())

    node['coord'] = np.vstack(node['coord']).T

    return node
