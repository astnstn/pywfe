"""
vtk tools
------

This module contains functions for sorting/returning data to be visualised
in ParaView

"""
import numpy as np
from pyevtk.hl import pointsToVTK


def generate_coordinates(model, x):
    """Generate multi-dimensional coordinates for nodes in a model based on an input `x`.

    If `x` is iterable, each value in `x` is added to the first coordinate 
    of the nodes, and other coordinates are repeated for each value in `x`.

    If `x` is not iterable, it is simply added to the first coordinate of the nodes.

    Parameters
    ----------
    model : ModelClass
        The model object.

    x : int, float, or Iterable
        The value(s) to be added to the first coordinate of the nodes.

    Returns
    -------
    coords : list of numpy.ndarray
        List of arrays of coordinates.
    """
    node_coords = model.node['coord']

    if hasattr(x, '__iter__'):
        # `x` is iterable
        x_coords = np.array([])
        for x_val in x:
            x_coords = np.append(x_coords, node_coords[0] + x_val)

        other_coords = [np.tile(coord, len(x)) for coord in node_coords[1:]]
        coords = [x_coords] + other_coords

        # Convert coordinates to contiguous arrays
        coords = [np.ascontiguousarray(coord) for coord in coords]

    else:
        # `x` is not iterable, copy the node coordinates and add `x` to the first coordinate
        coords = [np.ascontiguousarray(coord) for coord in node_coords]
        coords[0] += x

    return coords


def sort_field(model, displacements, vtk_fmt=True):
    """
    This function sorts the displacements based on the field variables and prepares the displacement data
    to be written to a VTK file.

    Args:
        model (Model object): A model object that contains the nodes and degrees of freedom data.
        displacements (ndarray): An array of displacements, it can be 1D or 2D.
        vtk_fmt (boolean, optional): If True, all values in the field dictionary are converted to real values. 
                                      Defaults to False.

    Returns:
        field (dict): A dictionary where each key corresponds to a unique field variable ('fieldvar')
                      and each value is a contiguous array of displacements corresponding to that fieldvar.
    """
    # Getting all unique fieldvar values from the model's degrees of freedom
    all_fieldvars = list(set(model.dof['fieldvar'].tolist()))

    # Initializing a dictionary with the fieldvar values as keys
    field = dict.fromkeys(all_fieldvars)

    # If the displacements are one dimensional, reshape them into a 2D array
    if len(displacements.shape) == 1:
        displacements = np.reshape(displacements, (1, len(displacements)))

    # Get the length of the input displacements array
    x_len = displacements.shape[0]
    # Get the number of nodes in the model
    disp_len = len(model.node['number'])

    # For each fieldvar, initialize an array of zeros with size equal to the number of nodes times the length of displacements
    for var in all_fieldvars:
        field[var] = np.ascontiguousarray(
            np.zeros((disp_len*x_len), dtype='complex'))

    # Iterate over all nodes in the model
    for i in range(len(model.node['number'])):
        # For each degree of freedom in the current node
        for j in range(len(model.node['dof'][i])):
            # Get the current fieldvar and dof
            var = model.node['fieldvar'][i][j]
            dof = model.node['dof'][i][j]

            # For each displacement, add it to the corresponding fieldvar array in the field dictionary
            for k in range(x_len):
                field[var][i + k*disp_len] = displacements[k, dof]

    # If vtk_fmt flag is set, convert all values in the field dictionary to real values
    if vtk_fmt:
        for key in field.keys():
            field[key] = np.ascontiguousarray(field[key].real)

    # Return the field dictionary
    return field


def save_vtk(model, filename, x_arr, displacements):
    """
    This function saves the model displacements to a VTK file.

    Args:
        model (Model object): A model object that contains the nodes and degrees of freedom data.
        filename (str): The name of the VTK file to be saved.
        x_arr (ndarray): An array representing the x-axis coordinates.
        displacements (ndarray): An array of displacements.

    Returns:
        None. Writes data to a VTK file.
    """
    # First, sort the displacements and convert to real values
    displacements = sort_field(model, displacements, vtk_fmt=True)

    # Generate the coordinates for the displacements
    coords = generate_coordinates(model, x_arr)
    if len(coords) == 2:
        coords.insert(-1, np.zeros_like(coords[0]))
        
    x, y, z = coords

    # Finally, write the displacements to the VTK file
    pointsToVTK(filename, x, y, z, displacements)