"""
vtk tools
---------

This module contains functions for sorting/returning data to be visualised
in ParaView

"""
import numpy as np
from pyevtk.hl import pointsToVTK
from pywfe.core import model_setup
import copy


def generate_coordinates(dof, x):
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
    node = model_setup.create_node_dict(dof)
    node_coords = node['coord']

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


def sort_to_vtk(displacements, dof, vtk_fmt=True, fieldmap=None):
    """
    sorts a displacement array to a paraview-ready format.

    Parameters
    ----------
    displacements : np.ndarray
        field array.
    dof : TYPE
        pywfe.Model.dict.dof dict.
    vtk_fmt : TYPE, optional
        DESCRIPTION. The default is True.
    fieldmap : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    node = model_setup.create_node_dict(dof)

    # Getting all unique fieldvar values from the model's degrees of freedom
    all_fieldvars = list(set(dof['fieldvar'].tolist()))

    # Initializing a dictionary with the fieldvar values as keys
    field = dict.fromkeys(all_fieldvars)

    # If the displacements are one dimensional, reshape them into a 2D array
    if len(displacements.shape) == 1:
        displacements = np.reshape(displacements, (1, len(displacements)))

    # Get the length of the input displacements array
    x_len = displacements.shape[0]
    # Get the number of nodes in the model
    disp_len = len(node['number'])

    # For each fieldvar, initialize an array of zeros with size equal to the number of nodes times the length of displacements
    for var in all_fieldvars:
        field[var] = np.ascontiguousarray(
            np.zeros((disp_len*x_len), dtype='complex'))

    # Iterate over all nodes in the model
    for i in range(len(node['number'])):
        # For each degree of freedom in the current node
        for j in range(len(node['dof'][i])):
            # Get the current fieldvar and dof
            var = node['fieldvar'][i][j]
            dof = node['dof'][i][j]

            # For each displacement, add it to the corresponding fieldvar array in the field dictionary
            for k in range(x_len):
                field[var][i + k*disp_len] = displacements[k, dof]

    # If vtk_fmt flag is set, convert all values in the field dictionary to real values
    if vtk_fmt:
        for key in field.keys():
            field[key] = np.ascontiguousarray(field[key].real)

    # Return the field dictionary
    if fieldmap is not None:

        new_field = {}

        for oldvar, newvar in fieldmap.items():

            new_field[newvar] = field[oldvar]

        return new_field

    return field


def save_as_vtk(filename, field_array, x, dof, fieldmap=None):
    """
    Save data as a VTK file for visualization in ParaView.

    This function generates a structured point cloud from the given degree-of-freedom (DOF) data and saves 
    the provided field array in a format that can be visualized using VTK-compatible software like ParaView.

    Parameters
    ----------
    filename : str
        The name of the output VTK file (without the .vtk extension).
    field_array : np.ndarray
        The numerical field values to be stored in the VTK file.
    x : float or array-like
        The x-coordinate(s) where the field data is defined.
    dof : dict
        Dictionary containing the model’s degrees of freedom (DOF) information.
    fieldmap : dict, optional
        A dictionary mapping original field variable names to new names for improved readability in visualization.
        If None, the original field variable names are used. Default is None.

    Returns
    -------
    None
        The function writes a VTK file to disk but does not return a value.

    Notes
    -----
    - This function internally calls `generate_coordinates()` to determine the spatial positions of the DOFs.
    - It uses `sort_to_vtk()` to transform the field data into a VTK-compatible format.
    - The output can be loaded directly into ParaView for further analysis.
    - If the model is 2D, the function adds a zero-valued third coordinate (`zz = 0`) to ensure compatibility with VTK.

    Examples
    --------
    Save a model's displacement field as a VTK file:

    ```python
    save_as_vtk("output/displacement_field", displacements, x, model.dof)
    ```

    With a field variable mapping:

    ```python
    fieldmap = {"displacement_x": "Ux", "displacement_y": "Uy"}
    save_as_vtk("output/displacement_field", displacements, x, model.dof, fieldmap=fieldmap)
    ```

    """

    coords = generate_coordinates(dof, x)
    field = sort_to_vtk(field_array, dof, fieldmap=fieldmap)

    if len(coords) == 2:
        xx, yy = coords
        zz = np.zeros_like(xx)
    else:
        xx, yy, zz = coords

    for key in field.keys():
        field[key] = np.ascontiguousarray(field[key].real)

    pointsToVTK(filename, xx, yy, zz, field)


def rotate_dofs(dofs, theta):
    """
    Rotates the 2D coordinates of the DOFs by angle theta.

    Parameters:
    dofs (dict): Original DOFs dictionary with 'coord' key.
    theta (float): Angle in radians to rotate.

    Returns:
    dict: New DOFs dictionary with rotated coordinates.
    """
    # Create a deep copy of the DOFs dictionary
    rotated_dofs = copy.deepcopy(dofs)

    # Define the 2D rotation matrix components
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Apply rotation to each coordinate pair
    for i in range(rotated_dofs['coord'].shape[1]):  # Iterate over columns
        x, y = rotated_dofs['coord'][:, i]
        rotated_dofs['coord'][0, i] = x * cos_theta - y * sin_theta
        rotated_dofs['coord'][1, i] = x * sin_theta + y * cos_theta

    return rotated_dofs
