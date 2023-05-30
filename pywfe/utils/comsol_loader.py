"""
COMSOL loader
-------------

This module contains the functionality needed to convert COMSOL data
extracted from MATLAB LiveLink into a pywfe.Model class. 
"""

import logging
import json
import numpy as np
import pywfe


def load_comsol(folder, axis = 0, logging_level = 20, solver = 'transfer_matrix'):
    """

    Parameters
    ----------
    folder : string
        path to the folder containing the COMSOL LiveLink data.
    axis : int, optional
        Waveguide axis. The default is 0.
    logging_level : int, optional
        Logging level. The default is 20 (info).

    Returns
    -------
    model : pywfe.model class
        a pywfe model.

    """

    comsol_i2j(f"{folder}/K.txt", skiprows=0)
    comsol_i2j(f"{folder}/M.txt", skiprows=0)
    
    K = np.loadtxt(f"{folder}/K.txt", dtype='complex')
    M = np.loadtxt(f"{folder}/M.txt", dtype='complex')
        
    try:
        null = np.loadtxt(f"{folder}/Null.txt")
        nullf = np.loadtxt(f"{folder}/Nullf.txt")
    except:
        logging.info("No boundary conditions found for COMSOL model")
        null, nullf = None, None
        
    with open(f"{folder}/mesh_info.json") as json_file:
        info = json.load(json_file)
        
    dof = {}
    dof['coord'] = np.array(info['dofs']['coords'])
    dof['coord'] = np.round(dof['coord'], decimals = 8)
    
    dof['node'] = np.array(info['dofs']['nodes'])
    
    dof['index'] = info['dofs']['solvectorinds']
    
    
    dof_names = np.array(info['dofs']['dofnames'])
    dof_inds = np.array(info['dofs']['nameinds'])

    dof['fieldvar'] = dof_names[dof_inds]
    
    return pywfe.Model(K, M, dof,
                        null = null,
                        nullf = nullf,
                        axis=axis,
                        logging_level = 20,
                        solver = solver)

    
def comsol_i2j(filename, skiprows=0):
    """
    Converts complex 'j' imaginary unit from COMSOL to python 'j'

    Parameters
    ----------
    filename : string
        filename to convert.
    skiprows : int, optional
        see numpy loadtxt. The default is 1.

    Returns
    -------
    None.
    """

    with open(f"{filename}", "rt") as fin:
        lines = fin.readlines()

    lines = lines[:skiprows] + \
        [line.replace("i", "j") for line in lines[skiprows:]]

    with open(f"{filename}", "wt") as fout:
        fout.writelines(lines)

