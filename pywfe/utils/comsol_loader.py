"""
comsol_loader
-------------

This module contains the functionality needed to convert COMSOL data
extracted from MATLAB LiveLink into a pywfe.Model class.

To extract the relevant matrices from COMSOL, use the following code
after `out = model;`
    
.. code-block:: matlab

    MA = mphmatrix(model, 'sol1', 'out', {'K', 'D', 'E', 'L', 'Kc', 'Ec', 'Null', 'Nullf'});
    info = mphxmeshinfo(model)

    fid = fopen("mesh_info.json", 'w');
    encodedJSON = jsonencode(info); 
    fprintf(fid, encodedJSON); 
    fclose('all'); 

    writematrix(MA.K, 'K.txt', 'Delimiter', 'tab');
    writematrix(MA.E, 'M.txt', 'Delimiter', 'tab');

    writematrix(MA.Kc, 'Kc.txt', 'Delimiter', 'tab');
    writematrix(MA.Ec, 'Mc.txt', 'Delimiter', 'tab');

    writematrix(MA.Null, 'Null.txt', 'Delimiter', 'tab');
    writematrix(MA.Nullf, 'Nullf.txt', 'Delimiter', 'tab');
"""

import logging
import json
import numpy as np
import pywfe


def load_comsol(folder, axis=0,
                logging_level=20,
                solver='transfer_matrix'):
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
    except FileNotFoundError:
        logging.info("No boundary conditions found for COMSOL model")
        null, nullf = None, None

    with open(f"{folder}/mesh_info.json") as json_file:
        info = json.load(json_file)

    dof = {}
    dof['coord'] = np.array(info['dofs']['coords'])
    dof['coord'] = np.round(dof['coord'], decimals=8)

    dof['node'] = np.array(info['dofs']['nodes'])

    dof['index'] = info['dofs']['solvectorinds']

    dof_names = np.array(info['dofs']['dofnames'])
    dof_inds = np.array(info['dofs']['nameinds'])

    dof['fieldvar'] = dof_names[dof_inds]
    dof['fieldvar'] = np.array([_.split(".")[-1] for _ in dof['fieldvar']])

    return pywfe.Model(K, M, dof,
                       null=null,
                       nullf=nullf,
                       axis=axis,
                       logging_level=20,
                       solver=solver)


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
