"""
io_utils
--------

This module contains the functionality needed to save and load pywfe.Model objects
"""

import logging
import json
import numpy as np
import pywfe
import os
import shutil


def load(folder, source='local'):
    '''
    Load a model from the path to its folder.

    Parameters
    ----------
    folder : Path or str
        The model to load.
    source : str, optional
        Where to load the model from. The default is 'local'.
        If 'database' is chosen, searches for model in the database folder
        located in the package directory. If 'local' is chosen but not found,
        searches database.

    Raises
    ------
    FileNotFoundError
        Could not find the model folder.

    Returns
    -------
    model : pywfe.Model
        The pywfe Model object.

    '''

    database_path = pywfe.DATABASE_PATH
    local_folder = folder

    # If source is 'database', look only in the database
    if source == 'database':
        folder = os.path.join(database_path, folder)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Model not found in database: {folder}")
    else:
        # If source is 'local', first look in the local directory
        if not os.path.exists(local_folder):
            # If not found locally, look in the database
            folder = os.path.join(database_path, folder)
            if not os.path.exists(folder):
                raise FileNotFoundError(
                    f"Model not found in local directory or database: {folder}")
        else:
            folder = local_folder

    # Proceed with loading the model
    K = np.load(f'{folder}/K.npy')
    M = np.load(f'{folder}/M.npy')

    with np.load(f'{folder}/dof.npz') as data:
        dof = {key: data[key] for key in data.keys()}

    model = pywfe.Model(K, M, dof,
                        axis=0,
                        logging_level=20,
                        solver='transfer_matrix')

    try:
        with open(f"{folder}/description.txt", 'r') as file:
            description = file.read()
        model.description = description
    except FileNotFoundError:
        pass

    return model


def save(folder, model, source='local'):
    '''
    Saves the model. Only saves mesh information and description, not results.

    Parameters
    ----------
    folder : path or str
        Folder to save the model to. Creates a folder or overwrites if exists.
    model : pywfe.Model
        The model to save.
    source : str, optional
        Where to save the model to.
        If 'local' uses current working directory.
        If 'database' saves in the database folder in the package directory.
        The default is 'local'.

    Returns
    -------
    None.

    '''

    database_path = pywfe.DATABASE_PATH

    if source == 'database':
        folder = os.path.join(database_path, folder)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)
    print(f"saving to {folder}")

    np.save(f'{folder}/K.npy', model.K)
    np.save(f'{folder}/M.npy', model.M)
    np.savez(f'{folder}/dof.npz', **model.dof)

    if model.description is not None:
        with open(f"{folder}/description.txt", 'w') as file:
            file.write(model.description)

    print("Model Saved")


def database():
    '''
    Lists all models in the database

    Returns
    -------
    None.

    '''

    database_path = pywfe.DATABASE_PATH

    print(os.listdir(database_path))
