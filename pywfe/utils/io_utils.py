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
        Where to load the model from.
        - If 'database', searches **user database** first, then falls back to **package database**.
        - If 'local', searches **local directory** first, then user database, then package database.

    Raises
    ------
    FileNotFoundError
        Could not find the model folder.

    Returns
    -------
    model : pywfe.Model
        The pywfe Model object.
    '''

    user_database_path = pywfe.USER_DATABASE_PATH
    package_database_path = pywfe.PACKAGE_DATABASE_PATH

    # Ensure user database exists
    os.makedirs(user_database_path, exist_ok=True)

    # Construct potential paths
    user_model_path = os.path.join(user_database_path, folder)
    package_model_path = os.path.join(package_database_path, folder)
    # Convert relative path to absolute
    local_model_path = os.path.abspath(folder)

    # ---- CASE 1: If 'database' is chosen ----
    if source == 'database':
        if os.path.exists(user_model_path):
            folder = user_model_path
        elif os.path.exists(package_model_path):
            folder = package_model_path
        else:
            raise FileNotFoundError(
                f"Model '{folder}' not found in user database or package database."
            )

    # ---- CASE 2: If 'local' is chosen ----
    else:  # Default to 'local'
        if os.path.exists(local_model_path):
            folder = local_model_path
        elif os.path.exists(user_model_path):
            folder = user_model_path
        elif os.path.exists(package_model_path):
            folder = package_model_path
        else:
            raise FileNotFoundError(
                f"Model '{folder}' not found in local directory, user database, or package database."
            )

    # ---- Proceed with loading the model ----
    K = np.load(f'{folder}/K.npy')
    M = np.load(f'{folder}/M.npy')

    with np.load(f'{folder}/dof.npz') as data:
        dof = {key: data[key] for key in data.keys()}

    model = pywfe.Model(K, M, dof,
                        axis=0,
                        logging_level=20,
                        solver='transfer_matrix')

    # Load description if it exists
    try:
        with open(f"{folder}/description.txt", 'r') as file:
            description = file.read()
        model.description = description
    except FileNotFoundError:
        pass

    print(f'model found and loaded from: {folder}')
    return model


def save(folder, model, source='local'):
    '''
    Saves the model. Only saves mesh information and description, not results.

    Parameters
    ----------
    folder : path or str
        Folder to save the model to. Creates a folder or overwrites if it exists.
    model : pywfe.Model
        The model to save.
    source : str, optional
        Where to save the model.
        - If 'local', saves in the current working directory (`./`).
        - If 'database', saves in the **user database** (`~/.pywfe/database/`).
        - The default is 'local'.

    Returns
    -------
    None.
    '''

    user_database_path = pywfe.USER_DATABASE_PATH

    # ---- CASE 1: Save to user database ----
    if source == 'database':
        folder = os.path.join(user_database_path, folder)

        # Ensure user database exists before saving
        os.makedirs(user_database_path, exist_ok=True)

    # ---- CASE 2: Save locally ----
    else:  # Default: 'local'
        folder = os.path.abspath(folder)  # Convert to absolute path

    # ---- Overwrite existing folder if needed ----
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)
    print(f"Saving to {folder}")

    # ---- Save model data ----
    np.save(f'{folder}/K.npy', model.K)
    np.save(f'{folder}/M.npy', model.M)
    np.savez(f'{folder}/dof.npz', **model.dof)

    if model.description is not None:
        with open(f"{folder}/description.txt", 'w') as file:
            file.write(model.description)

    print("Model Saved")


def database():
    '''
    Lists all models in the database, printing user-saved models first,
    followed by the default package models. Each model is printed on a new line.

    Returns
    -------
    None.
    '''

    package_database_path = pywfe.PACKAGE_DATABASE_PATH
    user_database_path = pywfe.USER_DATABASE_PATH

    print('USER SAVED MODELS:')
    user_models = [path for path in os.listdir(user_database_path)
                   if os.path.isdir(os.path.join(user_database_path, path))]

    if user_models:
        for model in sorted(user_models):
            print(f" - {model}")
    else:
        print(" (No user models found.)")

    print('\n---- DEFAULT MODELS IN PACKAGE ----')
    package_models = [path for path in os.listdir(package_database_path)
                      if os.path.isdir(os.path.join(package_database_path, path))]

    if package_models:
        for model in sorted(package_models):
            print(f" - {model}")
    else:
        print(" (No default models found.)")
