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

    database_path = pywfe.DATABASE_PATH

    if source == 'database':
        folder = os.path.join(database_path, folder)

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

    database_path = pywfe.DATABASE_PATH
    
    print(os.listdir(database_path))
