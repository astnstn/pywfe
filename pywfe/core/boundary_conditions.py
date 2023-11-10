"""
Boundary Conditions
-------------------
***UNFINISHED - NOT IN USE***

This module contains functions for generating the boundary conidition matrices
"""


import numpy as np


def zero_matrix(n):
    return np.zeros((n, n))


def identity_matrix(n):
    return np.eye(n)


conditions = {"fixed": lambda n: (zero_matrix(n), identity_matrix(n)),
                       "free": lambda n: (identity_matrix(n), zero_matrix(n))}
