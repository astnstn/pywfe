.. pywfe documentation master file, created by
   sphinx-quickstart on Sat May 27 10:51:44 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================
Python Wave Finite Element
==========================

pywfe - a python package for the wave finite element method
===========================================================
.. image:: logo.png
   :alt: Alternative text for the image
   :width: 1000px
   :align: center

Purpose
+++++++
This package implements the Wave Finite Element Method (WFEM) in Python to analyse guided waves in 1 dimension. Initially Written to analyse mechanical waves in fluid filled pipes meshed in COMSOL.

**Currently only works for infinite waveguides.**

The `pywfe.Model` class provides a high level api to calculate the free and forced response in the waveguide. It is initialised with the stiffness and mass matrix
:math:`K` and :math:`M`. It is assumed that there is a shared ordering between rows and columns of these matrices. The :math:`N` degrees of freedom are described by arrays with shapes

- coordinate :math:`(n_D, N)`
- node number :math:`(N)`
- field variable :math:`(N)`
- index :math:`(N)`

Where :math:`n_D` is the number of spatial dimensions and ``coordinate[0]`` ideally gives the axial coordinate array.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   model
   core
   utils
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
