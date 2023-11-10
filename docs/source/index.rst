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

The `pywfe.Model` class provides a high level api to calculate the free and forced response in the waveguide. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   pywfe
   model
   core
   utils
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
