pywfe.Model Class
=================

.. currentmodule:: pywfe.model

Introduction
------------

The pywfe.model class is the main object in this package. It brings together most WFE functionality into a single API. The class represents a waveguide and is initialised with the mesh information for a single segment.


Initialisation
--------------

.. automethod:: pywfe.model.Model.__init__

Attributes
----------

.. autoattribute:: pywfe.model.Model.K
.. autoattribute:: pywfe.model.Model.M
.. autoattribute:: pywfe.model.Model.dof
.. autoattribute:: pywfe.model.Model.node  

.. autoattribute:: pywfe.model.Model.K_sub
.. autoattribute:: pywfe.model.Model.M_sub

.. autoattribute:: pywfe.model.Model.eigensolution
.. autoattribute:: pywfe.model.Model.force

Methods
-------

.. autoclass:: pywfe.Model
   :members:
   :exclude-members: __init__, is_same_frequency, reflection_matrices, set_boundary, 
