# pyWFE

##### Python implementation of the Wave Finite Element (WFE) method.

---

The WFE method predicts wave propagation in waveguides of any length by meshing a small segment of the waveguide in conventional FE and applying periodic boundary conditions. 



This Python package applies the WFE method to given mesh data and uses a single class which provides the functionality to:



- Sort system matrices and condense internal degrees of freedom.

- solve the WFE eigenproblem (mode shapes, propagation constants) and sort the solutions

- Apply harmonic forcing to the system

- Solve response any axial distance



![Demo](imgs/animation.gif)

**[THIS REPO IS STILL A WORK IN PROGRESS]**

[Find the documentation HERE](https://pywfe.readthedocs.io/en/latest/)
