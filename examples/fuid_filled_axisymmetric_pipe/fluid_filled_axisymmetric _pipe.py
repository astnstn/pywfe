import numpy as np
import matplotlib.pyplot as plt
import pywfe
pywfe.log(level=10)


model = pywfe.load(
    "20-21cm axisymmetric water filled steel shell", source='database')


# %%

# model.see()

# %%

forcing_index = 82
model.force[forcing_index] = 1

# %%

f0 = 4000
x_range = np.linspace(0, 5, 300)
Q = model.displacements(x_range, f=4000)

structure_dofs = model.select_dofs(fieldvar=["u", "w"])
pressure_dofs = model.select_dofs(fieldvar="p")

Q_s = Q[:, model.selection_index(structure_dofs)]
P = Q[:, model.selection_index(pressure_dofs)]

# %%

V_s = 1j*2*np.pi*f0*Q_s

# %%

structure_field = pywfe.vtk_sort(structure_dofs, V_s, fieldmap={
                                 'u': 'u_dot', 'w': 'w_dot'}) | pywfe.vtk_sort(structure_dofs, Q_s)


pywfe.vtk_save("structure_field", structure_dofs, x_range, structure_field)
