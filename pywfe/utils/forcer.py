"""
forcer
------

This module contains the Forcer class, which is used to add forces to a 
pywfe.Model object.

"""
import numpy as np
import matplotlib.pyplot as plt


class Forcer:

    def __init__(self, model):
        self.model = model
        self.selected_nodes = []

    def select_nodes(self):

        self.selected_nodes = []

        ndim = len(self.model.node['coord'])

        if ndim == 2:
            x_plot, y_plot = (np.zeros_like(self.model.node['coord'][0]),
                              self.model.node['coord'][-1])
        elif ndim == 3:
            x_plot, y_plot = (self.model.node['coord'][-2],
                              self.model.node['coord'][-1])

        self.fig, self.ax = plt.subplots()

        colors = []

        for i in range(len(self.model.node['number'])):

            if len(self.model.node['fieldvar'][i]) == 1:

                colors.append('C0')

            elif len(self.model.node['fieldvar'][i]) == 4:

                colors.append('black')

            else:
                colors.append('grey')

        self.ax.scatter(x_plot, y_plot, s=8, color=colors, picker=5)

        self.callbacks = self.fig.canvas.callbacks.connect(
            'pick_event', self.on_pick)

    def on_pick(self, event):

        selected_node = self.model.node['number'][event.ind[0]]
        self.selected_nodes.append(selected_node)

        print(selected_node)

    def add_nodal_force(self, node, force_dict):

        node_index = np.argmax(
            self.model.node['number'] == node)

        dofs = self.model.node['dof'][node_index]
        fieldvars = self.model.node['fieldvar'][node_index]
            

        for degree_of_freedom, field_variable in zip(dofs, fieldvars):
            
            if field_variable in force_dict:
                self.model.force[degree_of_freedom] = force_dict[field_variable]
