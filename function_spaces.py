#!/usr/bin/env python3

"""
Implementation of common elements.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.tri as tri
import femtolib as femto


class Lagrange(femto.FunctionSpace):
    def __init__(self, mesh, reference, n_dof, idx=0):
        super().__init__(mesh, reference, n_dof, idx)
        
    def eval(self, mesh, x, D=None):
        elt_id = mesh.find_element(x)
        xi = mesh.get_ref_coords(elt_id, x)
        uh = 0.0
        if D is None:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.phi(i, *xi)
                       *self.dof[self.mesh.elements[elt_id][i]] )
        else:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.d_phi(i, D, *xi)
                       *self.dof[self.mesh.elements[elt_id][i]] )
        return uh
        
    def plot(self, mesh, plot_type='contour'):
        if plot_type == 'surface':
            plt.figure(figsize=(6, 6))
            ax = plt.axes(projection='3d')
            triangulation = tri.Triangulation(self.mesh.nodes[:, 0],
                                              self.mesh.nodes[:, 1])
            surf = ax.plot_trisurf(triangulation, self.dof, color='red')
            # The next two lines may not be necessary
            surf._facecolors2d = surf._facecolors
            surf._edgecolors2d = surf._edgecolors

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'Field {self.idx}')
            ax.legend()
            
        elif plot_type == 'contour':
            fig, ax = plt.subplots(1, 1)
            triangulation = tri.Triangulation(self.mesh.nodes[:, 0],
                                              self.mesh.nodes[:, 1])
            ax.tricontour(triangulation, self.dof,
                          levels=40, linewidths=0.5, colors='k')
            surf = ax.tricontourf(triangulation, self.dof,
                                  levels=40, cmap="RdBu_r")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(surf, ax=ax)
            plt.subplots_adjust(hspace=0.5)
            
        plt.show()
        
        
class P1(Lagrange):
    def __init__(self, mesh, reference, idx=0):
        n_dof = mesh.n_nodes
        super().__init__(mesh, reference, n_dof, idx)
        
        
def _find_edge(edges, n1, n2):
    i_edge = -1
    for i, edge in enumerate(edges):
        if ( (n1 == edge[0] and n2 == edge[1]) or
             (n2 == edge[0] and n1 == edge[1]) ):
             i_edge = i
             break
    if i_edge == -1:
        raise Exception('Invalid edge!')
    else:
        return i_edge
        
def _get_P2_mesh(mesh):
    boundary = []
   
    nodes = mesh.nodes.tolist()
    bdy_nodes = mesh.boundary[0].tolist()
    node_count = 0
    
    if mesh.dim == 2:
        for i in range(mesh.n_edges):
            x1, y1 = mesh.nodes[mesh.edges[i, 0]]
            x2, y2 = mesh.nodes[mesh.edges[i, 1]]
            nodes.append([0.5*(x1 + x2), 0.5*(y1 + y2)])
            if i in mesh.boundary[1]:
                bdy_nodes.append(mesh.n_nodes + node_count)
            node_count += 1
        
    nodes = np.array(nodes)
    bdy_nodes = np.array(bdy_nodes, dtype=int)
    boundary.append(bdy_nodes)
    boundary.append(mesh.boundary[1])

    elements = mesh.elements.tolist()
    for i in range(mesh.n_elements):
        n_vertices = len(mesh.elements[i])
        for j in range(n_vertices - 1):
            i_edge = _find_edge(mesh.edges,
                                mesh.elements[i, j], mesh.elements[i, (j + 1)])
            elements[i].append(mesh.n_nodes + i_edge)
        i_edge = _find_edge(mesh.edges, 
                            mesh.elements[i, (n_vertices - 1)], mesh.elements[i, 0])
        elements[i].append(mesh.n_nodes + i_edge)
        
    mesh_P2 = femto.Mesh(mesh.dim, nodes, mesh.edges, mesh.facets,
                         elements, boundary, mesh.reference)
    return mesh_P2
        
        
class P2(Lagrange):
    def __init__(self, mesh, reference, idx=0):
        n_dof = mesh.n_nodes + mesh.n_edges
        mesh_P2 = _get_P2_mesh(mesh)
        super().__init__(mesh_P2, reference, n_dof, idx)

        
# class Q1(femto.FunctionSpace):
#     def __init__(self, mesh, reference, idx=0):
#         n_dof = len(mesh.nodes)
#         super().__init__(reference, n_dof, idx)
#
#     def eval(self, mesh, x, D=None):
#         raise NotImplementedError()
#
#     def plot(self, mesh):
#         triangulation = tri.Triangulation(self.mesh.nodes[:, 0],
#                                           self.mesh.nodes[:, 1])
#
#         plt.figure(figsize=(6, 6))
#         ax = plt.axes(projection='3d')
#
#         surf = ax.plot_trisurf(triangulation, self.dof, color='red')
#         # The next two lines may not be necessary
#         surf._facecolors2d = surf._facecolors
#         surf._edgecolors2d = surf._edgecolors
#
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel(f'Field {self.idx}')
#         ax.legend()
#         plt.show()
