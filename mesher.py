#!/usr/bin/env python3

"""
Functions to create meshes.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import femtolib as femto


def triangle_to_femto(t_dict):
    nodes = t_dict['vertices']
    elements = t_dict['triangles']
    edges = t_dict['edges']
    facets = edges

    boundary = []
    
    boundary_nodes = []
    markers = t_dict['vertex_markers']
    for i in range(len(nodes)):
        if markers[i, 0] == 1:
            boundary_nodes.append(i)
    boundary_nodes = np.array(boundary_nodes, dtype=int)
    boundary.append(boundary_nodes)

    boundary_edges = []
    markers = t_dict['edge_markers']
    for i in range(len(edges)):
        if markers[i, 0] == 1:
            boundary_edges.append(i)
    boundary_edges = np.array(boundary_edges, dtype=int)
    boundary.append(boundary_edges)

    return nodes, edges, facets, elements, boundary


def _in_triangle(x, y, x1, x2, x3, y1, y2, y3):
    J = np.zeros((2, 2))
    J[0, 0] = x2 - x1
    J[0, 1] = x3 - x1
    J[1, 0] = y2 - y1
    J[1, 1] = y3 - y1
    l1, l2 = np.linalg.inv(J) @ np.array([x - x1, x - y1])
    if l1 >= 0 and l2 >= 0 and (l1 + l2) <= 1:
        return True
    else:
        return False


def _find_element_triangle(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        coords = nodes[elt]
        x1, x2, x3 = coords[:, 0]
        y1, y2, y3 = coords[:, 1]
        if _in_triangle(x[0], x[1], 
                        x1, x2, x3, 
                        y1, y2, y3):
            elt_id = iE
            break
    return elt_id
    

class TriMesh(femto.Mesh):
    def __init__(self, nodes, edges, facets, elements, boundary, reference):
        dim = 2
        super().__init__(dim, nodes, edges, facets, elements, boundary, reference)
        
    def find_element(self, x):
        if self.reference.affine:
            return _find_element_triangle(x, self.nodes, self.elements)
        
    def Jacobian(self, elt_id, xi):
        nodes = self.nodes[self.elements[elt_id]] 
        J = np.zeros((self.dim, self.dim))
        if self.reference.affine:
            x1, x2, x3 = nodes[:, 0]
            y1, y2, y3 = nodes[:, 1]
            J[0, 0] = x2 - x1
            J[0, 1] = x3 - x1
            J[1, 0] = y2 - y1
            J[1, 1] = y3 - y1
        else:
            for i in range(self.dim):
                for j in range(self.dim):
                    J[i, j] = 0.0
                    for k in range(self.reference.n_dof):
                        J[i, j] += nodes[k, i]*self.reference.d_phi(k, j, *xi)
        return J

    def get_ref_coords(self, elt_id, x):
        if self.reference.affine:
            nodes = self.nodes[self.elements[elt_id]]
            J = self.Jacobian(elt_id, None)
            xi = np.linalg.inv(J) @ np.array([(x[0] - nodes[0, 0]),
                                              (x[1] - nodes[0, 1])])
            return xi


class UnitSquareTri(TriMesh):
    def __init__(self, nx=4, ny=4, reference=None):
        self.nx = nx
        self.ny = ny
        hx = 1.0/nx
        hy = 1.0/ny
        boundary = []
        
        nodes = []
        node_count = 0
        bdy_nodes = []
        tol = 1e-6

        for j in range(ny + 1):
            y = j*hy
            for i in range(nx + 1):
                x = i*hx
                nodes.append([x, y])

                if (   abs(x) <= tol
                    or abs(1 - x) <= tol
                    or abs(y) <= tol
                    or abs(1 - y) <= tol ):
                    bdy_nodes.append(node_count)

                node_count += 1

        nodes = np.array(nodes)
        bdy_nodes = np.array(bdy_nodes, dtype=int)
        boundary.append(bdy_nodes)
        
        edges = []
        bdy_edges = []
        edge_count = 0
        
        for j in range(ny):
            for i in range(nx):
                edges.append([j*(nx + 1) + i, j*(nx + 1) + i + 1])
                if j == 0:
                    bdy_edges.append(edge_count)
                edge_count += 1
                
                edges.append([j*(nx + 1) + i + 1, (j + 1)*(nx + 1) + i])
                edge_count += 1
                
                edges.append([(j + 1)*(nx + 1) + i, j*(nx + 1) + i])
                if i == 0:
                    bdy_edges.append(edge_count)
                edge_count += 1
                
            edges.append([j*(nx + 1) + nx, (j + 1)*(nx + 1) + nx])
            bdy_edges.append(edge_count)
            edge_count += 1
        
        for i in range(nx):
            edges.append([ny*(nx + 1) + i, ny*(nx + 1) + i + 1])
            bdy_edges.append(edge_count)
            edge_count += 1
        
        edges = np.array(edges, dtype=int)
        bdy_edges = np.array(bdy_edges, dtype=int)
        boundary.append(bdy_edges)
        
        facets = edges
        elements = []

        for j in range(ny):
            for i in range(nx):
                elements.append([
                    j*(nx + 1) + i,
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

                elements.append([
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

        elements = np.array(elements, dtype=int)

        super().__init__(nodes, edges, facets, elements, boundary, reference)
        

def _find_element_quad(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        coords = nodes[elt]
        x1, x2, x3, x4 = coords[:, 0]
        y1, y2, y3, y4 = coords[:, 1]
        t1 = _in_triangle(x[0], x[1],
                          x1, x2, x4,
                          y1, y2, y4)
        t2 = _in_triangle(x[0], x[1],
                          x3, x4, x2,
                          y3, y4, y2)
        if t1 or t2:
            elt_id = iE
            break
    return elt_id
    
    
class QuadMesh(femto.Mesh):
    def __init__(self, nodes, edges, facets, elements, boundary, reference):
        dim = 2
        super().__init__(dim, nodes, edges, facets, elements, boundary, reference)

    def find_element(self, x):
        if self.reference.affine:
            return _find_element_quad(x, self.nodes, self.elements)

    def get_ref_coords(self, x, nodes):
        raise NotImplementedError()
        

class UnitSquareQuad(QuadMesh):
    def __init__(self, nx=4, ny=4, reference=None):
        self.nx = nx
        self.ny = ny
        hx = 1.0/nx
        hy = 1.0/ny
        
        boundary = []
        
        nodes = []
        node_count = 0
        bdy_nodes = []
        tol = 1e-6

        for j in range(ny + 1):
            y = j*hy
            for i in range(nx + 1):
                x = i*hx
                nodes.append([x, y])

                if (   abs(x) <= tol
                    or abs(1 - x) <= tol
                    or abs(y) <= tol
                    or abs(1 - y) <= tol ):
                    bdy_nodes.append(node_count)

                node_count += 1

        nodes = np.array(nodes)
        bdy_nodes = np.array(bdy_nodes, dtype=int)
        boundary.append(bdy_nodes)
        
        edges = []
        bdy_edges = []
        edge_count = 0
        
        for j in range(ny):
            for i in range(nx):
                edges.append([j*(nx + 1) + i, j*(nx + 1) + i + 1])
                if j == 0:
                    bdy_edges.append(edge_count)
                edge_count += 1
                
                edges.append([j*(nx + 1) + i + 1, (j + 1)*(nx + 1) + i + 1])
                if i == nx - 1:
                    bdy_edges.append(edge_count)
                edge_count += 1
                
                edges.append([(j + 1)*(nx + 1) + i + 1, (j + 1)*(nx + 1) + i])
                if j == ny - 1:
                    bdy_edges.append(edge_count)
                edge_count += 1
                
            edges.append([(j + 1)*(nx + 1), j*(nx + 1)])
            bdy_edges.append(edge_count)
            edge_count += 1
        
        edges = np.array(edges, dtype=int)
        bdy_edges = np.array(bdy_edges, dtype=int)
        boundary.append(bdy_edges)
        
        facets = edges
        elements = []

        for j in range(ny):
            for i in range(nx):
                elements.append([
                    j*(nx + 1) + i,
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

        elements = np.array(elements, dtype=int)

        super().__init__(nodes, edges, facets, elements, boundary, reference)
