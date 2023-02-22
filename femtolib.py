#!/usr/bin/env python3

"""
Femto: Object Oriented Finite Element Library

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.special import comb
from copy import deepcopy


class FiniteElement:
    def __init__(self, dim=1, n_dof=2, 
                 quad_order=1, quad_type=None, affine=True):
        self.dim = dim
        self.n_dof = n_dof
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.affine = affine
        
        # These are set by self.init_quadrature()
        self.n_quad = 0
        self.qpts = None
        self.qwts = None
        
        # These are set by self.init_Gauss()
        self.phi_q = None
        self.d_phi_q = None
        
        self.init_quadrature()
        self.init_Gauss()

    def init_quadrature(self):
        raise NotImplementedError()

    def integrate(self, g):
        intgl = 0.0
        for i in range(self.n_quad):
            intgl += self.qwts[i] * g(self.qpts[i])
        return intgl
        
    def integrate_gauss(self, gs):
        return np.sum(self.qwts * gs)

    def phi(self, idx_phi, *xi):
        raise NotImplementedError()

    def d_phi(self, idx_phi, idx_x, *xi):
        raise NotImplementedError()
        
    def init_Gauss(self):
        self.phi_q = np.zeros((self.n_dof, self.n_quad))
        for i in range(self.n_dof):
            for j in range(self.n_quad):
                self.phi_q[i, j] = self.phi(i, *self.qpts[j])
        
        self.d_phi_q = np.zeros((self.n_dof, self.dim, self.n_quad))
        for i in range(self.n_dof):
            for j in range(self.dim):
                for k in range(self.n_quad):
                    self.d_phi_q[i, j, k] = self.d_phi(i, j, *self.qpts[k])

    ### Beware: This code written by a novice programmer using GitHub Copilot OpenAI Codex ###
    # get shape functions at quadrature points
    def get_phi_gauss_pts(self):
        # return phi_gauss_pts if it exists to avoid recomputing the shape functions
        # at the quadrature points
        if hasattr(self, 'phi_gauss_pts'):
            return self.phi_gauss_pts
    
        self.phi_gauss_pts = np.zeros((self.n_quad, self.n_dof))
        for i in range(self.n_quad):
            for j in range(self.n_dof):
                self.phi_gauss_pts[i, j] = self.phi(j, self.qpts[i, 0], self.qpts[i, 1])
        
        return self.phi_gauss_pts # n_quad x n_dof

    # get derivatives of shape functions at quadrature points
    def get_d_phi_gauss_pts(self):
        # return d_phi_gauss_pts if it exists to avoid recomputing the shape function
        # derivatives at the quadrature points
        if hasattr(self, 'd_phi_gauss_pts'):
            return self.d_phi_gauss_pts
    
        self.d_phi_gauss_pts = np.zeros((self.n_quad, self.dim, self.n_dof))
        for i in range(self.n_quad):
            for j in range(self.dim):
                for k in range(self.n_dof):
                    self.d_phi_gauss_pts[i, j, k] = self.d_phi(k, j, self.qpts[i, 0], self.qpts[i, 1])
        
        return self.d_phi_gauss_pts # n_quad x dim x n_dof

    # AI written code ends here


class Mesh:
    def __init__(self, dim=1, 
                 nodes=None, edges=None, facets=None, elements=None, 
                 boundary=None, reference=None):
        self.dim = dim
        self.nodes = nodes
        self.edges = edges
        self.facets = facets
        self.elements = elements
        self.boundary = boundary
        self.reference = reference
        
        self.n_nodes = len(nodes)
        self.n_edges = len(edges)
        self.n_facets = len(facets)
        self.n_elements = len(elements)

    def find_element(self, x):
        raise NotImplementedError()
        
    def get_coords(self, elt_id, xi):
        nodes = self.nodes[self.elements[elt_id]]
        x = np.zeros(self.dim)
        for i in range(self.dim):
            for j in range(self.reference.n_dof):
                x[i] += self.reference.phi(j, *xi)*nodes[j, i]
        return x

    def Jacobian(self, elt_id, xi):
        nodes = self.nodes[self.elements[elt_id]] 
        J = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                J[i, j] = 0.0
                for k in range(self.reference.n_dof):
                    J[i, j] += nodes[k, i]*self.reference.d_phi(k, j, *xi)
        return J

    # # compute Jacobian at quadrature points
    # def get_Jacobian_gauss_pts(self, elt_id):
    #     nodes = self.nodes[self.elements[elt_id]] # n_dof x dim
    #     J = self.reference.get_d_phi_gauss_pts() @ nodes # n_quad x dim x dim
    #     return J

    def get_ref_coords(self, elt_id, x):
        raise NotImplementedError()

    # def integrate_over_element(self, elt_id, func):
    #     nodes = self.nodes[self.elements[elt_id]] # n_dof x dim
    #     x = self.reference.get_phi_gauss_pts() @ nodes # n_quad x dim
    #     J = self.get_Jacobian_gauss_pts(elt_id) # n_quad x dim x dim
    #     detJ = np.linalg.det(J) # n_quad
    #     intgl = np.sum(detJ * func(x) * self.reference.qwts)  # scalar
    
    #     return intgl

    def integrate_over_element(self, elt_id, func):
        intgl = 0.0
        for i in range(self.reference.n_quad):
            x = self.get_coords(elt_id, self.reference.qpts[i])
            J = self.Jacobian(elt_id, self.reference.qpts[i])
            detJ = np.linalg.det(J)
            intgl += detJ * func(x) * self.reference.qwts[i]

        return intgl

    # # 
    # def integrate_over_elements(self, elem_ids, func):
    #     nodes = self.nodes[self.elements[elem_ids]] # n_elements x n_dof x dim
    #     x = self.reference.get_phi_gauss_pts() @ nodes # n_elements x n_quad x dim
    #     # convert x to torch tensor
    #     x = torch.from_numpy(x).float()
    #     J = self.reference.get_d_phi_gauss_pts()[None, :, :, :] @ nodes[:, None, :, :] # n_elements x n_quad x dim x dim
    #     detJ = np.linalg.det(J) # n_elements x n_quad
    #     # convert detJ to torch tensor
    #     detJ = torch.from_numpy(detJ).float()
    #     out = func(x) # n_elements x n_quad
    #     # convert qwts to torch tensor
    #     qwts = torch.from_numpy(self.reference.qwts).float()
    #     intgl = torch.sum(detJ * out * qwts, dim=1) # n_elements
    #     intgl = torch.sum(intgl) # scalar

    #     return intgl

    def get_stuff_GQ_intgl_over_elems(self, elem_ids):
        nodes = self.nodes[self.elements[elem_ids]] # n_elements x n_dof x dim
        x = self.reference.get_phi_gauss_pts() @ nodes # n_elements x n_quad x dim
        J = self.reference.get_d_phi_gauss_pts()[None, :, :, :] @ nodes[:, None, :, :] # n_elements x n_quad x dim x dim
        detJ = np.linalg.det(J) # n_elements x n_quad
        return x, detJ, self.reference.qwts


class FunctionSpace:
    def __init__(self, mesh, reference, n_dof, idx=0):
        self.idx = idx
        self.mesh = deepcopy(mesh)
        self.reference = reference
        self.n_dof = n_dof
        self.dof = np.array([np.nan for _ in range(n_dof)])
        self.dbc = None
        self.n_solve = n_dof
        self.n_dirichlet = 0
        
        self._add_dofs_to_mesh()
        
    def _add_dofs_to_mesh(self):
        pass

    def eval(self, mesh, x, D=None):
        raise NotImplementedError()
        
    def plot(self, mesh):
        raise NotImplementedError()


class Model:
    def __init__(self, mesh, fields, exact=None):
        self.mesh = mesh
        self.exact = exact
        
        self.fields = fields
        self.n_fields = len(fields)

        self.n_dof = [field.n_dof for field in fields]
        cum_dof = [0 for _ in range(self.n_fields)]
        for i in range(1, self.n_fields):
            cum_dof[i] = cum_dof[i-1] + fields[i-1].n_dof
        self.cum_dof = cum_dof
        self.n_tot = np.sum(self.n_dof)

        self.n_dof_elt = [field.reference.n_dof for field in fields]
        cum_dof_elt = [0 for _ in range(self.n_fields)]
        for i in range(1, self.n_fields):
            cum_dof_elt[i] = cum_dof_elt[i-1] + fields[i-1].reference.n_dof
        self.cum_dof_elt = cum_dof_elt
        self.n_tot_elt = np.sum(self.n_dof_elt)

        self.dirichlet = None
        self.neumann = None

        self.node_idx = None
        self.node_idx_inv = None

        self.stiffness = None
        self.load = None

    # Works only for point DOFs!
    def apply_dirichlet_bc(self):
        if self.dirichlet is not None:
            for i, u in enumerate(self.fields):
                bc = []
                for j in u.mesh.boundary[0]:
                    on_bdy, val = self.dirichlet(i, *u.mesh.nodes[j])
                    if on_bdy:
                        bc.append([j, val])

                for j, val in bc:
                    u.dof[j] = val

                u.dbc = bc
                u.n_dirichlet = len(bc)
                u.n_solve = u.n_dof - u.n_dirichlet

    # Works only for point DOFs!
    def apply_neumann_bc(self):
        '''
        self.renumber() needs to be called before this method is called!
        '''
        if self.neumann is not None:
            for i, u in enumerate(self.fields):
                bc = []
                for j in u.mesh.boundary[0]:
                    on_bdy, val = self.neumann(i, *u.mesh.nodes[j])
                    if on_bdy:
                        bc.append([j, val])

                for node, val in bc:
                    ii = self.cum_dof[i] + node
                    self.load[self.node_idx[ii]] += val

    def _compute_inverse_node_indices(self):
        node_idx_inv = np.zeros_like(self.node_idx, dtype=int)

        idx_count = 0
        for n in self.node_idx:
            node_idx_inv[n] = idx_count
            idx_count += 1

        self.node_idx_inv = node_idx_inv

    def renumber(self):
        '''
        self.appy_dirichlet_bc() needs to be called before calling this!
        '''
        node_idx = np.zeros(self.n_tot, dtype=int)
        node_count = 0

        for i_field, field in enumerate(self.fields):
            for i, u in enumerate(field.dof):
                if np.isnan(u):
                    node_idx[self.cum_dof[i_field] + i] = node_count
                    node_count += 1

        for i_field, field in enumerate(self.fields):
            for i, u in enumerate(field.dof):
                if not np.isnan(u):
                    node_idx[self.cum_dof[i_field] + i] = node_count
                    node_count += 1

        self.node_idx = node_idx
        self._compute_inverse_node_indices()

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        raise NotImplementedError()

    def load_kernel(self, i, x, v, grad_v):
        raise NotImplementedError()
        
    def reference_stiffness_matrix(self, elt_id):
        ke = np.zeros((self.n_tot_elt, self.n_tot_elt))
        
        affine = self.mesh.reference.affine
        if affine:
            J = self.mesh.Jacobian(elt_id, None)
            vol = np.abs(np.linalg.det(J))
            J = np.linalg.inv(J).transpose()

        for i_field, ui in enumerate(self.fields):
            n_dof_elt_i = ui.reference.n_dof
            n_quad = ui.reference.n_quad
            xis = ui.reference.qpts
            
            if not affine:
                Js = []
                vols = []
            xs = []
            
            for k in range(n_quad):
                if not affine:
                    J = self.mesh.Jacobian(elt_id, xis[k])
                    vol = np.abs(np.linalg.det(J))
                    J = np.linalg.inv(J).transpose()
                    Js.append(J)
                    vols.append(vol)
                x = self.mesh.get_coords(elt_id, xis[k])
                xs.append(x)
            
            for i in range(n_dof_elt_i):
                ii = self.cum_dof_elt[i_field] + i
                
                for j_field, uj in enumerate(self.fields):
                    n_dof_elt_j = uj.reference.n_dof
                
                    for j in range(n_dof_elt_j):
                        jj = self.cum_dof_elt[j_field] + j
                        gs = np.zeros(n_quad)
                
                        for k in range(n_quad):
                            fi = ui.reference.phi_q[i, k]
                            fj = uj.reference.phi_q[j, k] 
                            if affine:
                                dfi = J @ ui.reference.d_phi_q[i, :, k]
                                dfj = J @ uj.reference.d_phi_q[j, :, k]
                                gs[k] = self.stiffness_kernel(
                                    i_field, j_field, xs[k], fi, fj, dfi, dfj
                                ) * vol
                            else:
                                dfi = Js[k] @ ui.reference.d_phi_q[i, :, k]
                                dfj = Js[k] @ uj.reference.d_phi_q[j, :, k]
                                gs[k] = self.stiffness_kernel(
                                    i_field, j_field, xs[k], fi, fj, dfi, dfj
                                ) * vols[k]
                        
                        ke[ii, jj] = ui.reference.integrate_gauss(gs)
                        
        return ke
           
    def reference_load_vector(self, elt_id):
        fe = np.zeros(self.n_tot_elt)
        
        affine = self.mesh.reference.affine
        if affine:
            J = self.mesh.Jacobian(elt_id, None)
            vol = np.abs(np.linalg.det(J))
            J = np.linalg.inv(J).transpose()

        for i_field, field in enumerate(self.fields):
            n_dof_elt = field.reference.n_dof
            n_quad = field.reference.n_quad
            xis = field.reference.qpts
            
            for i in range(n_dof_elt):
                ii = self.cum_dof_elt[i_field] + i
                gs = np.zeros(n_quad)
                
                for k in range(n_quad):
                    if not affine:
                        J = self.mesh.Jacobian(elt_id, xis[k])
                        vol = np.abs(np.linalg.det(J))
                        J = np.linalg.inv(J).transpose()
                    x = self.mesh.get_coords(elt_id, xis[k])
                    fi = field.reference.phi_q[i, k]
                    dfi = J @ field.reference.d_phi_q[i, :, k]
                    gs[k] = self.load_kernel(i_field, x, fi, dfi)*vol

                fe[ii] = field.reference.integrate_gauss(gs)
                
        return fe
  
    def assemble_stiffness(self):
        II = []
        JJ = []
        V = []
        
        for i_elt in range(self.mesh.n_elements):
            ke = self.reference_stiffness_matrix(i_elt)

            for i_field, ui in enumerate(self.fields):
                elt_i = ui.mesh.elements[i_elt]
                for i in range(self.n_dof_elt[i_field]):
                    ii = self.cum_dof[i_field] + elt_i[i]
                    
                    for j_field, uj in enumerate(self.fields):
                        elt_j = uj.mesh.elements[i_elt]
                        for j in range(self.n_dof_elt[j_field]):
                            jj = self.cum_dof[j_field] + elt_j[j]
                            
                            II.append(self.node_idx[ii])
                            JJ.append(self.node_idx[jj])
                            V.append(ke[self.cum_dof_elt[i_field] + i,
                                        self.cum_dof_elt[j_field] + j])

        K = coo_matrix((V, (II, JJ)), shape=(self.n_tot, self.n_tot))
        self.stiffness = K
  
    def assemble_load(self):
        F = np.zeros(self.n_tot)

        for i_elt in range(self.mesh.n_elements):
            fe = self.reference_load_vector(i_elt)
            
            for i_field, ui in enumerate(self.fields):
                elt = ui.mesh.elements[i_elt]
                for i in range(self.n_dof_elt[i_field]):
                    ii = self.cum_dof[i_field] + elt[i]
                    F[self.node_idx[ii]] += fe[self.cum_dof_elt[i_field] + i]

        self.load = F

    def solve(self):
        self.apply_dirichlet_bc()
        self.renumber()
        
        self.assemble_stiffness()
        self.stiffness = self.stiffness.tocsr()
        
        self.assemble_load()
        self.apply_neumann_bc()

        U_dbc = []
        for i, u in enumerate(self.fields):
            for _, val in u.dbc:
                U_dbc.append(val)
        U_dbc = np.array(U_dbc)

        N = 0
        for u in self.fields:
            N += u.n_solve

        U = spsolve(self.stiffness[:N, :N],
                    self.load[:N] - self.stiffness[:N, N:] @ U_dbc)

        solve_count = 0
        for i_field, u in enumerate(self.fields):
            for i in range(u.n_solve):
                ii = self.node_idx_inv[solve_count] - self.cum_dof[i_field]
                u.dof[ii] = U[solve_count]
                solve_count += 1

    def plot(self, i_field, n_plot):
        raise NotImplementedError()
