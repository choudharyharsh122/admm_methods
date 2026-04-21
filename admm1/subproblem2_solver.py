import gurobipy as gp
from gurobipy import GRB
import mergesplit.mergesplit as ms
import networkx as nx
import numpy as np
from fenics import *
import math

class Subproblem2Solver:
    def __init__(self, mesh, n_x, n_y, alpha, seed):
        """
        n_x, n_y : ints
            dimensions of your 2D grid
        alpha : float
            TV weight
        seed : int
            RNG seed for mergesplit
        """
        self.n = n_x * n_y
        self.alpha = alpha
        self.seed = seed
        self.mesh = mesh

        # build the graph once
        #self.graph = self._build_graph(n_x, n_y)
        self.graph = self.build_graph(n_x, n_y)

        # precompute the per-edge scale factors
        # (so you dont recompute abs(u-v)==1 each iteration)
        self.scale = np.zeros(len(self.graph.edges()))
        for k, (u, v) in enumerate(self.graph.edges()):
            self.scale[k] = math.sqrt(2) if abs(int(u) - int(v)) == 1 else 1.0  

    def compute_TV(self, a, b, lam, rho):
        """Total variation term at (a,b,lam,rho)"""
        # note: lam, rho arent used here  but signature stays same
        diffs = []
        for (u, v), s in zip(self.graph.edges(), self.scale):
            diffs.append(s * abs(a[u] - a[v]))
        Gg = sum(diffs)
        return (1.0 / math.sqrt(len(a)/2)) * Gg * self.alpha

    def computeF(self, a, b, lam, rho):
        """Quadratic penalty term"""
        # lam and rho now feed into F
        return ((rho/2) * (a - b + lam)**2).sum() / len(b)

    # def _build_graph(self, mesh):
    #     G = nx.Graph()
    #     num_cells = mesh.num_cells()
    #     G.add_nodes_from(range(num_cells))

    #     # get the connectivity: cell → facets → neighboring cells
    #     mesh.init(2, 1)
    #     mesh.init(1, 2)

    #     for cell in cells(mesh):
    #         cid = cell.index()
    #         for facet in facets(cell):
    #             for neighbor in facet.entities(2):
    #                 if neighbor != cid:
    #                     G.add_edge(cid, neighbor)

    #     return G

    def build_graph(self, n_x, n_y):
        N = n_x * n_y
        graph = nx.Graph()
        graph.add_nodes_from(range(N))

        for k in range(0, N, 2):
            if k + 1 < N:
                graph.add_edge(k, k + 1)
            if (k + 2) % n_y != 0 and k + 3 < N:
                graph.add_edge(k, k + 3)
            if (k // n_y) != 0:
                nb = k - (n_y - 1)
                if nb >= 0:
                    graph.add_edge(k, nb)

        return graph

    def run(self, a, b, lam, rho, V_max, seed, backend):
            """
            Solve the subproblem with the selected backend.

            Parameters
            ----------
            backend : {'mergesplit','gurobi'}
                Which implementation to use.
            return_raw : bool
                If True and backend='mergesplit', also return the raw updown object.

            Returns
            -------
            x : np.ndarray or None
                Binary solution (0/1) when available. None if no feasible solution.
            status : int or str
                Backend-specific status (e.g., Gurobi status code, 'OK'/'FAIL' for mergesplit).
            raw : object (optional)
                Only returned if return_raw=True for 'mergesplit'; the PyUpDownMergeSplit object.
            """
            if backend == "mergesplit":
                x, status = self._run_mergesplit(a, b, lam, rho, V_max, seed)
                return x, status
            elif backend == "gurobi":
                x, status = self._run_gurobi(a, b, lam, rho, V_max, seed)
                return x, status
            else:
                raise ValueError(f"Unknown backend '{backend}'. Use 'mergesplit' or 'gurobi'.")

    # ---------- backend implementations ----------
    def _run_mergesplit(self, a, b, lam, rho, V_max, seed):
        """
        Original mergesplit implementation. Tries to return an np.array solution too.
        """
        F = lambda x: ((rho/2) * (x - b + lam)**2) / len(b)
        G = lambda y: (self.alpha * self.scale * np.abs(y)) / math.sqrt(len(b)/2)
        H = lambda x: x.flatten()

        updown = ms.PyUpDownMergeSplit(
            self.graph, F, G, H, 1,
            trust_region_active=True,
            delta=V_max * len(b),
            seed=seed,
            efficiency_ordering=True
        )
        updown.initialize(a.astype(np.int32))
        updown.optimize()
        
        sol = updown.x
        
        status = "OK" if sol is not None else "FAIL"
        
        return sol, status

    def _run_gurobi(self, a, b, lam, rho, V_max, seed):
        """
        Original Gurobi implementation. Returns (x, status).
        """
        N = len(self.graph.nodes)
        E = list(self.graph.edges())

        m = gp.Model("graph_binary_opt")
        m.Params.OutputFlag = 0
        m.Params.Seed = int(seed)

        # Binary decision vars: one per node
        w = m.addMVar(N, vtype=GRB.BINARY, name="w")
        # Start from 'a' if provided
        try:
            w.Start = a
        except Exception:
            pass

        # Budget constraint
        m.addConstr(w.sum() <= V_max * N, name="budget")

        # Quadratic penalty (scaled)
        expr = w - b + lam
        quad_term = (rho/2) * (expr @ expr) / len(b)

        # TV term using absolute differences on edges
        tv_terms = []
        for k, (i, j) in enumerate(E):
            d = m.addVar(vtype=GRB.BINARY, name=f"d_{i}_{j}")
            m.addConstr(d >=  w[i] - w[j])
            m.addConstr(d >=  w[j] - w[i])
            tv_terms.append(self.alpha * self.scale[k] * d)
        tv_term = gp.quicksum(tv_terms) / math.sqrt(len(b))

        m.setObjective(quad_term + tv_term, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount > 0:
            x = np.asarray(w.X, dtype=int)
            return x, m.Status
        else:
            return None, m.Status