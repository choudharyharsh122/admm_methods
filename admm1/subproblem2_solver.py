from graph_tv import chambolle_pock_graph_tv
import gurobipy as gp
from gurobipy import GRB
import mergesplit.mergesplit as ms
import networkx as nx
import numpy as np
from fenics import *
import math

class Subproblem2Solver:
    def __init__(self, n_x, n_y, alpha, seed):
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
            elif backend == "chambolle-pock":
                x, status = self._run_chambolle_pock(a, b, lam, rho, V_max)
                return x, status
            else:
                raise ValueError(f"Unknown backend '{backend}'. Use 'mergesplit', 'gurobi', or 'chambolle-pock'.")

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

        #w = m.addMVar(N, vtype=GRB.BINARY, name="w")
        w = m.addMVar(N, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="w")
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
            #d = m.addVar(vtype=GRB.BINARY, name=f"d_{i}_{j}")
            d = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"d_{i}_{j}")
            m.addConstr(d >=  w[i] - w[j])
            m.addConstr(d >=  w[j] - w[i])
            tv_terms.append(self.alpha * self.scale[k] * d)
        tv_term = gp.quicksum(tv_terms) / math.sqrt(len(b))

        m.setObjective(quad_term + tv_term, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount > 0:
            x = np.asarray(w.X, dtype=float)
            return x, m.Status
        else:
            return None, m.Status
        
    def _run_chambolle_pock(self, a, b, lam, rho, V_max):
        n = len(b)
        budget = V_max * n
        edges = np.asarray(list(self.graph.edges()), dtype=int)

        # ADMM quadratic term in y-update (scaled by 1/n to match subproblem1):
        # (1/n) * [lambda.(b-y) + (rho/2)||b-y||^2]
        #   = (rho/(2n)) y^2 + ((-rho*b - lambda)/n) y + const
        # TV term scaled by 1/sqrt(n/2) to match.
        a_quad = np.full(n, rho / (2.0 * n), dtype=float)
        b_lin = (-rho * np.asarray(b, dtype=float) + rho * np.asarray(lam, dtype=float)) / n
        alpha_scaled = self.alpha / np.sqrt(n / 2.0)

        x_init = np.asarray(a, dtype=float) if a is not None else None
        sol, info = chambolle_pock_graph_tv(
            n_vertices=n,
            edges=edges,
            a=a_quad,
            b=b_lin,
            budget=budget,
            alpha=alpha_scaled,
            x_lo=0.0,
            max_iter=2000,
            tol=1e-8,
            x_init=x_init,
            edge_weights=self.scale,
        )

        status = "OK" if info.get("converged", False) else "MAX_ITER"
        return np.asarray(sol, dtype=float), status