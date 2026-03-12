import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class Subproblem1Solver:
    """
    Subproblem 1 solver on a triangular UnitSquareMesh(dim, dim)-type grid.

    PDE:
        -div(k(b) grad u) = f   in Omega
        u = 0                   on west U north
        k grad u . n = 0        on bottom U right

    Material interpolation:
        k(b) = eps + (1-eps) * b^penal

    Main API:
        sub1 = Subproblem1Solver(dim, f, alpha=alpha, graph=graph, scale=scale)
        b_opt, U_opt = sub1.solve(a, b, lam, rho)
        compliance, None, None, None = sub1.compute_Objs(a_k, b_k, lam_k, rho_k)
    """

    def __init__(self, dim, f, volfrac, alpha, graph=None, scale=None,
                 penal=3.0, eps=1e-3, move=0.2, tol=1e-2, maxiter=50):
        self.dim = dim
        self.n = dim
        self.f = f
        self.volfrac = volfrac

        # kept for later use, as requested
        self.alpha = alpha
        self.graph = graph
        self.scale = scale

        self.penal = penal
        self.eps = eps
        self.move = move
        self.tol = tol
        self.maxiter = maxiter

        self.h = 1.0 / self.n

        # mesh + boundary data
        (
            self.coords,
            self.tris,
            self.tri_type,
            self.bottom_edges,
            self.right_edges,
            self.fixeddofs
        ) = self._build_unitsquaremesh_right_tri(self.n)

        self.ndof = self.coords.shape[0]
        self.nele = self.tris.shape[0]

        alldofs = np.arange(self.ndof, dtype=int)
        self.freedofs = np.setdiff1d(alldofs, self.fixeddofs)

        # reference triangle stiffnesses
        self.KE0, self.KE1, self.KE_all = self._prepare_element_matrices()

        # assembly pattern
        self.iK = np.repeat(self.tris, 3, axis=1).ravel()
        self.jK = np.tile(self.tris, (1, 3)).ravel()

        # load vector for body force f and zero Neumann on bottom/right
        self.F = self._build_load_vector()

        # element weights v_e in the OC update; kept simple for now
        self.v = np.ones(self.nele, dtype=float)

    # ------------------------------------------------------------------
    # mesh / FEM helpers
    # ------------------------------------------------------------------

    def _tri3_stiffness(self, xy):
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]

        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area <= 0:
            raise ValueError("Triangle must be counterclockwise.")

        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)
        B = np.vstack([b, c]) / (2.0 * area)

        return area * (B.T @ B)

    def _build_unitsquaremesh_right_tri(self, n):
        h = 1.0 / n
        nn = n + 1

        nodenrs = np.arange(nn * nn).reshape((nn, nn), order="C")

        coords = np.zeros((nn * nn, 2), dtype=float)
        for j in range(nn):
            for i in range(nn):
                coords[nodenrs[j, i]] = [i * h, j * h]

        tris = []
        tri_type = []
        bottom_edges = []
        right_edges = []

        for i in range(n):
            for j in range(n):
                bl = nodenrs[j, i]
                br = nodenrs[j, i + 1]
                tl = nodenrs[j + 1, i]
                tr = nodenrs[j + 1, i + 1]

                # default "right" diagonal: bl -> tr
                tris.append([bl, br, tr]); tri_type.append(0)
                tris.append([bl, tr, tl]); tri_type.append(1)

                if j == 0:
                    bottom_edges.append([bl, br])
                if i == n - 1:
                    right_edges.append([br, tr])

        tris = np.asarray(tris, dtype=int)
        tri_type = np.asarray(tri_type, dtype=int)
        bottom_edges = np.asarray(bottom_edges, dtype=int)
        right_edges = np.asarray(right_edges, dtype=int)

        west_nodes = nodenrs[:, 0]
        north_nodes = nodenrs[-1, :]
        dirichlet_nodes = np.unique(np.r_[west_nodes, north_nodes]).astype(int)

        return coords, tris, tri_type, bottom_edges, right_edges, dirichlet_nodes

    def _prepare_element_matrices(self):
        h = self.h
        KE0 = self._tri3_stiffness(np.array([[0.0, 0.0], [h, 0.0], [h, h]], dtype=float))
        KE1 = self._tri3_stiffness(np.array([[0.0, 0.0], [h, h], [0.0, h]], dtype=float))

        KE_all = np.zeros((self.nele, 3, 3), dtype=float)
        KE_all[self.tri_type == 0] = KE0
        KE_all[self.tri_type == 1] = KE1
        return KE0, KE1, KE_all

    def _build_load_vector(self):
        area_tri = 0.5 * self.h * self.h
        fe = (self.f * area_tri / 3.0) * np.ones(3)

        F = np.zeros(self.ndof, dtype=float)
        for e in range(self.nele):
            F[self.tris[e]] += fe

        # zero Neumann on bottom/right => nothing extra to add
        return F

    def _solve_state(self, b):
        kvec = self.eps + (1.0 - self.eps) * b**self.penal
        sK = (self.KE_all.reshape(self.nele, 9) * kvec[:, None]).ravel()

        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()

        U = np.zeros(self.ndof, dtype=float)
        U[self.freedofs] = spsolve(K[self.freedofs][:, self.freedofs], self.F[self.freedofs])
        U[self.fixeddofs] = 0.0
        return U

    # ------------------------------------------------------------------
    # OC update with bisection in mu
    # ------------------------------------------------------------------

    def _oc_update_augmented(self, a, b, lam, rho, ce, volfrac, safe_eps=1e-12):
        """
        Update:
            b_new = b * sqrt( numer / denom )

        numer = g'(b) * ce
        denom = v * mu + lam + rho * (b - a + lam)

        mu is found by bisection so that mean(b_new) = volfrac.
        """
        gprime = (1.0 - self.eps) * self.penal * np.maximum(b, 1e-5)**(self.penal - 1)
        numer = np.maximum(gprime * ce, 0.0)

        shift = 1/len(b) * (rho*(b - a + lam))

        # ensure denom > 0: v*mu + shift > 0
        #mu_low = np.max(-shift / np.maximum(self.v, safe_eps)) + safe_eps
        mu_low = max(0.0, np.max(-shift / np.maximum(self.v, safe_eps))) + safe_eps
        #mu_high = max(2.0 * mu_low, 1.0)
        mu_high = 1e5
        gprime = (1.0 - self.eps) * self.penal * np.maximum(b, 1e-5)**(self.penal - 1)
        phys = gprime * ce
        aug  = (1/len(b)) * rho * (b - a + lam)
        vol_cons = self.v * mu_low

        print("phys min/max:", phys.min(), phys.max())
        print("aug  min/max:", aug.min(), aug.max())
        print("ratio min/max:", (phys / np.maximum(np.abs(aug + vol_cons),1e-14)).min(),
                                (phys / np.maximum(np.abs(aug + vol_cons),1e-14)).max())
        print("aug  min/max:", aug.min(), aug.max())
        print("mean b:", b.mean())

        def trial_update(mu):
            denom = self.v * mu + shift
            denom = np.maximum(denom, safe_eps)

            b_candidate = b * np.sqrt(numer / denom)

            b_new = np.maximum(
                1e-3,
                np.maximum(
                    b - self.move,
                    np.minimum(1.0, np.minimum(b + self.move, b_candidate))
                )
            )
            return b_new

        # enlarge upper bracket until target volume is below
        b_test = trial_update(mu_high)
        while b_test.mean() > volfrac:
            mu_high *= 2.0
            b_test = trial_update(mu_high)
            if mu_high > 1e16:
                break

        while (mu_high - mu_low) > 1e-6:
            mu_mid = 0.5 * (mu_low + mu_high)
            b_mid = trial_update(mu_mid)

            if b_mid.mean() > volfrac:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
        
        mu = 0.5 * (mu_low + mu_high)
        print("Final vol constraint (min/max):", np.min(self.v * mu), np.max(self.v * mu))
        b_new = trial_update(mu)
        return b_new, mu

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def solve(self, a, b, lam, rho):
        """
        Solve the b-subproblem:
            compliance(b) + (rho/2) ||a - b + lam||^2

        Returns
        -------
        b : final control
        U : final PDE state at that control
        """
        b = np.asarray(b, dtype=float).copy()
        a = np.asarray(a, dtype=float)
        lam = np.asarray(lam, dtype=float)

        change = 1.0
        loop = 0

        while change > self.tol and loop < self.maxiter:
            loop += 1
            b_old = b.copy()

            U = self._solve_state(b)

            Ue = U[self.tris]
            ce = np.einsum("ei,eij,ej->e", Ue, self.KE_all, Ue)

            print("b shape:", b.shape)
            print("a shape:", a.shape)
            print("lam shape:", lam.shape)
            print("Ue shape:", Ue.shape)
            print("ce shape:", ce.shape)

            b, mu = self._oc_update_augmented(
                a=a,
                b=b,
                lam=lam,
                rho=rho,
                ce=ce,
                volfrac=self.volfrac
            )
            
            print("b shape:",b.shape)
            print("b: ", b)

            change = np.max(np.abs(b - b_old))

        U = self._solve_state(b)
        return b, U

    def compute_Objs(self, a_k, b_k, lam_k, rho_k):
        """
        As requested:
        returns compliance, None, None, None
        """
        b_k = np.asarray(b_k, dtype=float)
        U = self._solve_state(b_k)
        compliance = float(self.F @ U)
        pen_k = (rho_k / 2.0) * np.sum((b_k - a_k + lam_k)**2) / len(b_k)
        obj = compliance + pen_k
        return obj, compliance, pen_k, None
