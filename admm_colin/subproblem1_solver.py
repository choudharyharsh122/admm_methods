import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def cubic_roots_cardano(a, b, c, d, tol=1e-14):
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    if not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("All coefficient arrays must have the same shape.")
    if np.any(np.abs(a) < tol):
        raise ValueError("Leading coefficient 'a' must be nonzero.")

    # Normalize: x^3 + A x^2 + B x + C = 0
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic y^3 + p y + q = 0 with x = y - A/3
    p = B - A**2 / 3
    q = 2*A**3 / 27 - A*B / 3 + C

    Delta = (q/2)**2 + (p/3)**3
    sqrtDelta = np.sqrt(Delta)

    z1 = -q/2 + sqrtDelta
    z2 = -q/2 - sqrtDelta

    u = np.power(z1, 1/3)
    v = np.zeros_like(u)

    # Use uv = -p/3 whenever u is not too small
    mask_u = np.abs(u) > tol
    v[mask_u] = -p[mask_u] / (3*u[mask_u])

    # Where u is too small, use z2 instead
    mask_small_u = ~mask_u
    if np.any(mask_small_u):
        v_alt = np.power(z2[mask_small_u], 1/3)
        u_alt = np.zeros_like(v_alt)

        mask_v = np.abs(v_alt) > tol
        u_alt[mask_v] = -p[mask_small_u][mask_v] / (3*v_alt[mask_v])

        # If both u and v are ~0, then p=q=0 and the depressed cubic is y^3=0,
        # so all three roots are just 0 before shifting.
        u[mask_small_u] = u_alt
        v[mask_small_u] = v_alt

    omega = -0.5 + 0.5j*np.sqrt(3)
    omega2 = -0.5 - 0.5j*np.sqrt(3)

    shift = A / 3

    r1 = u + v - shift
    r2 = omega*u + omega2*v - shift
    r3 = omega2*u + omega*v - shift

    return r1, r2, r3


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
        self.last_bisection_history = []

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

        for j in range(n):
            for i in range(n):
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

        #shift = 1/len(b) * (rho*(b - a + lam))

        # ensure denom > 0: v*mu + shift > 0
        #mu_low = np.max(-shift / np.maximum(self.v, safe_eps)) + safe_eps
        mu_low = -1e5 #max(0.0, np.max(-shift / np.maximum(self.v, safe_eps))) + safe_eps
        #mu_high = max(2.0 * mu_low, 1.0)
        mu_high = 1e5
        gprime = (1.0 - self.eps) * self.penal * np.maximum(b, 1e-5)**(self.penal - 1)
        #phys = gprime * ce
        #aug  = (1/len(b)) * rho * (b - a + lam)
        #vol_cons = self.v * mu_low

        # print("phys min/max:", phys.min(), phys.max())
        # print("aug  min/max:", aug.min(), aug.max())
        # print("ratio min/max:", (phys / np.maximum(np.abs(aug + vol_cons),1e-14)).min(),
        #                         (phys / np.maximum(np.abs(aug + vol_cons),1e-14)).max())
        # print("aug  min/max:", aug.min(), aug.max())
        # print("mean b:", b.mean())

        def trial_update(mu):

            # Coefficients of cubic
            c0 = -gprime * ce * b ** 2
            c1 = np.zeros_like(c0)
            c2 = (lam + mu * self.v - rho * a)/self.nele
            c3 = rho/self.nele * np.ones_like(c0)
            
            b_candidate = np.real(cubic_roots_cardano(c3, c2, c1, c0)[0])

            b_new = np.maximum(
                self.eps,
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

        bisection_history = []
        while (mu_high - mu_low) > 1e-6:
            mu_mid = 0.5 * (mu_low + mu_high)
            b_mid = trial_update(mu_mid)
            b_mid_mean = float(b_mid.mean())
            bisection_history.append((float(mu_mid), b_mid_mean))

            if b_mid_mean > volfrac:
                mu_low = mu_mid
            else:
                mu_high = mu_mid

        self.last_bisection_history = bisection_history
        #print("Bisection history (mu, mean(b)):")
        #for idx, (mu_i, mean_b_i) in enumerate(bisection_history, start=1):
            #print(f"  iter {idx:03d}: mu={mu_i:.10e}, mean(b)={mean_b_i:.10e}")
        
        mu = 0.5 * (mu_low + mu_high)
        #print("Final vol constraint (min/max):", np.min(self.v * mu), np.max(self.v * mu))
        b_new = trial_update(mu)
        return b_new, mu

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def solve(self, a, b, lam, rho, track_oc_convergence=False):
        """
        Solve the b-subproblem:
            compliance(b) + (rho/2) ||a - b + lam||^2

        Returns
        -------
        b : final control
        U : final PDE state at that control
        track_data : dict or None
            Convergence data when track_oc_convergence=True, else None.
            Dict keys:
            - "F_list": list of F(b) over OC iterations (including final b)
            - "grad_F_list": list of gradients ∇F(b) over OC iterations
            - "grad_F_norm_list": list of ||∇F(b)||_2 over OC iterations
        """
        b = np.asarray(b, dtype=float).copy()
        a = np.asarray(a, dtype=float)
        lam = np.asarray(lam, dtype=float)

        F_list = []
        grad_F_list = []
        grad_F_norm_list = []

        change = 1.0
        loop = 0

        while change > self.tol and loop < self.maxiter:
            loop += 1
            b_old = b.copy()

            if track_oc_convergence:
                L_k, grad_L_k = self.compute_augmented_lagrangian_and_grad(
                    b=b,
                    a=a,
                    lam=lam,
                    rho=rho,
                )
                F_list.append(float(L_k))
                grad_F_list.append(grad_L_k.copy())
                grad_F_norm_list.append(float(np.linalg.norm(grad_L_k)))

            U = self._solve_state(b)

            Ue = U[self.tris]
            ce = np.einsum("ei,eij,ej->e", Ue, self.KE_all, Ue)

            # print("b shape:", b.shape)
            # print("a shape:", a.shape)
            # print("lam shape:", lam.shape)
            # print("Ue shape:", Ue.shape)
            # print("ce shape:", ce.shape)

            b, mu = self._oc_update_augmented(
                a=a,
                b=b,
                lam=lam,
                rho=rho,
                ce=ce,
                volfrac=self.volfrac
            )
            
            # print("b shape:",b.shape)
            # print("b: ", b)

            change = np.max(np.abs(b - b_old))
            # print("change:", change)

        U = self._solve_state(b)
        if track_oc_convergence:
            # Record convergence metrics at the final control as well.
            L_final, grad_L_final = self.compute_augmented_lagrangian_and_grad(
                b=b,
                a=a,
                lam=lam,
                rho=rho,
            )
            F_list.append(float(L_final))
            grad_F_list.append(grad_L_final.copy())
            grad_F_norm_list.append(float(np.linalg.norm(grad_L_final)))

            track_data = {
                "F_list": np.array(F_list, dtype=float),
                "grad_F_list": np.vstack(grad_F_list) if len(grad_F_list) > 0 else np.zeros((0, len(b))),
                "grad_F_norm_list": np.array(grad_F_norm_list, dtype=float),
            }
            return b, U, track_data

        return b, U, None

    def compute_Objs(self, a_k, b_k, lam_k, rho_k):
        """
        returns compliance, None, None, None
        """
        b_k = np.asarray(b_k, dtype=float)
        a_k = np.asarray(a_k, dtype=float)
        lam_k = np.asarray(lam_k, dtype=float)

        U = self._solve_state(b_k)
        compliance = float(self.F @ U)
        diff = (b_k - a_k)
        lag_k = float(np.sum(lam_k * diff))
        pen_k = (rho_k / 2.0) * np.sum(diff**2)
        obj = compliance + lag_k + pen_k
        return obj, compliance, pen_k, None

    # ------------------------------------------------------------------
    # OC convergence tracking helpers
    # ------------------------------------------------------------------

    def compute_F_and_grad(self, a):
        
        a = np.asarray(a, dtype=float)
        U = self._solve_state(a)
        F_val = float(self.F @ U)

        Ue = U[self.tris]
        ce = np.einsum("ei,eij,ej->e", Ue, self.KE_all, Ue)
        gprime = (1.0 - self.eps) * self.penal * np.maximum(a, 1e-5) ** (self.penal - 1)
        grad_F = -ce * gprime

        return F_val, grad_F

    def compute_augmented_lagrangian_and_grad(self, b, a, lam, rho):
        
        b = np.asarray(b, dtype=float)
        a = np.asarray(a, dtype=float)
        lam = np.asarray(lam, dtype=float)
        rho = float(rho)

        F_val, grad_F = self.compute_F_and_grad(b)
        diff = b - a

        L_val = float(F_val + np.sum(lam * diff) + 0.5 * rho * np.sum(diff**2))
        grad_L = grad_F + lam + rho * diff
        return L_val, grad_L
