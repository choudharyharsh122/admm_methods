from fenics import *
from fenics_adjoint import *
import numpy as np
import math

set_log_active(False)                  # globally disable all DOLFIN log_output
set_log_level(LogLevel.ERROR)

import logging
logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)


# --- Geometry / BC helper --------------------------------

class WestNorth(SubDomain):
    """Top and left sides of the unit square."""
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) or near(x[1], 1.0)) and on_boundary

# --- Core PDE + adjoint wrapper -------------------------

class Solver:
    """
    FEM forward solve + build augmented-Lagrangian ReducedFunctional.
    """
    def __init__(self, mesh, A, P, bc, f, alpha, V_max, p=3.0, eps=1e-3, beta=1, gamma=1e-4, delta=0.01, nu=1e-4):
        self.mesh  = mesh
        self.A, self.P = A, P
        self.bc, self.f = bc, f

        self.p     = Constant(p)
        self.eps   = Constant(eps)
        self.alpha = Constant(alpha)
        self.beta  = Constant(beta)
        self.gamma = Constant(gamma)
        self.V_max = Constant(V_max)
        self.delta = Constant(delta)
        self.nu   = Constant(nu)
        self.eps_tv = 1e-4

        # bind measure to mesh so dx always knows its domain
        self.dx = Measure("dx", domain=mesh)

    def simp_k(self, b):
        """SIMP conductivity: k(b) = eps + (1-eps)*b^p"""   
        #return self.beta*b + self.gamma*(1 - b)
        return self.eps + (1-self.eps)*b**(self.p)
    

    def forward(self, b):
        """Solve PDE """
        u = Function(self.P, name="Temperature")
        v = TestFunction(self.P)
        F = inner(grad(v), self.simp_k(b)*grad(u))*self.dx \
            - self.f*v*self.dx

        solve(F == 0, u, self.bc,
            solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-5,
                    "maximum_iterations": 20,
                    "report": False,            # <� disable per-step reports
                    "error_on_nonconvergence": False
                }
            })
        return u
    

    def compute_Objs(self, a_np, b_np, lam_np, rho, graph, scale):

        a   = Function(self.A, name="a");   a.vector().set_local(a_np)
        b   = Function(self.A, name="b");   b.vector().set_local(b_np)
        lam = Function(self.A, name="lambda"); lam.vector().set_local(lam_np)

        u = self.forward(b)


        AL_form = self.f*u*self.dx + (rho/2)*inner(a - b + lam, a - b + lam)*self.dx
        ALhat = assemble(AL_form)

        J_form = self.f*u*self.dx
        Jhat = assemble(J_form)

        rest_form =  (rho/2) * inner(a - b + (lam) , a - b + (lam))*self.dx
        rest = assemble(rest_form)
        

        with stop_annotating():
            grads_full_L = compute_gradient(ALhat, Control(b))
        grads_full_L = grads_full_L.vector().get_local()
       

        tape = get_working_tape()
        tape.clear_tape()
        
        return  float(ALhat), float(Jhat), float(rest), grads_full_L


    def lagrangian(self, a, b, lam, rho, graph, scale):
        """
        Build and return a fenics_adjoint ReducedFunctional
        """
        u = self.forward(b)

        #AL_form  = self.f*u*self.dx + (rho/2) * inner(a - b + (lam) , a - b + (lam))*self.dx

        AL_form = self.f*u*self.dx + (rho/2)*inner(a - b + lam, a - b + lam)*self.dx
        ALhat = assemble(AL_form)
       

        # 2) assemble under AD context ? AnnotatedScalar
        

        #grads = compute_gradient(ALhat, Control(b))

        # 3) wrap in Control and ReducedFunctional
        return ReducedFunctional(ALhat, Control(b)), self.dx

# --- Volume constraint ---------------

class VolumeConstraint(InequalityConstraint):
    """
    Volume constraint
    """
    def __init__(self, V_max, space):
        self.V_max = float(V_max)
        self.fs    = space
        # lumped mass for ? b dx
        self.mass  = assemble(TestFunction(self.fs)*Constant(1.0)*dx)
        self.tmp   = Function(self.fs)

    def function(self, b_numpy):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmp, b_numpy)
        vol = self.mass.inner(self.tmp.vector())
        return [self.V_max - vol]

    def jacobian(self, b_numpy):
        # derivative g'(b) = -mass
        return [-self.mass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        return 1

# --- Subproblem 1: IPOPT wrapper -------------------------

class Subproblem1Solver:
    """
    ADMM subproblem 1: given numpy arrays,
    solve problem via IPOPT.
    """
    def __init__(self, core_solver, space_A):
        self.core   = core_solver
        #self.rho    = Constant(rho)
        #self.V_max  = V_max
        self.A      = space_A


    def solve(self, a_np, b_np, lam_np, rho, graph, scale):
        # 1) lift numpy?FEniCS Functions
        a   = Function(self.A, name="a");   a.vector().set_local(a_np)
        b   = Function(self.A, name="b");   b.vector().set_local(b_np)
        lam = Function(self.A, name="lambda"); lam.vector().set_local(lam_np)


        # 2) build the ReducedFunctional
        Lhat, dx = self.core.lagrangian(a, b, lam, rho, graph, scale)

        # 3) box bounds 0 <= b <= 1
        lb, ub = 0.0, 1.0

        # 4) volume constraint
        vc = VolumeConstraint(self.core.V_max, self.A)

        # 5) IPOPT problem + solve
        problem = MinimizationProblem(Lhat,
                                      bounds=[(lb, ub)],
                                      constraints=vc)
        solver  = IPOPTSolver(
            problem,
            parameters={
                "acceptable_tol":     1e-3,
                "maximum_iterations": 30,
                "print_level":        0,   #  no IPOPT console output
                "file_print_level":   0    #  no file output either
            }
        )
        b_opt = solver.solve()

        u_opt = self.core.forward(b_opt)

        volume = float(assemble(b_opt * dx))


        step = (1/(len(a_np))) * float(assemble(1*dx))
        #print(">>>>>>>>>>>>>>>>Volume =", volume)

        tape = get_working_tape()
        tape.clear_tape()

        return b_opt, step, u_opt
