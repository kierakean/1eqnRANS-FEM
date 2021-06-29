#Michael Schneier's code
#Some changes thanks to Michael McLaughlin
#Some other changes due to Kiera Kean
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pdb
from os import sys
import os


#Solver
MAX_ITERS = 10000
solver = PETScKrylovSolver('gmres','none')
solver.ksp().setGMRESRestart(MAX_ITERS)
solver.parameters["monitor_convergence"] = False
solver.parameters['relative_tolerance'] = 1e-6
solver.parameters['maximum_iterations'] = MAX_ITERS


#Pi to high accuracy
Pi = np.pi

#DOF/Mesh input
N =45
meshrefine = 1 # 0,1, or 2: how many times boundary is refined
refine1 = .03 #distance from wall refined first time
refine2 = .03 #distance from wall refined second time
#Timestep
dt = .01

#Initial, final time, total number of steps
t_init = 0.0
t_final = 50
t_num = int((t_final - t_init)/dt)
t = t_init

#Viscosity
nu = .0015

#Parameters for model
mu = .55
eps = 1.e-6
tau = 0.0

#Choice of mxing length
# 0 = NSE (no turbulent viscosity)
# 2 = l = min{sqrt(2k)tau,.41d sqrt{d/L}}
# 4 = l = .41d
l_choice = 4


#Final angular velocity of inner and outer cylinders
omega_inner = 4 
omega_outer =0
omega_diff = abs(omega_inner-omega_outer)

#Time it takes to reach full speed of cylinders
x = 5
#Turn k equation on when cylinder is at full speed
k_on = int(x/dt)


#Paraview Setup
savespersec = 20. #How many snapshots are taken per second
velocity_paraview_file = File("vfolder/3d_TC_"+str(N)+"_"+str(dt)+"_"+str(omega_diff)+".pvd")
snapspersec = int(1./dt) #timesteps per second
frameRate = int(snapspersec/savespersec) # how often snapshots are saved to paraview
if frameRate ==0:
    frameRate =1


#Define Domain
outer_top_radius =1
inner_top_radius = .5
outer_bot_radius = outer_top_radius
inner_bot_radius = inner_top_radius

b_ox = 0.
b_oy = 0.
b_oz = 0.

t_ox = 0.
t_oy = 0.
t_oz = 2.2

b_ix = 0.
b_iy = 0.
b_iz = 0.

t_ix = 0.
t_iy = 0.
t_iz = 2.2

domain = Cylinder(Point(b_ox, b_oy, b_oz), Point(t_ox, t_oy, t_oz), outer_top_radius, outer_bot_radius) - Cylinder(Point(b_ix, b_iy, b_iz), Point(t_ix, t_iy, t_iz), inner_top_radius, inner_bot_radius)


#Generate mesh 
mesh = generate_mesh ( domain, N )

#Refine mesh
if (meshrefine > .5): #mesh refine greater than zero: refine once
    sub_domains_bool = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
    sub_domains_bool.set_all(False)

    class SmallInterest(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1]**2 + x[0]**2 < (inner_top_radius+refine1)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine1)**2)


    interest = SmallInterest()
    interest.mark(sub_domains_bool,True)
    mesh = refine(mesh,sub_domains_bool)


    if (meshrefine > 1.5): #Greater than 2, refine a second time
        class BigInterest(SubDomain):
            def inside(self, x, on_boundary):
                return (x[1]**2 + x[0]**2 < (inner_top_radius+refine2)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine2)**2)
        #
        sub_domains_bool2 = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
        sub_domains_bool2.set_all(False)
        interest = BigInterest()
        interest.mark(sub_domains_bool2,True)
        mesh = refine(mesh,sub_domains_bool2)



hmax = mesh.hmax()
hmin = mesh.hmin()

#Reynolds/Taylor number information
height = t_iz - b_iz #height of the cylinder
radius_ratio = inner_bot_radius/outer_bot_radius #eta
L =  outer_bot_radius - inner_bot_radius
aspect_ratio =  height/L
domain_volume = assemble(1*dx(mesh))
Re = abs(L*(omega_outer-omega_inner)/nu) #Reynold's number as defined in Bilson/Bremhorst
Ta =  Re*Re*4*(1-radius_ratio)/(1+radius_ratio)




#Boundary Conditions: Setup and Definition
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Bottom of the cylinder is the "target domain"
    def inside(self, x, on_boundary):
        return x[2] <= b_iz + DOLFIN_EPS and on_boundary

    # Map top of the cylinder to the bottom of the cylinder
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - height

#Sub domain for the inner cylinder
class Inner_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]**2 + x[1]**2 <= inner_top_radius**2 +10**-2 and on_boundary

#Sub domain for the outer cylinder
class Outer_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return outer_top_radius**2 - 10**-2 <= x[0]**2 + x[1]**2 and on_boundary

#Specify a point on the boundary for the pressures
mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]) and near(x[2], mesh_points[0,2]) )
# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

# Mark all facets as sub domain 0
sub_domains.set_all(0)


#Mark the inner cylinder as sub domain 1
inner_cyl = Inner_cyl()
inner_cyl.mark(sub_domains, 1)

#Mark the outer cylinder as sub domain 2
outer_cyl= Outer_cyl()
outer_cyl.mark(sub_domains, 2)


#Smooth bridge for increasing inner angular velocity
def smooth_bridge(t):
    s = t/x
    #Smoothly increase from 0 at t=0 to 1 at t=s
    if(s>1+1e-14):
        return 1.0
    elif(abs(1-s)>1e-14):
        return np.exp(-np.exp(-1./(1-s)**2)/s**2)
    else:
        return 1.0
mint_val = smooth_bridge(t)


#Taylor Hood element creation
V_h = VectorElement("Lagrange", mesh.ufl_cell(), 2) #Velocity space
Q_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #Pressure space
K_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #TKE space

W = FunctionSpace(mesh,MixedElement([V_h,Q_h]),constrained_domain=PeriodicBoundary())
K = FunctionSpace(mesh,K_h,constrained_domain=PeriodicBoundary())


#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
k = TrialFunction(K)
phi = TestFunction(K)

#Solution vectors
w_ = Function(W)
wnMinus1 = Function(W)
(unMinus1,pnMinus1) = wnMinus1.split(True) #Velocity solution vector at time n
wn = Function(W)
(un,pn) = wn.split(True) #Velocity solution vector at time n
wnPlus1 = Function(W)
(unPlus1,pnPlus1) = wnPlus1.split(True) #Velocity solution vector at time n+1
k_ = Function(K)
kn = Function(K) #TKE solution vector at time n
knPlus1 = Function(K) #TKE solution vector at time n+1
knMinus1 = Function(K)


#Define boundary conditions for the velocity equation
#Velocity
noslip_u_outer = Expression(("mint*omega*r*x[1]/r", "mint*omega*r*-1*x[0]/r","0.0"), mint = 0.0,degree=4, r = outer_bot_radius,omega = omega_outer)
noslip_u_inner = Expression(("mint*omega*r*x[1]/r", "mint*omega*r*-1*x[0]/r","0.0"), mint = 0.0,degree=4, r = inner_bot_radius,omega = omega_inner)

#Pressure
originpoint = OriginPoint()
bc_inner= DirichletBC(W.sub(0),noslip_u_inner,sub_domains,1) #boundary condition for inner cylinder
bc_outer = DirichletBC(W.sub(0),noslip_u_outer,sub_domains,2) #boundary condition for outer cylinder
bcp = DirichletBC(W.sub(1), 0.0, originpoint, 'pointwise') #specify a point on the boundary for the pressure
bcs_u = [bc_outer,bc_inner,bcp]


#Define boundary conditions for the k equation
bc_inner_k = DirichletBC(K,0.0,sub_domains,1) #boundary condition for k on the inner cylinder
bc_outer_k = DirichletBC(K,0.0,sub_domains,2) #boundary condition for k on the outer cylinder
bcs_k = [bc_inner_k,bc_outer_k]



#Assign initial conditions for u equation (Start from rest)
unMinus1.assign(Constant((0.0,0.0,0.0)))
un.assign(Constant((0.0,0.0,0.0)))



#Weak Formulations
def a(u,v):
    return inner(nabla_grad(u),nabla_grad(v))
def a_sym(u,v):
    return inner(.5*(nabla_grad(u)+nabla_grad(u).T),nabla_grad(v))
def b(u,v,w):
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w)
def c(p,v):
	return inner(p,div(v))

#Wall normal distance
d = Expression('-abs(pow(x[0]*x[0] + x[1]*x[1],.5)-(r_out+r_in)/2)+(r_out-r_in)/2',r_in = inner_top_radius, r_out = outer_top_radius,  degree=2) 

#u euqation for NSE:
u_lhs0 = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx #Use this before we start the k equation
u_rhs = (1./dt)*(inner(un,v))*dx


Au = None
Ak = None
Bu = None
Bk = None

#Output to save
edr = np.zeros(t_num) # energy dissipation rate at time t
ke = np.zeros(t_num) # ke at t
ks = np.zeros(t_num) # k at t



for jj in range(0,t_num):
    t = t + dt
    print('Numerical Time Level: t = '+ str(t))
    if jj <= k_on: #Before cylinder is up to speed, only NSE
        nu_t = 0 
        #Matrix Assembly for the u equation
        Au = assemble(u_lhs0)
        Bu = assemble(u_rhs)
        mint_val = smooth_bridge(t)
        noslip_u_outer.mint = mint_val
        noslip_u_inner.mint = mint_val
        #Application of boundary conditions for the u equation
        [bc.apply(Au,Bu) for bc in bcs_u]
        #Solve
        solver.solve(Au,w_.vector(),Bu)
        #Solution of the u equation
        (unPlus1,pnPlus1) = w_.split(True)
        #Apply time filter to correct dissipation in BE
        unPlus1.vector()[:] = unPlus1.vector()[:] -(1./3.)*(unPlus1.vector()[:]-2*un.vector()[:]+unMinus1.vector()[:])
        #Assign values for next time step
        unMinus1.assign(un)
        un.assign(unPlus1)

        if jj == k_on: #Initialize k equation 
            I = .16*(pow(Re,-(1/8))) 
            kn.assign(1.5*I*I*project(inner(unPlus1,unPlus1)))

    else: #NSE + k equation
        if l_choice == 0: #no k equation, continue with NSE
            u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+eps*p*q*dx
            u_rhs = (1./dt)*(inner(un,v))*dx

        if l_choice ==2: 
            sqrtk = sqrt(.5*(kn+abs(kn))) #square root of positive part of k
            d_func = .41*d*sqrt(d/L)
            l_func = .5*(d_func+sqrtk*sqrt(2)*tau-abs(d_func-sqrt(2)*sqrtk*tau)) # min approximated with (x+y-|x-y|)/2
            nu_t = mu*sqrtk*l_func

            u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ nu_t*a_sym(u,v)*dx #
            u_rhs = (1./dt)*(inner(un,v))*dx

        if l_choice == 4:
            sqrtk = sqrt(.5*(kn+abs(kn))) #square root of positive part of k
            l_func = .41*d
            nu_t = mu*sqrtk*l_func

            u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ nu_t*a_sym(u,v)*dx #
            u_rhs = (1./dt)*(inner(un,v))*dx
            
        #Matrix Assembly for the u equation
        Au = assemble(u_lhs)
        Bu = assemble(u_rhs)
        mint_val = smooth_bridge(t)
        noslip_u_outer.mint = mint_val
        noslip_u_inner.mint = mint_val
        #Application of boundary conditions for the u equation
        [bc.apply(Au,Bu) for bc in bcs_u]
        #Solve
        solver.solve(Au,w_.vector(),Bu)
        #Solution of the u equation
        (unPlus1,pnPlus1) = w_.split(True)
        #Apply time filter to correct dissipation in BE
        unPlus1.vector()[:] = unPlus1.vector()[:] -(1./3.)*(unPlus1.vector()[:]-2*un.vector()[:]+unMinus1.vector()[:])
        #Assign values for next time step
        unMinus1.assign(un)
        un.assign(unPlus1)


        #Matrix Assembly for the k equation
        u_Sym = a_sym(unPlus1,unPlus1)

        if l_choice == 0:

            k_lhs = (1./dt)*inner(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx

        if l_choice ==2: 
            coef_l = .5*(1/(sqrt(2)*tau)+sqrtk/(.41*d*sqrt(d/L))+abs( 1/(sqrt(2)*tau)-sqrtk/(.41*d*sqrt(d/L)) ))

            k_lhs = (1./dt)*inner(k,phi)*dx + (nu+ nu_t)*a(k,phi)*dx + b(unPlus1,k,phi)*dx + coef_l*inner(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx + nu_t*inner(u_Sym,phi)*dx


        if l_choice == 4: 
            k_lhs = (1./dt)*inner(k,phi)*dx + b(unPlus1,k,phi)*dx + (sqrtk/l_func)*inner(k,phi)*dx+ (nu_t)*a(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx  + nu_t*inner(u_Sym,phi)*dx


        Ak = assemble(k_lhs)
        Bk = assemble(k_rhs)
        #Application of boundary conditions for the k equation
        [bc.apply(Ak,Bk) for bc in bcs_k]
        #Solve
        solve(Ak,k_.vector(),Bk)
        kn.assign(k_)



    #Saving relevant terms
    edr[jj] = assemble((nu+nu_t)*a(unPlus1,unPlus1)*dx) 
    ke[jj] = assemble(inner(unPlus1,unPlus1)*dx)
    ks[jj] = assemble(kn*dx)


    if(jj%frameRate == 0):
        velocity_paraview_file << (unPlus1,t)




np.savetxt('eps.txt',edr)
np.savetxt('ke.txt',ke)
np.savetxt('ks.txt',ks)

