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


# solver = KrylovSolver('bicgstab','jacobi')
# solver.parameters['report'] = True
# solver.parameters['monitor_convergence'] = False
# solver.parameters['relative_tolerance'] = 1.0e-7
# #solver.parameters['absolute_tolerance'] = 1.0e-9
#

#Solvercd
MAX_ITERS = 10000
solver = PETScKrylovSolver('gmres','none')
solver.ksp().setGMRESRestart(MAX_ITERS)
solver.parameters["monitor_convergence"] = False
solver.parameters['relative_tolerance'] = 1e-6
solver.parameters['maximum_iterations'] = MAX_ITERS


#Pi to high accuracy
Pi = np.pi

#DOF/timestep size
N =45
dt = .01
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
omega_inner = 4  #Michal McLaughlin suggested increasing Re
omega_outer =0
omega_diff = abs(omega_inner-omega_outer)

#Time it takes to reach full speed of cylinders
x = 5
#Turn k equation on when cylinder is at full speed
k_on = int(x/dt)

#Initial, final time, timestep
t_init = 0.0
t_final = 50
t_num = int((t_final - t_init)/dt)
t = t_init

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

#Construct Mesh
meshrefine = 1 #f 0,1, or 2: how many times boundary is refined
refine1 = .03 #distance from wall refined first time
refine2 = .03 #distance from wall refined second time



#Generate mesh for given cylinders
mesh = generate_mesh ( domain, N )

if (meshrefine > .5): #mesh refine greater than zero: refine once
    sub_domains_bool = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
    sub_domains_bool.set_all(False)

    class SmallInterest(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1]**2 + x[0]**2 < (inner_top_radius+refine1)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine1)**2)


    interest = SmallInterest()
    interest.mark(sub_domains_bool,True)
    mesh = refine(mesh,sub_domains_bool)
    print("mesh refined")

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
        print("mesh refined again")


hmax = mesh.hmax()
hmin = mesh.hmin()

#Reynolds/Taylor number information
height = t_iz - b_iz #hieght of the cylinder
radius_ratio = inner_bot_radius/outer_bot_radius #eta
L =  outer_bot_radius - inner_bot_radius
aspect_ratio =  height/L
domain_volume = assemble(1*dx(mesh))
print(domain_volume)
Re = abs(L*(omega_outer-omega_inner)/nu) #Reynold's number as defined in Bilson/Bremhorst
Ta =  Re*Re*4*(1-radius_ratio)/(1+radius_ratio)



#Print statements to ensure running correct version
if l_choice ==1 or l_choice == 2 or l_choice == 3:
    keyword = "N_" + str(N) + "_omega_" + str(omega_diff) + "_nu_"+str(nu)+ "_l_choice_"+str(l_choice)+"_tau_"+str(tau)

if l_choice ==0 or l_choice == 4:
    keyword = "N_" + str(N) + "_omega_" + str(omega_diff) + "_nu_"+str(nu)+  "_l_choice_"+str(l_choice)


#Print statements to see what is running
print("N equals " + str(N))
print("dt equals " + str(dt))
print("nu equals " + str(nu))
print("hmax equals " + str(hmax))
print("hmin equals " + str(hmin))
print("l choice is " + str(l_choice))
print("omega (outer) equals " + str(omega_outer))
print("omega (inner) equals " + str(omega_inner))

#Print the total number of degrees of freedom
X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)
vdof = X_test.dim()
pdof = Q_test.dim()
print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))

print("The Reynolds number is " + str(Re))
print("The Taylor number is " + str(Ta))


#Paraview Setup
savespersec = 20. #How many snapshots are taken per second
velocity_paraview_file = File(keyword+"/vfolder/3d_TC_"+str(N)+"_"+str(dt)+"_"+str(omega_diff)+".pvd")
snapspersec = int(1./dt) #timesteps per second
frameRate = int(snapspersec/savespersec) # how often snapshots are saved to paraview
if frameRate ==0:
    frameRate =1

savefile =5. #Save output every 5 seconds
saveRate = int(savefile/dt+.00001)

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



#Assign initial conditions (Start from rest)
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
d = Expression('-abs(pow(x[0]*x[0] + x[1]*x[1],.5)-(r_out+r_in)/2)+(r_out-r_in)/2',r_in = inner_top_radius, r_out = outer_top_radius,  degree=2) #wall distance formula

#u euqation/kequation options:
u_lhs0 = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx #Use this before we start the k equation
u_rhs = (1./dt)*(inner(un,v))*dx
# if l_choice == 0: #no k equation
#     u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+eps*p*q*dx
#     u_rhs = (1./dt)*(inner(un,v))*dx
#     #These aren't actually used but I haven't taken them out yet
#     k_lhs = (1./dt)*inner(k,phi)*dx
#     k_rhs = (1./dt)*inner(kn,phi)*dx
# if l_choice == 1: #'standard' k equation: l = sqrt(2k)tau, small penalty term
#
#     u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ np.sqrt(2.)*mu*tau*kn*a_sym(u,v)*dx+eps*p*q*dx
#     u_rhs = (1./dt)*(inner(un,v))*dx
#
#     k_lhs = (1./dt)*inner(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx+ (nu+np.sqrt(2.)*mu*kn*tau)*a(k,phi)*dx
#     k_rhs = (1./dt)*inner(kn,phi)*dx  + np.sqrt(2.0)*mu*tau*kn*a_sym(unPlus1,unPlus1)*phi*dx #Last term here is the square of the symmetric gradient write function for this later
#
#
# #This still needs debugging (and is wildly slow) but we need lchoice 1 working first
# if l_choice ==2: # min of two options, small penalty
#     u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ mu*(.5*(.41*d*sqrt(d/L) + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau )-.5*sqrt((eps + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau - .41*d*sqrt(d/L))**2))*sqrt(.5*(abs(kn)+kn))*a_sym(u,v)*dx #
#     u_rhs = (1./dt)*(inner(un,v))*dx
#
#     k_lhs = (1./dt)*inner(k,phi)*dx + (nu+ mu*(.5*(.41*d*sqrt(d/L) + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau )-.5*sqrt((eps + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau - .41*d*sqrt(d/L))**2))*sqrt(.5*(abs(kn)+kn)))*a(k,phi)*dx + b(unPlus1,k,phi)*dx + .5*(1./(sqrt(2)*tau) + sqrt(L/d)*sqrt(.5*(abs(kn)+kn))/(.41*d)+ sqrt((1./(sqrt(2)*tau) - sqrt(.5*(abs(kn)+kn))*sqrt(L/d)/(.41*d))**2)      )*inner(k,phi)*dx
#     k_rhs = (1./dt)*inner(kn,phi)*dx +  mu*(.5*(.41*d*sqrt(d/L) + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau )-.5*sqrt((eps + sqrt(2.)*sqrt(.5*(abs(kn)+kn))*tau - .41*d*sqrt(d/L))**2))*sqrt(.5*(abs(kn)+kn))*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*phi*dx #Last term here is the square of the symmetric gradient write function for this later
#



Au = None
Ak = None
Bu = None
Bk = None

#Output to save
ts = np.zeros(t_num)# time array
edr = np.zeros(t_num) # energy dissipation rate at time t
ke = np.zeros(t_num) # ke at t
ks = np.zeros(t_num) # k at t
nu_bar = np.zeros(t_num)# nu total at t


##THINGS FOR CHECKING THE LAMINAR SOLUTION
###Location variables, probably more than we need if we're honest but I'm not taking chances
thetaPoints = 500 #number of points we take along varying theta (points along donut shaped cross sections)
theta = np.zeros((thetaPoints))
xy = np.zeros((2,thetaPoints))
zpoints = 500 #number of points we take along z axis
avg = np.zeros((zpoints)) #average at one timestep, we save the last timestep (as god willing we've got some sort of fully evolved flow)
avgavg = np.zeros((zpoints)) #average over donut, over time, store here
z = np.zeros((zpoints))
averageCount = 0#keep track of how many things we've added to average

for i in range(0,thetaPoints):
    theta[i] = i*(2*Pi)/thetaPoints
    xy[0,i] = .5*(outer_top_radius+inner_top_radius)*cos(theta[i])
    xy[1,i] = .5*(outer_top_radius+inner_top_radius)*sin(theta[i])
for i in range(0,zpoints):
    z[i] = b_iz+t_iz*i/zpoints



#If I knew how to do this any other way I would
#Makes a file that tells me what I did so I can figure it out later
picsavefile =File(keyword+"/picfolder/test.pvd")
arrsavefile =File(keyword+"/arrfolder/test.pvd")

text_file = open(keyword+"/arrfolder/Information of this Run.txt", "w")

text_file.write("N equals " + str(N)+"\n")
text_file.write("dt equals " + str(dt)+"\n")
text_file.write("nu equals " + str(nu)+"\n")
text_file.write("hmax equals " + str(hmax)+"\n")
text_file.write("hmin equals " + str(hmin)+"\n")
text_file.write("omega (outer) equals " + str(omega_outer)+"\n")
text_file.write("omega (inner) equals " + str(omega_inner)+"\n")

text_file.write("r (outer) equals " + str(outer_bot_radius)+"\n")
text_file.write("r (inner) equals " + str(inner_bot_radius)+"\n")

text_file.write("The number of velocity DOFs is:" + str(vdof)+"\n")
text_file.write("The number of pressure DOFs is:" + str(pdof)+"\n")
text_file.write("The mesh has been refined " + str(meshrefine)+" times\n")

text_file.write("The Reynolds number is " + str(Re)+"\n")
text_file.write("The Taylor number is " + str(Ta)+"\n")

U = abs(omega_outer*outer_top_radius-omega_inner*inner_top_radius)

text_file.write("U^3/l= " + str((U*U*U/L))+"\n")


text_file.close()



count =1 #for test ramping up tau
for jj in range(0,t_num):
    t = t + dt
    print('Numerical Time Level: t = '+ str(t))
    if jj <= k_on: #Before cylinder is up to speed, NSE
        nu_t = 0 #Need this for eps in definiton of energy dissipation rate
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

        if jj == k_on: #Initialize k equation when up to speed
            I = .16*(pow(Re,-(1/8))) #Missing a factor of .16 as we were testing a bigger IC
            kn.assign(1.5*I*I*project(inner(unPlus1,unPlus1)))
            # knMinus1.assign(kn) #For the time filter?

    else: #NSE + k equation

        if l_choice == 0: #no k equation

            u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+eps*p*q*dx
            u_rhs = (1./dt)*(inner(un,v))*dx

        if l_choice == 1: #'standard' k equation: l = sqrt(2k)tau, small penalty term
            sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            l_func = sqrt(2)*sqrtk*tau
            nu_t = mu*sqrtk*l_func

            u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ nu_t*a_sym(u,v)*dx #
            u_rhs = (1./dt)*(inner(un,v))*dx


        if l_choice ==2: # min of two options, small penalty
            sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            d_func = .41*d*sqrt(d/L)
            l_func = .5*(d_func+sqrtk*sqrt(2)*tau-abs(d_func-sqrt(2)*sqrtk*tau))
            nu_t = mu*sqrtk*l_func

            u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ nu_t*a_sym(u,v)*dx #
            u_rhs = (1./dt)*(inner(un,v))*dx
        if l_choice == 3: #'standard' k equation: l = sqrt(2k)tau, small penalty term
            sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            l_func = sqrt(2)*sqrtk*tau
            nu_t = mu*sqrtk*l_func

            u_lhs = (1./dt)*inner(u,v)*dx +eps*p*q*dx+ b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx+ nu_t*a_sym(u,v)*dx #
            u_rhs = (1./dt)*(inner(un,v))*dx
        if l_choice == 4: #'standard' k equation: l = sqrt(2k)tau, small penalty term
            sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
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

        if l_choice == 0: #'standard' k equation: l = sqrt(2k)tau
            # sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            # l_func = sqrt(2)*sqrtk*tau
            # nu_t = mu*sqrtk*l_func

            k_lhs = (1./dt)*inner(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx


        if l_choice == 1: #'standard' k equation: l = sqrt(2k)tau
            # sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            # l_func = sqrt(2)*sqrtk*tau
            # nu_t = mu*sqrtk*l_func

            k_lhs = (1./dt)*inner(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx+ (nu+nu_t)*a(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx  + nu_t*inner(u_Sym,phi)*dx

        if l_choice ==2: # min of two options
            # sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            # d_func = .41*d*sqrt(d/L)
            # l_func = .5*(d_func+sqrtk*sqrt(2)*tau-abs(d_func-sqrt(2)*sqrtk*tau))
            # nu_t = mu*sqrtk*l_func

            coef_l = .5*(1/(sqrt(2)*tau)+sqrtk/(.41*d*sqrt(d/L))+abs( 1/(sqrt(2)*tau)-sqrtk/(.41*d*sqrt(d/L)) ))

            k_lhs = (1./dt)*inner(k,phi)*dx + (nu+ nu_t)*a(k,phi)*dx + b(unPlus1,k,phi)*dx + coef_l*inner(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx + nu_t*inner(u_Sym,phi)*dx

        if l_choice == 3: #'standard' k equation: l = sqrt(2k)tau NO VISC
            # sqrtk = sqrt(.5*(kn+abs(kn))) #1/2(k+abs(k)*2)
            # l_func = sqrt(2)*sqrtk*tau
            # nu_t = mu*sqrtk*l_func

            k_lhs = (1./dt)*inner(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx+ (nu_t)*a(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx  + nu_t*inner(u_Sym,phi)*dx

        if l_choice == 4: #l = .41 d

            k_lhs = (1./dt)*inner(k,phi)*dx + b(unPlus1,k,phi)*dx + (sqrtk/l_func)*inner(k,phi)*dx+ (nu_t)*a(k,phi)*dx
            k_rhs = (1./dt)*inner(kn,phi)*dx  + nu_t*inner(u_Sym,phi)*dx




        Ak = assemble(k_lhs)
        Bk = assemble(k_rhs)
        #Application of boundary conditions for the k equation
        [bc.apply(Ak,Bk) for bc in bcs_k]
        #Solve
        solve(Ak,k_.vector(),Bk)

        if(TFK == 1): #I have no clue why these aren't all the same size
            k_.vector()[:] = k_.vector()[:] -(1./3.)*(k_.vector()[:]-2*kn.vector()[:]+knMinus1.vector()[:])

        knMinus1.assign(kn)
        kn.assign(k_)



    #Saving relevant terms
    edr[jj] = assemble((nu+nu_t)*a(unPlus1,unPlus1)*dx) #FIX THIS LATER (Needs nu_t as well)
    ts[jj] = t
    ke[jj] = assemble(inner(unPlus1,unPlus1)*dx)
    ks[jj] = assemble(kn*dx)
    nu_bar[jj] = assemble((2*nu+nu_t)*dx(mesh))
    if ks[jj]>5*ks[jj-1]:
        print('yikes')
        plt.figure(4)
        plt.plot(ts,ks,"r", label=r"k over time",linewidth =.5 )
        plt.xlabel("t")
        plt.ylabel("k")

        plt.savefig(keyword+"/picfolder/k over time")
        plt.close()


    if(jj%frameRate == 0):
        velocity_paraview_file << (unPlus1,t)
        print(ks[jj])
        print(assemble(nu_t*dx(mesh))/domain_volume)
        print(assemble(nu*dx(mesh))/domain_volume)

    #####Every 5 seconds save where we're at so I don't completely lose everything if this crashes
    if (jj % saveRate == 0):

        filename_init_v = keyword+'/arrfolder/velocity_init1.txt'
        filename_init_p =  keyword+'/arrfolder/pressure_init1.txt'
        u_init_hold = unPlus1.vector().get_local()
        p_init_hold = pnPlus1.vector().get_local()
        np.savetxt(filename_init_v,u_init_hold)
        np.savetxt(filename_init_p,p_init_hold)
        np.savetxt(keyword+'/arrfolder/time.txt"',ts)
        np.savetxt(keyword+ '/arrfolder/eps.txt',edr)
        np.savetxt(keyword+ '/arrfolder/ke.txt',ke)
        np.savetxt(keyword+ '/arrfolder/ks.txt',ks)
        np.savetxt(keyword+ '/arrfolder/nu_bar.txt',nu_bar)

        plt.figure(1)
        plt.plot(ts,edr,"r", label=r"energy dissipation over time",linewidth =.5 )
        plt.xlabel("t")
        plt.ylabel("eps")

        plt.savefig(keyword+"/picfolder/eps over time")
        plt.close()

        plt.figure(2)
        plt.plot(z,avg,"r", label=r"Average v at end",linewidth =.5 )
        plt.xlabel("z")
        plt.ylabel("avg v")

        plt.savefig(keyword+"/picfolder/Avg_v_end")
        plt.close()


        plt.figure(3)
        plt.plot(ts,ke,"r", label=r"KE over time",linewidth =.5 )
        plt.xlabel("t")
        plt.ylabel("ke")

        plt.savefig(keyword+"/picfolder/KE over time")
        plt.close()


        plt.figure(4)
        plt.plot(ts,ks,"r", label=r"k over time",linewidth =.5 )
        plt.xlabel("t")
        plt.ylabel("k")

        plt.savefig(keyword+"/picfolder/k over time")
        plt.close()

    if (jj % saveRate == 1): #Second initial condition so we can do time filter easily
        filename_init_v = keyword+'/arrfolder/velocity_init2.txt'
        filename_init_p =  keyword+'/arrfolder/pressure_init2.txt'
        u_init_hold = unPlus1.vector().get_local()
        p_init_hold = pnPlus1.vector().get_local()
        np.savetxt(filename_init_v,u_init_hold)
        np.savetxt(filename_init_p,p_init_hold)

    #### This tests vs analytic solution
    #     thetaPoints = 2 #number of points we take along varying theta
    #     zpoints = 2 #number of points we take along z axis
    #     rpoints = 100 #points going out radially
    #
    #     r = np.zeros((rpoints))
    #     theta = np.zeros((thetaPoints))
    #     z = np.zeros((zpoints))
    #
    #
    #     for i in range(0,thetaPoints):
    #        theta[i] = i*(2*Pi)/thetaPoints
    #     for i in range(0,zpoints):
    #        z[i] = b_iz+t_iz*(i+1)/(zpoints+1)
    #     for i in range(0,rpoints):
    #        r[i] = inner_top_radius+(outer_top_radius-inner_top_radius)*(i+1)/(rpoints+2)
    #     for i in range(0,thetaPoints):
    #         for j in range(0,zpoints):
    #            u_r_vals = np.zeros(rpoints)
    #            u_th_vals = np.zeros(rpoints)
    #            u_z_vals = np.zeros(rpoints)
    #            u_x_vals = np.zeros(rpoints)
    #            u_y_vals = np.zeros(rpoints)
    #            p_vals = np.zeros(rpoints)
    #            for k in range(0,rpoints):
    #             # print("r "+str(r[k]))
    #             # print("theta "+str(cos(theta[i])))
    #             # print("x "+str(r[k]*cos(theta[i])))
    #             # print("y "+str(r[k]*sin(theta[i])))
    #             # print("z "+str(z[j]))
    #             uvw = unPlus1(r[k]*cos(theta[i]), r[k]*sin(theta[i]),z[j])
    #             u_x_vals[k] = uvw[0]
    #             u_y_vals[k] = uvw[1]
    #             u_r_vals[k] = uvw[0]*cos(theta[i])+uvw[1]*sin(theta[i])
    #             u_th_vals[k]= uvw[1]*cos(theta[i])-uvw[0]*sin(theta[i])
    #             u_z_vals[k]= uvw[2]
    #             p_vals[k] = pnPlus1(r[k]*cos(theta[i]), r[k]*sin(theta[i]),z[j])
    #
    #             plt.figure(1)
    #             plt.plot(r,u_r_vals,"b",r,u_th_vals,"k",r,u_z_vals,"r", label=r"velocity plots",linewidth =.5 )
    #             plt.xlabel("r")
    #             plt.ylabel("velocity_"+str(omega_diff))
    #
    #             plt.savefig("picfolder/N_" + str(N) +"_i_"+str(i)+"_j_"+str(j))
    #             plt.close()
    #
    #             np.savetxt("arrfolder/N_" + str(N) + "omega"+str(omega_diff)+"_i_"+str(i)+"_j_"+str(j)+"pvals.txt", p_vals)
    #             np.savetxt("arrfolder/N_" + str(N) + "omega"+str(omega_diff)+"_i_"+str(i)+"_j_"+str(j)+"Uth.txt", u_th_vals)
    #             np.savetxt("arrfolder/N_" + str(N) + "omega"+str(omega_diff)+"_i_"+str(i)+"_j_"+str(j)+"Ur.txt", u_r_vals)
    #             np.savetxt("arrfolder/N_" + str(N) + "omega"+str(omega_diff)+"_i_"+str(i)+"_j_"+str(j)+"Uz.txt", u_z_vals)
    #             #



np.savetxt(keyword+'/arrfolder/time.txt"',ts)
np.savetxt(keyword+ '/arrfolder/eps.txt',edr)
np.savetxt(keyword+ '/arrfolder/ke.txt',ke)
np.savetxt(keyword+ '/arrfolder/ks.txt',ks)
np.savetxt(keyword+ '/arrfolder/nu_bar.txt',nu_bar)

#Also test vs known information
# for j in range(0,zpoints):
#     currentavg = 0
#     for i in range(0,thetaPoints):
#         u1 = unPlus1(xy[0,i],xy[1,i] ,z[j])[0] #velocity in x
#         u2 = unPlus1(xy[0,i],xy[1,i] ,z[j])[1] #velocity in y
#         v = u1*cos(theta[i])+u2*sin(theta[i])
#         currentavg= currentavg + (1./thetaPoints)*v
#     avg[j] = currentavg


# np.savetxt(keyword+'/arrfolder/average_end_long.txt',avg)

plt.figure(1)
plt.plot(ts,edr,"r", label=r"energy dissipation over time",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("eps")

plt.savefig(keyword+"/picfolder/eps over time")
plt.close()

plt.figure(2)
plt.plot(z,avg,"r", label=r"Average v at end",linewidth =.5 )
plt.xlabel("z")
plt.ylabel("avg v")

plt.savefig(keyword+"/picfolder/Avg_v_end")
plt.close()


plt.figure(3)
plt.plot(ts,ke,"r", label=r"KE over time",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("ke")

plt.savefig(keyword+"/picfolder/KE over time")
plt.close()



plt.figure(4)
plt.plot(ts,ks,"r", label=r"k over time",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("k")

plt.savefig(keyword+"/picfolder/k over time")
plt.close()








#Print statements at end to reference what just ran
print("N equals " + str(N))
print("dt equals " + str(dt))
print("nu equals " + str(nu))
print("hmax equals " + str(hmax))
print("hmin equals " + str(hmin))
print("l choice is " + str(l_choice))
print("omega (outer) equals " + str(omega_outer))
print("omega (inner) equals " + str(omega_inner))

print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))

print("The Reynolds number is " + str(Re))
print("The Taylor number is " + str(Ta))
