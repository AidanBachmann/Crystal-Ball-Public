FAST = True # Flag to use fast version of the code

if FAST:
    import Functions_Init_FAST as funcs
    print('\nUsing FAST code.\n\n')
else:
    import Functions_Init_SLOW as funcs
    print('\nUsing SLOW code.\n\n')

# ---------- Constants ----------

# Constants (SI Units)
r = 25.5e-10 # Radius of sphere
C = 2*funcs.np.pi*r # Cirfumference of sphere
ep = 8.85418782e-12 # Permitivitty of free space
m = 1.67262192e-27 # Mass of the proton
q = 1.602e-19 # Elementary charge
c = 2.99792458e8 # Speed of light
Z = 1 # Charge of molecule
k = (1/(4*funcs.np.pi*ep))*pow(Z*q,2) # Coulomb force constant, includes charge

# ---------- Simulation Parameters ----------

seed = 314159265 # Set initial seed for pseudo-random number generator, 123456789
N = int(500) # Number of particles, 500
Nt = int(6e4) # Number of time steps, 60000
particleIdx = 0 # Particle index for plotting single particle velocity time series
NN = int(6) # Number of nearest neighbors to find
NMC = int(8) # Number of Monte-Carlo simulations to run. Use 1 for single run. Also number of processes for parallel running mode. Usually 8.
n = int(0.95*Nt) # This defines beginning of "late time"
nu = 5e9 # Drag coefficient, 5e9
dt = 1e-17 # Time step, 1e-17
max_dt = 5e-15 # Max time step for adaptive time stepping
tau = (Nt*max_dt)/7.5 # Time scale for damping parameter to drop
data_dir = 'z_Data_5' # Directory to write data to

# Convention for particles array
# particles = np.zeros([N,Nt+1,9]) # Particle, time step, (x,y,z,vx,vy,vz,ay,az)

save = True # Save plots
log = False # Plot on log scale
write = False # Flag to write computed lattice to text file

#Narr = funcs.np.array([362,432,482,492,522,572,612,632,642,672,732,752,762,792,812],dtype='int') # Thomson problem lattices
#Narr = funcs.np.linspace(100,1000,10,dtype='int') # LCF lattices

#for i in Narr:
#    funcs.initMC_Multi(i,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log,tau)
#    seed += NMC+1 # Change seed for next set

#funcs.initMC_Multi(N,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log,tau)
#funcs.initSingle(N,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log,tau)
funcs.initSingle(N,Nt,NN,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log,tau)

funcs.plotLatticeConsts(data_dir,NMC,True)

#funcs.cleanAll()