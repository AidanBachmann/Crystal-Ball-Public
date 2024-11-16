# Description: Main function for initializing lattices.

FAST = False # Flag to use fast version of the code

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

seed = 314159265 # Set initial seed for random number generator
Nt = int(6e4) # Number of time steps
dt = 1e-17 # Initial time step
max_dt = 5e-15 # Max time step for adaptive time stepping
n = int(0.95*Nt) # This defines beginning of "late time" for averaging to assess convergence
NN = int(6) # Number of nearest neighbors to find
NMC = int(8) # Number of Monte-Carlo simulations to run. Use 1 for single run. For NMC > 1, this starts NMC parallel runs with unique seeds
nu = 5e9 # Drag coefficient
particleIdx = 0 # Particle index for plotting single particle velocity time series
data_dir = 'z_Data_1' # Directory to write data to

Narr = funcs.np.linspace(100,1000,10,dtype='int') # Array of N values for generating lattices

save = True # Save plots
log = False # Plot on log scale
write = True # Flag to write computed lattice to text file
runSingle = False # Run a single simulation for a particle seed (certain value of NMC)

# ---------- Simulation Loop ----------

if runSingle and (len(Narr) == int(1)):
    print('Initializing single lattice.\n')
    funcs.initSingle(Narr[0],Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log)
else:
    print(f'Initializing {NMC} lattices with unique seeds.\n')
    if NMC == int(1):
        for N_i in Narr:
            funcs.initSingle(N_i,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log)
    elif NMC > 1:
        for N_i in Narr:
            funcs.initMC_Multi(N_i,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log)
            seed += NMC+1 # Change seed for next set

funcs.plotLatticeConsts(data_dir,NMC,True) # Plot average lattice constants for all lattices

# Characterize generated lattices further using Voronoi.py and Thomson.py. These can be used to compare results with other
# numerical solutions to the Thomson problem. 