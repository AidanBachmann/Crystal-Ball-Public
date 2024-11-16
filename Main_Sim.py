FAST = False # Flag to use fast version of the code

if FAST:
    import Functions_Sim_FAST as funcs
    print('\nUsing FAST code.\n\n')
else:
    import Functions_Sim_SLOW as funcs
    print('\nUsing SLOW code.\n\n')

# ---------- Constants ----------
# SI Units

r = 25.5e-10 # Radius of sphere
C = 2*funcs.np.pi*r # Cirfumference of sphere
ep = 8.85418782e-12 # Permitivitty of free space
m = 11*1.67262192e-27 # Mass of the Boron-11 (11 * m_proton)
q = 1.602e-19 # Elementary charge
c = 2.99792458e8 # Speed of light
Z = 5 # Charge of molecule
k = (1/(4*funcs.np.pi*ep))*pow(Z*q,2) # Coulomb force constant, includes charge

# ---------- Simulation Parameters ----------

N = 500 # Number of particles in pre-initialized lattice, 500
NN = 6 # Number of nearest neighbors
Nrm = 7 # Number of particles to move in 'chunk' mode (use 1 for single)
NMC = 1 # Number of Monte-Carlo simulations to run. Use 0 for single run, 10
data_dir = 'z_Data_5' # Directory storing data for current run (contains Lattices/)
lattice = f'{data_dir}/Lattices/' # Dir storing lattice data
pIdx = int(350) # Index of particle to remove

Nt = 40000 # Number of time steps for simulation to perform, 40000
dt = 1e-17 # Update time step, 2.5e-17
max_dt = 1e-16 # Maximum time step size allowed for adaptive time stepping

save = True # Save plots
log = False # Plot on log scale

# --- Formatting ---

# particles = np.zeros([N,1,3]) --> Array to store particle ICs before removing particles from lattice
# simParticles = np.zeros([N-Nrm,Nt+1,10]) --> Array for storing simulation IC after removing particles

'''pIdx_i = 0
for i in funcs.np.linspace(1,10,10,dtype='int'): # Loop through different params
    Nrm_i = i
    Nt_i = Nt + (i-1)*1000
    print(f'\nUsing {Nt_i} time steps for Nrm = {Nrm_i}.\n')
    for N_i in funcs.np.linspace(100,1000,10,dtype='int'):
        funcs.simMC_Multi(N_i,Nt_i,Nrm_i,k,r,m,dt,C,max_dt,pIdx_i,lattice,NMC,data_dir)
    pIdx_i += 1
    print(f'\n\nFinished Simulations for Nrm = {Nrm_i}\n\n')'''

funcs.simSingle(N,Nt,Nrm,k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir,save,log) # Run single test
#funcs.simMC_Multi(N,Nt,Nrm,k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir)