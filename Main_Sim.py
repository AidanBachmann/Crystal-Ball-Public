# Description: Main function for performing simulations.

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

Nt = int(5000) # Number of time steps for simulation to perform for shortest simulation
dt = 1e-17 # Update time step
max_dt = 1e-16 # Maximum time step size allowed for adaptive time stepping
NN = int(6) # Number of nearest neighbors
NMC = int(8) # Number of Monte-Carlo simulations to run. Use 1 for single run. For a computer with 10 CPUs, use at most NMC = 8 for best performance.
pIdx = int(1) # Index of particle to remove. Must be smaller than N - 1 for the smallest N value used.
data_dir = 'z_Data_1' # Directory storing data for current run (contains Lattices/)
lattice = f'{data_dir}/Lattices/' # Dir storing lattice data

Narr = funcs.np.linspace(100,1000,10,dtype='int') # Array of N values to generate lattices for
Nrm_arr = funcs.np.linspace(1,10,10,dtype='int') # Array of Nrm values (number of particles to remove)

save = True # Save plots
generatePlots = False # If true, generate Mercator plots from time series
runSingle = False # Run a single simulation for a particle seed (certain value of NMC)
debug = False # Turn on debugging. At the moment, this just turns on plotting of initial condition.
log = False # Plot on log scale

# ---------- Simulation Loop ----------

if runSingle and (len(Narr) == int(1)) and (len(Nrm_arr) == int(1)):
    print('Running single simulation.\n')
    funcs.simSingle(Narr[0],Nt,Nrm_arr[0],k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir,save,log,generatePlots,debug) # Run single test
else:
    print(f'Running {NMC} Monte-Carlo simulations.\n')
    if NMC == int(1): # If NMC = 1, run regularly. No reason to create a process.
        for Nrm_i in Nrm_arr: # Loop through different params
            i = Nrm_i - 1
            Nt_i = Nt + i*1000
            print(f'\nUsing {Nt_i} time steps for Nrm = {Nrm_i}.\n')
            for N_i in funcs.np.linspace(100,1000,10,dtype='int'):
                funcs.simSingle(N_i,Nt_i,Nrm_i,k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir,save,log,generatePlots,debug) # Run single test
            print(f'\n\nFinished Simulations for Nrm = {Nrm_i}\n\n')
    elif NMC > 1:
        for Nrm_i in Nrm_arr: # Loop through different params
            i = Nrm_i - 1
            Nt_i = Nt + i*1000
            print(f'\nUsing {Nt_i} time steps for Nrm = {Nrm_i}.\n')
            for N_i in funcs.np.linspace(100,1000,10,dtype='int'):
                funcs.simMC_Multi(N_i,Nt_i,Nrm_i,k,r,m,dt,C,max_dt,i,lattice,NMC,data_dir,save,log,generatePlots,debug)
            print(f'\n\nFinished Simulations for Nrm = {Nrm_i}\n\n')

# Analyze results using Scaling.py.