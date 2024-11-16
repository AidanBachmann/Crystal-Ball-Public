# Description: Compute potential energy and average lattice constant for a lattice. Compare lattice constant scaling to theory.

# ---------- Imports ----------

import numpy as np
import pandas as pd
from scipy import spatial
from numba import jit
import matplotlib.pyplot as plt

# ---------- Plotting Options ----------
# Use LaTeX for rendering
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

# ---------- Function Definitions ----------

def alpha(N): # Interior angle of triangle
    return (N*np.pi)/(3*(N-2))

def theta(N): # Rotation angle between vertices
    return np.arccos(np.cos(alpha(N))/(1-np.cos(alpha(N))))

def a(N): # Area of individual triangle
    return 3*alpha(N) - np.pi

def A(N): # Total area of triangle
    return (2*N - 4)*a(N)

def findDist(p1,p2,r): # Find arc length distance between two particles
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    r1 = np.array([x1,y1,z1])
    r2 = np.array([x2,y2,z2])
    R = r1 - r2 # Separation vector
    Rmag = np.sqrt(R @ R) # Magnitude of separation vector
    return Rmag
    
def findParticleNN(particles,idx,N,r,NN=-1,n=-1): # Compute distance to N nearest neighbors for one particle
    # Note: NN is number of nearest neighbors to find, n is time step (default is final step)
    sArr = np.zeros([N-1])
    counter = 0
    for i in np.linspace(0,N-1,N,dtype='int'):
        if i != idx:
            sArr[counter] = findDist(particles[i,:],particles[idx,:],r)
            counter += 1
    sArr.sort()
    return sArr[0:NN]

def computeAllAvgDist(particles,N,r,NN=-1,n=-1):
    # Compute average distance to NN nearest neighbors for all particles
    sAvg = np.zeros([N])
    for i in np.linspace(0,N-1,N,dtype='int'):
        temp = findParticleNN(particles,i,N,r,NN,n)
        sAvg[i] = np.mean(temp)
    return sAvg

def scaling(N,R): # Scaling of lattice constant
    return R*np.sqrt( (2*(1-2*np.cos(alpha(N))))/(1-np.cos(alpha(N))) )

def estimateLatticeConst(dist): # Estimate lattice constant from collection of nearest neighbor distances
    return np.average(dist)

def read_state(N,lattice,NMC=0,r=1,rInit=25.5e-10,voronoi=True): # Read state for simulation
    # If initialized lattice with N particles, pass N-1 to this function to 
    # remove the last particle. rInit is the radius used to initialize the lattice.
    particles = np.zeros([N,3]) # Array to grab particles
    lattice += f'/Lattices/N_{N}_lattice_{NMC}.txt' # Get name of text file holding lattice
    data = pd.read_csv(lattice,delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
    for i in np.linspace(0,N-1,N).astype('int'):
        x,y,z = data[i,0],data[i,1],data[i,2]
        θ = np.arccos(z/rInit) # Compute angle of particles
        φ = np.arctan2(y,x)
        particles[i,0] = r*np.sin(θ)*np.cos(φ) # Save x,y,z
        particles[i,1] = r*np.sin(θ)*np.sin(φ)
        particles[i,2] = r*np.cos(θ)
    return particles

@jit(nopython = True)
def computePotential(particles,N): # Compute the potential energy of the final configuration
    U = 0 # Potential
    for i in np.linspace(0,N-1,N).astype('int'):
        r2 = np.array([particles[i,0],particles[i,1],particles[i,2]]) # Position of particle 2
        for j in np.linspace(0,N-1,N).astype('int'):
            if i != j:
                r1 = np.array([particles[j,0],particles[j,1],particles[j,2]]) # Position of particle 1
                U += (1/np.sqrt((r2-r1)@(r2-r1)))
    U *= 1/2 # Accounting for double count of particles
    return U

def compute_Un(N,NMC,data_dir,r=1):
    Un = 0
    aSim = 0
    for i in np.linspace(1,NMC,NMC,dtype='int'):
        particles = read_state(N,data_dir,NMC,r) # Read lattice
        aSim += estimateLatticeConst(computeAllAvgDist(particles,N,r,6,-1))
        Un += computePotential(particles,N) # Compute average energy for NMC lattices
    return Un/NMC,aSim/NMC,scaling(N,r)

def compute_all_Un(Narr,NMC,data_dir,r=1): # Compute all averages for NMC lattices with N particles
    Un = np.zeros([len(Narr)])
    a = np.zeros([len(Narr),2])
    for i in np.linspace(0,len(Narr)-1,len(Narr),dtype='int'):
        Un[i],a[i,0],a[i,1] = compute_Un(Narr[i],NMC,data_dir,r)
    err = (np.abs(a[:,1]-a[:,0])/a[:,1])*100
    return Un,a,err

def write_Un(Narr,Un,data_dir):
    f = open(f'{data_dir}/Potential Energy/U.txt','a')    
    for i in np.linspace(0,len(Narr)-1,len(Narr),dtype='int'):
        f.write(f'{Narr[i]},{Un[i]}\n')
    f.close()

def plot_Un(Narr,Un): # Plot energy
    plt.figure(figsize=(12,10))
    plt.scatter(Narr,Un)
    plt.xlabel('Number of Particles')
    plt.ylabel('Energy (Unitless)')
    plt.title('Energy of Lattice as a Function of N')
    plt.grid()
    plt.show()

def plot_a(Narr,a,err): # Plot energy
    fig,ax = plt.subplots(1,2,figsize=(20,10))

    a *= 1e10 # Scale lengths to angstroms

    ax[0].scatter(Narr,a[:,0],label='Simulation',s=50)
    ax[0].scatter(Narr,a[:,1],label='Theory',marker='+',c='r',s=100)
    ax[0].set_ylim(0,np.max(a[:,0])*(1+0.25))
    ax[0].set_xlabel('Number of Particles')
    ax[0].set_ylabel('Lattice Constant (Å)')
    ax[0].set_title('Lattice Constant as a Function of N')
    ax[0].legend()
    ax[0].grid()

    ax[1].scatter(Narr,err,s=75)
    ax[1].set_xlabel('Number of Particles')
    ax[1].set_ylabel('Percent Error')
    ax[1].set_title('Percent Error Between Theory and Simulation')
    ax[1].grid()

    plt.show()

# ---------- Main ----------

data_dir = 'z_Data_4' # Specify data directory
write = True # Toggle writing
Narr = np.linspace(100,1000,10,dtype='int') # Array of N values

NMC = 8 # Specify which Monte-Carlo simulation to grab
r = 25.5e-10 # Sphere radius to use for initialization

Un,a,err = compute_all_Un(Narr,NMC,data_dir,r) # Compute lattice constants and potential energy
plot_Un(Narr,Un) # Plot results
plot_a(Narr,a,err)

if write:
    write_Un(Narr,Un,data_dir)