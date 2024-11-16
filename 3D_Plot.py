# Description: Visualize a lattice in 3D.

# ---------- Imports ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Function Definitions ----------

def dictionaryOrder(theta,phi): # Order particles by increasing theta
    sorted = np.argsort(theta)
    phi = phi[sorted]
    theta = theta[sorted]
    return theta,phi

def read_state(N,r,lattice,rInit=10e-10,NMC=0): # Read state for simulation
    # If initialized lattice with N particles, pass N-1 to this function to 
    # remove the last particle. rInit is the radius used to initialize thie lattice.
    particles = np.zeros([N,3]) # Array to grab particles
    lattice += f'N_{N}_lattice_{NMC}.txt' # Get name of text file holding lattice
    data = pd.read_csv(lattice,delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
    x,y,z = data[:,0],data[:,1],data[:,2]
    θ = np.arccos(z/rInit) # Compute angle of particles
    φ = np.arctan2(y,x)
    θ,φ = dictionaryOrder(θ,φ)
    particles[:,0] = r*np.sin(θ)*np.cos(φ) # Save x,y,z
    particles[:,1] = r*np.sin(θ)*np.sin(φ)
    particles[:,2] = r*np.cos(θ)
    return particles

def computeSphere(r): # Compute spherical surface
    v = np.linspace(0,np.pi,500)
    u = np.linspace(0,2*np.pi,500)
    
    xs = r*np.outer(np.cos(u),np.sin(v))
    ys = r*np.outer(np.sin(u),np.sin(v))
    zs = r*np.outer(np.ones(np.size(u)),np.cos(v))

    return xs,ys,zs

def makePosPlots3D(particles,N,r): # Make plots of particle position in 3D
    xs,ys,zs = computeSphere(r)
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig = plt.figure(figsize=(12,10)) # Set plot size
    ax = fig.add_subplot(projection='3d') # Create plotting axes
    ax.plot_wireframe(xs,ys,zs,rstride=5,cstride=5,alpha=0.1) # Show sphere
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.scatter(particles[:,0],particles[:,1],particles[:,2],c=colors)
    plt.show()

# ---------- Main ----------

data_dir = 'z_Data_1' # Directory storing data for current run (contains Lattices/)
lattice = f'{data_dir}/Lattices/' # Dir storing lattice data

N = 500
NMC = 8
R = 25.5e-10

particles = read_state(N,R,lattice,R,NMC)
makePosPlots3D(particles,N,R)