# Description: Generate Voronoi construction for lattice.

# ---------- Imports ----------

import numpy as np
import pandas as pd
from scipy import spatial
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import jit
import matplotlib.pyplot as plt

# ---------- Function Definitions ----------

def parseWalesSoln(N,lattice): # Grab particle positions for Thomson minima by Wales et al
    particles = np.zeros([N,3]) # Array to grab particles
    lattice += f'/Wales Data/{N}.xyz' # Get name of text file holding lattice
    data = pd.read_csv(lattice,delim_whitespace=True,engine='python',header=None,skiprows=[0,1]).to_numpy() # Open CSVs
    particles[:,0] = data[:,1]
    particles[:,1] = data[:,2]
    particles[:,2] = data[:,3]
    return particles

def read_state(N,lattice,NMC=0,r=1,rInit=25.5e-10,voronoi=True): # Read state for a lattice initialized using this code
    # If initialized lattice with N particles, pass N-1 to this function to 
    # remove the last particle. rInit is the radius used to initialize thie lattice.
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

def makeVoronoi(particles,r,center,fig,subplot,title): # Generate Voronoi plots. Code adapted from scipy.spatial.SphericalVoronoi example.
    sv = spatial.SphericalVoronoi(particles,r,center)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    ax = fig.add_subplot(subplot, projection='3d')
    ax.set_axis_off()

    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = spatial.geometric_slerp(start, end, t_vals)
            ax.plot(result[..., 0],
                result[..., 1],
                result[..., 2],
                c='black')

    verts = np.empty([len(sv.regions)],dtype='object')
    color = np.empty([len(sv.regions)],dtype='str')
    counter = 0
    for region in sv.regions:
        polygon = [sv.vertices[i] for i in region]
        x,y,z = np.zeros([len(polygon)]),np.zeros([len(polygon)]),np.zeros([len(polygon)])
        for i in np.linspace(0,len(polygon)-1,len(polygon),dtype='int'):
            vertex = polygon[i]
            x[i] = vertex[0]
            y[i] = vertex[1]
            z[i] = vertex[2]
        verts[counter] = np.vstack((x,y,z)).T
        if int(len(region)) == int(5):
            color[counter] = 'r'
        elif int(len(region)) == int(7):
            color[counter] = 'b'
        elif int(len(region)) == int(6):
            color[counter] = 'g'
        elif int(len(region)) == int(4):
            color[counter] = 'y'
        else:
            color[counter] = 'o'
        counter += 1
    result = Poly3DCollection(verts,zsort='min')
    result.set_edgecolor('black')
    result.set_facecolor(color)
    ax.add_collection3d(result)

    ax.azim = 10
    ax.elev = 40
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_zticks([])
    fig.set_size_inches(12,12)
    ax.set_title(title)

# ---------- Main ----------

data_dir = 'z_Data_1' # Specify Data directory
write = True # Flag to toggle writing of data

NMC = 8 # Specify which Monte-Carlo simulation to grab
N = 500 # Particle number
r = 1 # Sphere radius to initialize particles at (for direct comparison, use unit radius)
center = np.array([0, 0, 0]) # Center of sphere

particles = read_state(N,data_dir,NMC,r) # Read lattice
walesParticles = parseWalesSoln(N,data_dir) # Read Wales et al lattice
# Wales et al. data is available at: https://www-wales.ch.cam.ac.uk/~wales/CCD/Thomson/table.html

fig = plt.figure(figsize=(12,10))
makeVoronoi(particles,r,center,fig,121,'Crystal Ball Voronoi') # Construct Voronoi plots
makeVoronoi(walesParticles,1,center,fig,122,'Wales et al. Voronoi')
plt.show()