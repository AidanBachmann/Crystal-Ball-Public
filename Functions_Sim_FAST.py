# ---------- Imports ----------

import numpy as np
import pandas as pd
from numba import jit
import os
from multiprocess import Process
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- Plotting Settings ----------

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

# ---------- Plotting and Analysis ----------

def computeSphere(r): # Compute spherical surface
    v = np.linspace(0,np.pi,500)
    u = np.linspace(0,2*np.pi,500)
    
    xs = r*np.outer(np.cos(u),np.sin(v))
    ys = r*np.outer(np.sin(u),np.sin(v))
    zs = r*np.outer(np.ones(np.size(u)),np.cos(v))

    return xs,ys,zs

def makePosPlots3D(particles,Nt,N,time,r): # Make plots of particle position in 3D
    xs,ys,zs = computeSphere(r)
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig = plt.figure(figsize=(12,10)) # Set plot size
    ax = fig.add_subplot(projection='3d') # Create plotting axes
    ax.plot_wireframe(xs,ys,zs,rstride=5,cstride=5,alpha=0.1) # Show sphere
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    for i in np.linspace(0,Nt-1,Nt).astype('int'):
        scatter = ax.scatter(particles[:,i,0],particles[:,i,1],particles[:,i,2],c=colors)
        txt = ax.text(r/4,3*r/4,4*r/5,s=f't = {time[i]}') # Display simulation time
        plt.savefig('step' + "{:05d}".format(i+1),bbox_inches='tight') # Save figure
        scatter.remove() # Remove points
        txt.remove() # Remove text
    plt.close(fig)

def findDist(p1,p2,r,n=-1): # Find arc length distance between two particles
    x1,y1,z1 = p1[n,0],p1[n,1],p1[n,2]
    x2,y2,z2 = p2[n,0],p2[n,1],p2[n,2]
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
            sArr[counter] = findDist(particles[i,:,:],particles[idx,:,:],r)
            counter += 1
    sArr.sort()
    return sArr[0:NN]

def findParticleNN_IDX(particles,idx,N,r,NN=-1,n=-1): # Find index of closest NN particles
    sArr = np.zeros([N-1])
    idxArr = np.zeros([N-1])
    counter = 0
    for i in np.linspace(0,N-1,N,dtype='int'):
        if i != idx:
            sArr[counter] = findDist(particles[i,:,:],particles[idx,:,:],r)
            idxArr[counter] = i
            counter += 1
    sortedArr = np.argsort(sArr)
    idxArr = idxArr[sortedArr]
    return idxArr[0:NN]

def computeAllAvgDist(particles,N,r,NN=-1,n=-1):
    # Compute average distance to NN nearest neighbors for all particles
    sAvg = np.zeros([N])
    for i in np.linspace(0,N-1,N,dtype='int'):
        temp = findParticleNN(particles,i,N,r,NN,n)
        sAvg[i] = np.mean(temp)
    return sAvg

def scaling(N,R): # Scaling of lattice constant
    return R*np.arccos((np.cos((N*np.pi)/(3*(N-2))))/(1-np.cos((N*np.pi)/(3*(N-2)))))

def estimateLatticeConst(dist): # Estimate lattice constant from collection of nearest neighbor distances
    return np.average(dist)

def plotAvgDistHist(particles,N,r,NN,save):
    sAvg = computeAllAvgDist(particles,N,r,NN)
    a_exp = estimateLatticeConst(sAvg) # Lattice constant from data
    a_th = scaling(N,r) # Lattice constant from scaling relation
    print(f'\n\nAverage Constant Estimate: {a_exp}')
    print(f'Theoretical Value: {a_th}')
    print(f'Percent Error: {(np.abs(a_th-a_exp)/a_th)*100}%\n\n')

    fig = plt.figure(figsize=(12,10))
    plt.hist(sAvg,label=f'Sphere Circumference: C = {2*np.pi*r}',bins=50)
    plt.title(f'Average Arclength to {NN} Nearest Neighbors for All Particles')
    plt.xlabel('Average Distance to Nearest Neighbor (m)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    
def makePosPlotsMercator(particles,Nt,N,time,r): # Make plots of particle position in (φ,θ) plane
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig = plt.figure(figsize=(12,10),dpi=72+1/3)
    ax = fig.add_subplot(111)
    ax.set_xlabel('rφ')
    ax.set_ylabel(r'$rln(\tan{(\frac{\pi}{4} + \frac{θ-\frac{\pi}{2})}{2}))}$')
    ylim = 10e-9 # For r = 10e-10, use 4.35e-9. For r = 20e-10, use 9.5e-9
    textPosy = 9.5e-9 # For r = 10e-10, use 3.75e-9. For r = 20e-10, use 9e-9
    textPosx = 2*np.pi*r*0.75
    ax.set_xlim([-0.25e-9,2*np.pi*r*(1+0.025)])
    ax.set_ylim([-ylim,ylim])
    ax.grid()
    for i in np.linspace(0,Nt,Nt+1).astype('int'):
        θ = np.arccos(particles[:,i,2]/r) # Find spherical angles
        φ = np.arctan2(particles[:,i,1],particles[:,i,0]) + np.pi
        x,y = r*φ,r*np.log(np.tan(np.pi/4 + (θ-np.pi/2)/2))
        scatter = ax.scatter(x,y,c=colors)
        txt = ax.text(textPosx,textPosy,s=f't = {time[i]}') # Display simulation time
        plt.savefig('step' + "{:05d}".format(i+1),bbox_inches='tight') # Save figure
        scatter.remove() # Remove points
        txt.remove() # Remove text
    plt.close(fig)
    
def plotIC(particles,N,r,NMC=0): # Plot initial condition with the particle we remove
    fig = plt.figure(figsize=(25,12))

    colors = plt.cm.hsv(np.linspace(0,1,N))
    ax = fig.add_subplot(121)
    θ = np.arccos(particles[:,-1,2]/r) # Find spherical angles
    φ = np.arctan2(particles[:,-1,1],particles[:,-1,0]) + np.pi
    ax.set_xlabel('rφ')
    ax.set_ylabel(r'$rln(\tan{(\frac{\pi}{4} + \frac{θ-\frac{\pi}{2})}{2}))}$')
    ax.set_title(f'Initial Condition for Lattice {NMC}')

    x,y = r*φ,r*np.log(np.tan(np.pi/4 + (θ-np.pi/2)/2))
    ax.scatter(x,y,c=colors)
    ax.legend() 
    
    plt.show()

def plotEnergyAll(particles,Nt,time,N,Nrm,NMC,r,m,pIdx,data_dir,log=False,save=False):
    fig,ax = plt.subplots(2,2,figsize=(12,10))
    colors = plt.cm.hsv(np.linspace(0,1,N))
    
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Energy (eV)')
    ax[0,0].set_title('Kinetic Energy Time Series')
    ax[0,0].grid()
    
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Energy (eV)')
    ax[0,1].set_title('Potential Energy Time Series')
    ax[0,1].grid()
    
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Energy (eV)')
    ax[1,0].set_title('Individual Particle Total Energy')
    ax[1,0].grid()
    
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].set_ylabel('Energy (eV)')
    ax[1,1].set_title('System Total Energy')
    ax[1,1].grid()
    
    ke = np.zeros([N,Nt+1]) # Kinetic energy array
    E = np.zeros([Nt+1]) # Total energy array
    
    for i in np.linspace(0,N-1,N,dtype='int'):
        vx,vy,vz = particles[i,:,3],particles[i,:,4],particles[i,:,5]
        ke[i,:] = 0.5*m*(pow(vx,2) + pow(vy,2) + pow(vz,2))*(1/1.602176634e-19) # Compute kinetic energy
        E[:] += ke[i,:] # Add kinetic energy to total

        ax[0,0].plot(time,ke[i,:],c=colors[i]) # Plot kinetic energy time series
        ax[0,1].plot(time,particles[i,:,9],c=colors[i]) # Plot potential energy time series
        ax[1,0].plot(time,ke[i,:] + particles[i,:,9],c=colors[i]) # Plot individual particle total energies
    
    U = 0.5*np.sum(particles[:,:,9],axis=0) # Total potential energy of system
    E[:] += U # Add total potential to total kinetic
    ax[1,1].plot(time,E) # Plot system total energy
    
    if log:
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
         
    if save:
        if log:
            plt.savefig(f'{data_dir}/Output Sim/Lattice_{NMC}_N_{N+Nrm}_Nrm_{Nrm}_Elog.png',bbox_inches='tight',format='png')
        else:
            plt.savefig(f'{data_dir}/Output Sim/Lattice_{NMC}_N_{N+Nrm}_Nrm_{Nrm}_E.png',bbox_inches='tight',format='png')
    plt.close(fig)
    
    write_Emax(ke,N+Nrm,Nrm,NMC,pIdx,data_dir)

def clean(): # Remove all PNGs in directory
    for file in os.listdir(os.getcwd()):
        if file.startswith("step"):
            os.remove(file) 
            
def cleanAll():
    for file in os.listdir(os.getcwd()):
        if file.endswith('.png') or file.endswith('.gif'):
            os.remove(file) 

def read_state(N,r,lattice,rInit=10e-10,NMC=0): # Read state for simulation
    # If initialized lattice with N particles, pass N-1 to this function to 
    # remove the last particle. rInit is the radius used to initialize thie lattice.
    particles = np.zeros([N,1,3]) # Array to grab particles
    lattice += f'N_{N}_lattice_{NMC}.txt' # Get name of text file holding lattice
    data = pd.read_csv(lattice,delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
    for i in np.linspace(0,N-1,N).astype('int'):
        x,y,z = data[i,0],data[i,1],data[i,2]
        θ = np.arccos(z/rInit) # Compute angle of particles
        φ = np.arctan2(y,x)
        particles[i,0,0] = r*np.sin(θ)*np.cos(φ) # Save x,y,z
        particles[i,0,1] = r*np.sin(θ)*np.sin(φ)
        particles[i,0,2] = r*np.cos(θ)
    return particles

def write_Emax(KE,N,Nrm,NMC,pIdx,data_dir): # Write maximum kinetic energy to file
    Emax = np.max(KE)
    f = open(f'{data_dir}/Scaling Data/N_{N}_lattice_{NMC}_scaling.txt','a')
    f.write(f'{N},{Nrm},{Emax},{pIdx}\n')
    f.close()




# ---------- Time Stepping ----------
# Convention for the particles array:
# particles[N,Nt,10] --> particle, time step, x,y,z,vx,vy,vz,ax,ay,az,U

@jit(nopython = True)
def updateTimeStep(dt,particles,C,max_dt=np.inf): # Adaptive time step
    # Magnitude of the velocity
    v = np.sqrt( particles[:,1,3]**2 + particles[:,1,4]**2 + particles[:,1,5]**2 )
    vmax = np.max(v) # Find maximum velocity
    T = 0.5*pow(vmax,2) # Max kinetic energy at this time step
    if vmax > 0:
        newdt = (C/vmax)*(1/1e5)
        if newdt < max_dt:
            return newdt,T
        else:
            return max_dt,T
    else:
        return dt,T

@jit(nopython = True)
def findNhat(x,y,z): # Find normal vector for the sphere
    θ = np.arccos(z/np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))) # Find spherical angles
    φ = np.arctan(y/x)
    rhat = np.array([np.sin(θ)*np.cos(φ),np.sin(θ)*np.sin(φ),np.cos(θ)]) # Normal vector
    return rhat

@jit(nopython = True)
def computeForce(r1,r2,k,m): # Compute force of p1 on p2, defaults to most recent step
    R = r2 - r1 # Separation vector
    R2 = R @ R
    Rhat = R/np.sqrt(R2)
    a = (k/m)*(Rhat/R2) # Acceleration due to Coulomb force
    return a

@jit(nopython = True)
def projectVector(a,x,y,z,r): # Project vector onto surface of sphere, (x,y,z) MUST be a point on the sphere
    θ = np.arccos(z/r) # Find spherical angles
    φ = np.arctan2(y,x)
    rhat = np.array([np.sin(θ)*np.cos(φ),np.sin(θ)*np.sin(φ),np.cos(θ)]) # Normal vector
    aNew = a - (a @ rhat)*rhat # Subtract components of acceleration along normal
    return aNew

@jit(nopython = True)
def projectVectorVec(a,x,y,z,r): # Project vector onto surface of sphere, vectorized form
    θ = np.arccos(z/r) # Find spherical angles
    φ = np.arctan2(y,x)
    rhat = np.vstack((np.sin(θ)*np.cos(φ),np.sin(θ)*np.sin(φ),np.cos(θ))).T # Normal vector
    proj = (a*rhat).sum(axis=1) # Compute dot product
    temp = np.vstack((proj,proj,proj)).T # Place into array to compute rescaling
    aNew = a - temp*rhat # Subtract components of acceleration along normal
    return aNew

@jit(nopython = True)
def pushParticle(p,dt,r):
    # Key:
    # x,y,z = p[n,0],p[n,1],p[n,2] # Position, velocity acceleration at current time
    # vx,vy,vz = p[n,3],p[n,4],p[n,5]
    # ax,ay,az = p[n+1,6],p[n+1,7],p[n+1,8]
    
    vxnew = p[:,0,3] + p[:,1,6]*dt # Update velocity
    vynew = p[:,0,4] + p[:,1,7]*dt
    vznew = p[:,0,5] + p[:,1,8]*dt
    
    xnew = p[:,0,0] + 0.5*(vxnew+p[:,0,3])*dt + 0.5*p[:,1,6]*pow(dt,2) # Update position
    ynew = p[:,0,1] + 0.5*(vynew+p[:,0,4])*dt + 0.5*p[:,1,7]*pow(dt,2)
    znew = p[:,0,2] + 0.5*(vznew+p[:,0,5])*dt + 0.5*p[:,1,8]*pow(dt,2)

    rnew = np.vstack((xnew,ynew,znew)).T # Assemble new position into array
    vnew = np.vstack((vxnew,vynew,vznew)).T # Assemble new velocity into array
    
    rmag = (rnew*rnew).sum(axis=1) # Compute squared magnitude of r
    rProjected = (r/np.sqrt( np.vstack((rmag,rmag,rmag)).T ))*rnew # Rescale position so it's on the sphere
    vProjected = projectVectorVec(vnew,rProjected[:,0],rProjected[:,1],rProjected[:,2],r) # Project v onto sphere
    
    p[:,1,0],p[:,1,1],p[:,1,2] = rProjected[:,0],rProjected[:,1],rProjected[:,2] # Append position
    p[:,1,3],p[:,1,4],p[:,1,5] = vProjected[:,0],vProjected[:,1],vProjected[:,2] # Append velocity




# ---------- Simulation Functions ----------

@jit(nopython = True)
def rmOneParticle(particles,simParticles,pIdx,N): #pIdx is the particle to remove  
    simParticles[0:pIdx,0,0] = particles[0:pIdx,-1,0] # Save particle position up to (but not including)
    simParticles[0:pIdx,0,1] = particles[0:pIdx,-1,1] # index pIdx
    simParticles[0:pIdx,0,2] = particles[0:pIdx,-1,2]
    
    simParticles[pIdx:N-1,0,0] = particles[pIdx+1:N,-1,0] # Save particle position from inde pIdx + 1 to
    simParticles[pIdx:N-1,0,1] = particles[pIdx+1:N,-1,1] # last particle
    simParticles[pIdx:N-1,0,2] = particles[pIdx+1:N,-1,2]
    
    return simParticles

def rmChunk(particles,simParticles,N,pIdx0,Nrm,r): # Remove chunk of particles
    idx = findParticleNN_IDX(particles,pIdx0,N,r,Nrm-1,0).astype('int') # Find nearest neighbors
    idx = np.concatenate((np.array([pIdx0]),idx)) # Save indices of all particles to remove
    counter = 0
    for i in np.linspace(0,N-1,N,dtype='int'):
        flag = True
        for j in np.linspace(0,Nrm-1,Nrm,dtype='int'):
            if i == idx[j]:
                flag = False
        if flag:
            simParticles[counter,0,0:3] = particles[i,0,:]
            counter += 1
    return simParticles,idx

@jit(nopython = True)
def computeFinalPotential(particles,N,k,r,m): # Compute the potential energy of the final configuration
# This functions exists only to compute the final time step potential. For all other steps, this is done
# within the loop to compute the force for the sake of efficiency. However, the final step is not computed
# in the loop because we compute up to Nt - 1 time steps since we update to the i+1th state. As a result, we 
# must compute the last energy value after we complete the final step.
    for i in np.linspace(0,N-1,N).astype('int'):
        Etemp = 0 # Variable to store potential energy
        r2 = np.array([particles[i,-1,0],particles[i,-1,1],particles[i,-1,2]]) # Position of particle 2
        for j in np.linspace(0,N-1,N).astype('int'):
            if i != j:
                r1 = np.array([particles[j,-1,0],particles[j,-1,1],particles[j,-1,2]]) # Position of particle 1
                Etemp += k*(1/np.sqrt((r2-r1)@(r2-r1)))
        particles[i,-1,9] = Etemp*(1/1.602176634e-19) # Save potential energy (in eV)

@jit(nopython = True)
def computeAllForce(particles,N,k,r,m): # Compute forces from all particles for sim, no drag term
    for i in np.linspace(0,N-1,N).astype('int'):
        r2 = np.array([particles[i,0,0],particles[i,0,1],particles[i,0,2]]) # Position of particle 2
        for j in np.linspace(i+1,N-1,N-i-1).astype('int'):
            r1 = np.array([particles[j,0,0],particles[j,0,1],particles[j,0,2]]) # Position of particle 1
            particles[i,1,6:9] += computeForce(r1,r2,k,m) # Force of particle 1 on particle 2
            particles[j,1,6:9] += -1*particles[i,1,6:9] # Force of particle 2 on particle 1
        particles[i,1,6:9] = projectVector(particles[i,1,6:9],r2[0],r2[1],r2[2],r) # Project acceleration onto tangent plane

@jit(nopython = True)
def updateStateNoDrag(particles,N,k,r,m,dt): # Compute single step
    computeAllForce(particles,N,k,r,m)
    pushParticle(particles,dt,r)




# ---------- Main Functions ----------

@jit(nopython = True)
def simulation(particles,N,Nt,k,r,m,dt,C,max_dt): # Run simulation
    t = 0
    Tmax = 0
    for i in np.linspace(0,Nt-1,Nt).astype('int'):
        updateStateNoDrag(particles,N,k,r,m,dt)
        t += dt
        #print('Finished time step '+str(i+1)+', t = ',time[i],'. '+str(Nt-i-1)+' steps remaining.')
        dt,T = updateTimeStep(dt,particles,C,max_dt)
        if T > Tmax:
            Tmax = T
        particles[:,0,:] = particles[:,1,:] # Set current step to updated step
        particles[:,1,:] = 0 # Set next step back to zero
    print('DONE\n')
    return Tmax

def simSingle(N,Nt,Nrm,k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir,save=True,log=False,generatePlots=False,debug=False): # Single simulation
    particles = read_state(N,r,lattice,r,NMC) # Read lattice from file
    # Format: N-Nrm particles, 2 slots for current step and next step, 9 slots for r,rdot,rdotdot
    simParticles = np.zeros([N-Nrm,2,9]) # New array for particles
    if Nrm == 1:
        rmOneParticle(particles,simParticles,pIdx,N) # Remove particle at index pIdx
    elif Nrm > 1:
        rmChunk(particles,simParticles,N,pIdx,Nrm,r) # Remove particles around pIdx
    print(f'Started Simulation with {N-Nrm} particles.')
    N -= Nrm # Reduce number of particles to account for removed particle
    if debug:
        plotIC(simParticles,N,r,NMC=0) # Plot initial condition
    Tmax = simulation(simParticles,N,Nt,k,r,m,dt,C,max_dt) # Run simulation
    write_Emax(Tmax*m*(1/1.602176634e-19),N+Nrm,Nrm,NMC,pIdx,data_dir)
    
def simMC_Multi(N,Nt,Nrm,k,r,m,dt,C,max_dt,pIdx,lattice,NMC,data_dir,save=True,log=False,generatePlots=False,debug=False): # NMC simulations on NMC processes
    print(f'Code started on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.\n')
    start = time.time() # Save start time
    numProcs = np.linspace(1,NMC,NMC,dtype='int')
    procs = []
    # instantiating process with arguments
    for i in numProcs:
        args_ = (N,Nt,Nrm,k,r,m,dt,C,max_dt,pIdx,lattice,i,data_dir,save,log,generatePlots,debug,)
        proc = Process(target=simSingle,args=args_)
        procs.append(proc)
        proc.start()
    # complete the processes
    for proc in procs:
        proc.join()
    end = time.time() # Save end time
    print(f'Code finished in {end - start} seconds on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.')




# ---------- Debugging Tools ----------

def printState(p,n): # Print single particle state at time step n
    print(f'Time Step {n}\n')
    print(f'x = {p[n,0]}, y = {p[n,1]}, z = {p[n,2]}')
    print(f'vx = {p[n,3]}, vy = {p[n,4]}, vz = {p[n,5]}')
    print(f'ax = {p[n,6]}, ay = {p[n,7]}, az = {p[n,8]}\n\n')
    
def printAllPos(p,n,N): # Print particle positions for time step n
    for i in np.linspace(0,N-1,N,dtype='int'):
        print(f'x = {p[i,n,0]}, y = {p[i,n,1]}, z = {p[i,n,2]}')
        
def printStateInfo(p,Nt): # Print all computed time steps for particle p
    for i in np.linspace(0,Nt-1,Nt,dtype='int'):
        printState(p,i)