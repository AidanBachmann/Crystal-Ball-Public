# ---------- Imports ----------

import numpy as np
import pandas as pd
from scipy.linalg import expm
import math
from numba import jit,types
from numba.extending import typeof_impl
import pickle
import glob
from PIL import Image
import os
import multiprocess as mp
from multiprocess import Process
from itertools import product
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

def finalTimeStepMercator(particles,N,r,data_dir,NMC=1):
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig = plt.figure(figsize=(12,10)) # Set plot size
    ax = fig.add_subplot(111)
    ax.set_xlabel('rφ')
    ax.set_ylabel(r'$rln(\tan{(\frac{\pi}{4} + \frac{θ-\frac{\pi}{2})}{2}))}$')
    ylim = 10e-9 # For r = 10e-10, use 4.35e-9. For r = 20e-10, use 9.5e-9
    textPos = 3.75e-9 # For r = 10e-10, use 3.75e-9. For r = 20e-10, use 
    ax.set_xlim([-0.25e-9,2*np.pi*r*(1+0.025)])
    ax.set_ylim([-ylim,ylim])
    ax.grid()
    θ = np.arccos(particles[:,0,2]/r) # Find spherical angles
    φ = np.arctan2(particles[:,0,1],particles[:,0,0]) + np.pi
    x,y = r*φ,r*np.log(np.tan(np.pi/4 + (θ-np.pi/2)/2))
    scatter = ax.scatter(x,y,c=colors)
    plt.savefig(f'{data_dir}/Output Init/N_{N}_Final_Lattice_{NMC}.png',bbox_inches='tight') # Save figure
    plt.close(fig)
    
    
def makePosPlotsMercator(particles,Nt,N,time,r): # Make plots of particle position in (φ,θ) plane
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig = plt.figure(figsize=(12,10)) # Set plot size
    ax = fig.add_subplot(111)
    ax.set_xlabel('rφ')
    ax.set_ylabel(r'$rln(\tan{(\frac{\pi}{4} + \frac{θ-\frac{\pi}{2})}{2}))}$')
    ylim = 4.35e-9 # For r = 10e-10, use 4.35e-9. For r = 20e-10, use 9.5e-9
    textPos = 9e-9 # For r = 10e-10, use 3.75e-9. For r = 20e-10, use 
    ax.set_xlim([-0.25e-9,2*np.pi*r*(1+0.025)])
    ax.set_ylim([-ylim,ylim])
    ax.grid()
    for i in np.linspace(0,Nt-1,Nt).astype('int'):
        θ = np.arccos(particles[:,i,2]/r) # Find spherical angles
        φ = np.arctan2(particles[:,i,1],particles[:,i,0]) + np.pi
        x,y = r*φ,r*np.log(np.tan(np.pi/4 + (θ-np.pi/2)/2))
        scatter = ax.scatter(x,y,c=colors)
        txt = ax.text(4.75e-9,textPos,s=f't = {time[i]}') # Display simulation time
        plt.savefig('step' + "{:05d}".format(i+1),bbox_inches='tight') # Save figure
        scatter.remove() # Remove points
        txt.remove() # Remove text
    plt.close(fig)

def plotVt(N,p,particleIdx,time,data_dir,log=False,save=False,NMC=1): # Plot velocity time series for a particle
    vx,vy,vz = p[particleIdx,:,3],p[particleIdx,:,4],p[particleIdx,:,5] # Get velocities
    vNorm = np.sqrt(pow(vx,2) + pow(vy,2) + pow(vz,2)) # Compute the norm

    fig,ax = plt.subplots(2,2,figsize=(12,10))
    
    ax[0,0].plot(time,vx) # Plot vx time series
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel(r'$v_{x} (m/s)$')
    ax[0,0].set_title(r'$v_{x}$ Time Series')
    ax[0,0].grid()
    
    ax[0,1].plot(time,vy) # Plot vy time series
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel(r'$v_{y} (m/s)$')
    ax[0,1].set_title(r'$v_{y}$ Time Series')
    ax[0,1].grid()
    
    ax[1,0].plot(time,vz) # Plot norm of v time series
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel(r'$v_{z} (m/s)$')
    ax[1,0].set_title(r'$v_{z}$ Time Series')
    ax[1,0].grid()
    
    ax[1,1].plot(time,vNorm) # Plot velocity space time series 
    ax[1,1].set_xlabel(r'Time (s)')
    ax[1,1].set_ylabel(r'|v|')
    ax[1,1].set_title(r'Velocity Norm Time Series')
    ax[1,1].grid()
      
    if log:
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xscale('log')
    
    if save:
        plt.savefig(f'{data_dir}/Output Init/N_{N}_Lattice_{NMC}_Vt_P{particleIdx+1}.png',bbox_inches='tight',format='png')
    plt.close(fig)
    
def plotVtFinal(particles,N,data_dir,log=False,save=False,NMC=1):
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    ax.set_xlabel(r'Particle Index')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.set_title(r'Velocity Norm at Final Time')
    ax.grid()

    Narr = np.linspace(1,N,N,dtype='int') # Array of particle indices
    vx,vy,vz = particles[:,0,3],particles[:,0,4],particles[:,0,5]
    vNorm = np.sqrt(pow(vx,2) + pow(vy,2) + pow(vz,2))
    ax.scatter(Narr,vNorm,c=colors) # Plot norm of v time series
         
    if log:
        ax.set_yscale('log')
        
    if save:
        plt.savefig(f'{data_dir}/Output Init/N_{N}_Lattice_{NMC}_Vt_Final.png',bbox_inches='tight',format='png')
    else:
        plt.show()
    plt.close(fig)
           
def findDist(p1,p2,r): # Find arc length distance between two particles
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    r1 = np.array([x1,y1,z1])
    r2 = np.array([x2,y2,z2])
    
    R = r1 - r2 # Separation vector
    Rmag = np.sqrt(R @ R) # Magnitude of separation vector
    gamma = 2*np.arcsin(Rmag/(2*r)) # Angle between r1 and r2
    s = r*gamma # Arc length separation of two particles

    s = Rmag # Get linear distance
    
    return s
    
def findParticleNN(particles,idx,N,r,NN=-1,n=-1): # Compute distance to N nearest neighbors for one particle
    # Note: NN is number of nearest neighbors to find, n is time step (default is final step)
    sArr = np.zeros([N-1])
    counter = 0
    for i in np.linspace(0,N-1,N,dtype='int'):
        if i != idx:
            sArr[counter] = findDist(particles[i,0,:],particles[idx,0,:],r)
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
    return R*np.arccos((np.cos((N*np.pi)/(3*(N-2))))/(1-np.cos((N*np.pi)/(3*(N-2)))))

def estimateLatticeConst(dist): # Estimate lattice constant from collection of nearest neighbor distances
    return np.average(dist)

def writeLConst(N,a_ex,a_th,data_dir): # Write lattice constant to file
    f = open(f'{data_dir}/Lattices/Lattice Consts.txt','a')
    f.write(f'{N},{a_ex},{a_th}\n')
    f.close()

def plotAvgDistHist(particles,N,r,NN,data_dir,save,NMC=1):
    sAvg = computeAllAvgDist(particles,N,r,NN)*(1e10) # Get distribution in angstroms
    writeLConst(N,estimateLatticeConst(sAvg),scaling(N,r),data_dir)

    fig = plt.figure(figsize=(12,10))
    plt.hist(sAvg,bins=50)
    #plt.text(0.75*np.max(sAvg)*1e10,0.1*N,s='C = {round(2*np.pi*r*1e10,3)}Å',fontsize='large')
    plt.text(np.min(sAvg)+0.75*(np.max(sAvg)-np.min(sAvg)),70,s=f'$C$ = {round(2*np.pi*r*1e10,3)}Å',fontsize='large')
    plt.title(f'Average Distance to {NN} Nearest Neighbors for All Particles')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Counts')
    plt.grid()
    #plt.legend()
    
    if save:
        plt.savefig(f'{data_dir}/Output Init/N_{N}_Lattice_{NMC}_NN_Hist.png',bbox_inches='tight',format='png')
    else:
        plt.show()
    plt.close(fig)

def computeLateTimeVavg(particles,N,n):
    avg,std,vAvg,vStd = np.zeros([N,3]),np.zeros([N,3]),np.zeros([N]),np.zeros([N])
    particleIdx = np.linspace(0,N-1,N,dtype='int')
    for i in particleIdx:
        vx,vy,vz = particles[i,n:-1,3],particles[i,n:-1,4],particles[i,n:-1,5] # Get velocities
        vNorm = np.sqrt(pow(vx,2) + pow(vy,2) + pow(vz,2)) # Compute the norm
        vAvg[i] = np.average(vNorm) # Compute average velocity
        vStd[i] = np.std(vNorm) # Compute standard deviation of average velocity
        
        x,y,z = particles[i,n:-1,0],particles[i,n:-1,1],particles[i,n:-1,2] # Get positions
        avg[i,0],avg[i,1],avg[i,2] = np.average(x),np.average(y),np.average(z) # Compute averages and STDs
        std[i,0],std[i,1],std[i,2] = np.std(x),np.std(y),np.std(z)
        
    return avg,std,vAvg,vStd,particleIdx 

def plotLateTimeVavg(particles,N,n,time,data_dir,save=True,NMC=1): 
    avg,std,vAvg,vStd,pIdx = computeLateTimeVavg(particles,N,n)
    colors = plt.cm.hsv(np.linspace(0,1,N)) # Create unique color for each particle
    
    fig,ax = plt.subplots(2,2,figsize=(12,10))
    
    ax[0,0].errorbar(pIdx,vAvg,yerr=vStd,xerr=None,ls='none',c='Black',alpha=0.5,label='Spread')
    ax[0,0].scatter(pIdx,vAvg,label='Average Velocity',c=colors)
    ax[0,0].set_title('Time Averaged Magnitude of Velocity at Late Times')
    ax[0,0].set_xlabel('Particle Index')
    ax[0,0].set_ylabel('Magnitude of Velocity (m/s)')
    ax[0,0].grid()
    ax[0,0].legend()
    
    ax[0,1].errorbar(avg[:,0],avg[:,1],xerr=std[:,0],yerr=std[:,1],ls='none',c='Black',alpha=0.5,label='Spread')
    ax[0,1].scatter(avg[:,0],avg[:,1],c=colors,label='Average Positions')
    ax[0,1].set_title('Time Averaged Position in XY Plane at Late Times')
    ax[0,1].set_xlabel('x (m)')
    ax[0,1].set_ylabel('y (m)')
    ax[0,1].grid()
    ax[0,1].legend()
    
    ax[1,0].errorbar(avg[:,1],avg[:,2],xerr=std[:,1],yerr=std[:,2],ls='none',c='Black',alpha=0.5,label='Spread')
    ax[1,0].scatter(avg[:,1],avg[:,2],c=colors,label='Average Positions')
    ax[1,0].set_title('Time Averaged Position in YZ Plane at Late Times')
    ax[1,0].set_xlabel('y (m)')
    ax[1,0].set_ylabel('z (m)')
    ax[1,0].grid()
    ax[1,0].legend()
    
    ax[1,1].errorbar(avg[:,0],avg[:,2],xerr=std[:,0],yerr=std[:,2],ls='none',c='Black',alpha=0.5,label='Spread')
    ax[1,1].scatter(avg[:,0],avg[:,2],c=colors,label='Average Positions')
    ax[1,1].set_title('Time Averaged Position in XZ Plane at Late Times')
    ax[1,1].set_xlabel('x (m)')
    ax[1,1].set_ylabel('z (m)')
    ax[1,1].grid()
    ax[1,1].legend()
    
    if save:
        plt.savefig(f'{data_dir}/Output Init/N_{N}_Lattice_{NMC}_Vt_Late_Time_Avg_{time[n]}.png',bbox_inches='tight',format='png')
    plt.close(fig)

def initLatticeAnalysis(particles,N,NN,r,data_dir,save=True,NMC=1): # Combined analysis tools for each simulation, combines most functions above
    plotAvgDistHist(particles,N,r,NN,data_dir,save,NMC) # Plot average distance hist for NN nearest neighbors

def plotLatticeConsts(data_dir,NMC,save=True):
    data = pd.read_csv(f'{data_dir}/Lattices/Lattice Consts.txt',delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
    numNVals = int(data.shape[0]/NMC)
    avgs = np.zeros([numNVals,2]) # Average and std of experimental lattice constants
    theory = np.zeros([numNVals]) # Theoretical values
    Nvals = np.zeros([numNVals])
    err = np.zeros([numNVals]) # Percent error between theory and experiment
    for i in np.linspace(0,numNVals-1,numNVals,dtype='int'):
        avgs[i,0] = np.average(data[i*NMC:(i+1)*NMC,1])
        avgs[i,1] = np.std(data[i*NMC:(i+1)*NMC,1])
        theory[i] = data[i*NMC,2] #+ 0.0425e-10
        err[i] = (np.abs(theory[i]-avgs[i,0])/(theory[i]))*100
        Nvals[i] = data[i*NMC,0]
    plt.figure(figsize=(12,10))
    plt.scatter(Nvals,theory,label='Theoretical Prediction',marker='+')
    plt.scatter(Nvals,avgs[:,0],label='Simulation Value')
    plt.errorbar(Nvals,avgs[:,0],yerr=avgs[:,1],xerr=None,ls='none',label='Standard Deviation',c='black')
    plt.xlabel('Number of Particles')
    plt.ylabel('Lattice Constant')
    plt.xticks(Nvals)
    plt.ylim(0,np.max(avgs[:,0])*(1+0.25))
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f'{data_dir}/Output Init/Lattice Constant Scaling',bbox_inches='tight')
    else:
        plt.show()
    
    plt.figure(figsize=(12,10))
    plt.scatter(Nvals,err)
    plt.title('Percent Error Between Theory and Simulation')
    plt.xlabel('N')
    plt.ylabel('Percent Error')
    plt.grid()
    if save:
        plt.savefig(f'{data_dir}/Output Init/Lattice Constant Scaling Error',bbox_inches='tight')
    else:
        plt.show()

def clean(): # Remove all PNGs in directory
    for file in os.listdir(os.getcwd()):
        if file.startswith("step"):
            os.remove(file) 
            
def cleanAll():
    for file in os.listdir(os.getcwd()):
        if file.endswith('.png') or file.endswith('.gif'):
            os.remove(file) 

def write_state(N,particles,data_dir,NMC=1): # Write final position to file lattice.txt
    f = open(f'{data_dir}/Lattices/N_{N}_lattice_{NMC}.txt','w')
    for i in np.linspace(0,N-1,N).astype('int'):
        f.write(f'{particles[i,0,0]},{particles[i,0,1]},{particles[i,0,2]}\n')





# ---------- Initial Conditions ----------

def initStateUniform(theta,phi,r):
    x = r*np.sin(theta)*np.cos(phi) # x
    y = r*np.sin(theta)*np.sin(phi) # y
    z = r*np.cos(theta) # z
    return x,y,z

def initParticles(particles,N,r): # Init particle array, N is number of particles    
    for i in np.linspace(0,N-1,N).astype('int'):
        x,y,z = initStateUniform(np.random.uniform(low=0,high=np.pi),
                         np.random.uniform(low=0,high=2*np.pi),
                         r)
        particles[i,0,0] = x
        particles[i,0,1] = y
        particles[i,0,2] = z
    print(f'Initialized {N} particles.')
    return particles




# ---------- Time Stepping ----------
# Convention for the particles array:
# particles[N,Nt,9] --> particle, time step, x,y,z,vx,vy,vz,ax,ay,az

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
def dragTerm(v,nu,t,tau): # Compute drag
    v2 = v @ v # Quadtratic drag
    if v2 != 0:
        vhat = v/np.sqrt(v2)
        drag = -nu*v2*vhat # Quadratic drag
        return drag
    else:
        return np.array([0.0,0.0,0.0])
    
    #return -nu*v*np.exp(-t/(tau)) # Linear drag

@jit(nopython = True)
def computeAllForce(particles,N,k,r,m,nu,t,tau): # Compute forces from all particles for sim, no drag term
    for i in np.linspace(0,N-1,N).astype('int'):
        r2 = np.array([particles[i,0,0],particles[i,0,1],particles[i,0,2]]) # Position of particle 2
        for j in np.linspace(i+1,N-1,N-i-1).astype('int'):
            r1 = np.array([particles[j,0,0],particles[j,0,1],particles[j,0,2]]) # Position of particle 1
            particles[i,1,6:9] += computeForce(r1,r2,k,m) # Force of particle 1 on particle 2
            particles[j,1,6:9] += -1*particles[i,1,6:9] # Force of particle 2 on particle 1
        drag = dragTerm(np.array([particles[i,0,3],particles[i,0,4],particles[i,0,5]]),nu,t,tau) # Compute drag force at current step
        particles[i,1,6:9] = projectVector(particles[i,1,6:9]+drag,r2[0],r2[1],r2[2],r) # Project acceleration onto tangent plane

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

@jit(nopython = True)
def updateState(particles,N,k,r,m,nu,dt,t,tau): # Compute single step
    computeAllForce(particles,N,k,r,m,nu,t,tau)
    pushParticle(particles,dt,r)

@jit(nopython = True)
def updateTimeStep(dt,particles,C,max_dt=np.inf): # Adaptive time step
    # Magnitude of the velocity
    v = np.sqrt( particles[:,1,3]**2 + particles[:,1,4]**2 + particles[:,1,5]**2 )
    vmax = np.max(v) # Find maximum velocity
    if vmax > 0:
        newdt = (C/vmax)*(1/1e3)
        if newdt < max_dt:
            return newdt
        else:
            return max_dt
    else:
        return dt




# ---------- Main Functions ----------

@jit(nopython = True)
def initLattice(particles,N,Nt,k,r,m,nu,dt,max_dt,C,tau): # Initialize lattice
    t = 0
    for i in np.linspace(0,Nt-1,Nt).astype('int'):
        t += dt
        updateState(particles,N,k,r,m,nu,dt,t,tau)
        dt = updateTimeStep(dt,particles,C,max_dt)
        particles[:,0,:] = particles[:,1,:] # Set current step to updated step
        particles[:,1,:] = 0 # Set next step back to zero
        #print('Finished time step '+str(i+1)+', t = ',t,'. '+str(Nt-i-1)+' steps remaining.')
    print('DONE\n')

def initSingle(N,Nt,NN,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write=True,save=True,log=False,tau=1): # Init single lattice with N particles
    np.random.seed(seed=seed)
    particles = np.zeros([N,2,9])
    initParticles(particles,N,r) # Init state
    start = time.time() # Save start time
    initLattice(particles,N,Nt,k,r,m,nu,dt,max_dt,C,tau) # Run simulation
    end = time.time() # Save end time
    #print(f'Lattice initializaiton completed in {end - start} seconds.')
    if write:
        write_state(N,particles,data_dir,NMC)
    start = time.time() # Save start time
    plotAvgDistHist(particles,N,r,NN,data_dir,save,NMC) # Plot average distance hist for NN nearest neighbors
    plotVtFinal(particles,N,data_dir,log,save,NMC)
    finalTimeStepMercator(particles,N,r,data_dir,NMC)
    end = time.time() # Save end time
    #print(f'Figures generated in {end - start} seconds.\n') # Print time to create figures
        
def initMC_Linear(N,Nt,k,r,m,nu,dt,max_dt,C,NMC,seed,write,save): # Init NMC lattices for constant N
    for i in np.linspace(1,NMC,NMC).astype('int'):
        initSingle(N,Nt,k,r,m,nu,dt,i,seed*i,write,save)

def initMC_Multi(N,Nt,NN,n,particleIdx,k,r,m,nu,dt,max_dt,C,NMC,seed,data_dir,write,save,log,tau): # Init NMC lattices for constant N in parallel
    print(f'Code started on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.\n')
    start = time.time() # Save start time
    numProcs = np.linspace(1,NMC,NMC,dtype='int')
    procs = []
    # instantiating process with arguments
    for i in numProcs:
        args_ = (N,Nt,NN,k,r,m,nu,dt,max_dt,C,i,seed+i,data_dir,write,save,log,tau,)
        proc = Process(target=initSingle,args=args_)
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