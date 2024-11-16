# Description: Generate energy scaling surfaces, fit theoretical model to data.

# ---------- Imports ----------

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt

# ---------- Constants ----------
# SI Units

R = 25.5e-10 # Radius of sphere
ep = 8.85418782e-12 # Permitivitty of free space
q = 1.602e-19 # Elementary charge
Z = 5 # Charge of molecule
k = (1/(4*np.pi*ep))*pow(Z*q,2) # Coulomb force constant, includes charge

# ---------- Plotting Options ----------
# Use LaTeX for rendering
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

# ---------- Fitting Functions ----------

def alpha(N): # Spherical triangle interior angle
    return (N*np.pi)/(3*(N-2))

def d_N(N,R): # Function for lattice constant d(N)
    return R*np.sqrt( (2*(1-2*np.cos(alpha(N))))/(1-np.cos(alpha(N))) )

def r_est(Nrm,N,R): # Length scale estimate correction to d(N)
    return 0.5*np.sqrt( (3*np.sqrt(3))/(2*np.pi) )*(1 + np.sqrt( (4*Nrm - 1)/3 ))*d_N(N,R)

def r_NN(Nrm,N,R): # Length scale estimate of qNN radius
    return 0.5*np.sqrt( (3*np.sqrt(3))/(2*np.pi) )*(3 + np.sqrt( (4*Nrm - 1)/3 ))*d_N(N,R)

def q_hole(Nrm,Z=Z,q=q): # Hole charge (charge of nuclei directly adjacent to hole)
    return ( 3 + np.sqrt( 3*(4*Nrm-1) ) )*Z*q

def q_NN(Nrm,Z=Z,q=q): # Charge of nearest neighbors (charges next to hole charges)
    return ( 9 + np.sqrt( 3*(4*Nrm-1) ) )*Z*q

def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), d(N) modified by r_est, fixed q_hole
    return c*(1/(4*np.pi*ep))*(1/d_N(N,R))*(q_hole(Nrm)*Nrm*Z*q)*(1/q)

def logLikelihood(data,err,c,N,Nrm,R): # Log likelihood
    return np.sum(pow((data-modelFitFunc(N,Nrm,R,c))/err,2))

lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],N_MESH,NRM_MESH,R) # Function that we will minimize

def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
    result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
    print(result)
    c = result.x[0]
    chi2 = logLikelihood(data,err,c,N,Nrm,R)
    return result.x,chi2

# ---------- Analyical Models ----------

def modelFunc(N,Nrm,R): # Model func with length scale d(N) --> r(N,Nrm) = f(Nrm)d(N). Derived using circle area approximation
    return 1*(1/(4*np.pi*ep))*(1/pow(d_N(N,R),2))*(q_hole(Nrm)*q_NN(Nrm))*r_est(Nrm,N,R)*(1/Nrm)*(1/q)

def chi2Model(data,err,N,Nrm,R): # Chi^2 of analytical model
    return np.sum(pow((data-modelFunc(N,Nrm,R))/err,2))

# ---------- Plotting and Averaging ----------

def computeAvgs(avg,err,Narr,numN,NMC,Nblocks,Nlines): # NEED TO FIX ******************************************
    for i in np.linspace(0,numN,numN+1,dtype='int'):
        for j in np.linspace(0,NMC-1,NMC,dtype='int'):
            fname = dir + f'/Scaling Data/N_{Narr[i]}_lattice_{j+1}_scaling.txt'
            temp = pd.read_csv(fname,delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
            nrmArr = temp[0:Nrd*Nblocks,1].astype('int')
            sortNrm = np.argsort(nrmArr)
            data[i,j,:] = temp[sortNrm,2] # Data Array Format: N,NMC,Nrm
        for k in np.linspace(0,Nrd-1,Nrd,dtype='int'):
            avg[i,k] = np.average(data[i,:,k*Nblocks:(1+k)*Nblocks])
            err[i,k] = np.std(data[i,:,k*Nblocks:(1+k)*Nblocks])
    
def plotSingle(Nrm,Tmax_sim,Tmax_scaling,err,N,save=False): # Plot single scaling
    plt.figure(figsize=(12,10))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(Nrm,Tmax_sim,label='Simulation')
    plt.errorbar(Nrm,Tmax_sim,yerr=err,xerr=None,ls='none')
    plt.plot(Nrm,Tmax_scaling,label=r'Scaling Relation')
    plt.title(f'Maximum Kinetic Energy of Individual Particles for N = {N}')
    plt.xlabel('Number of Particles Removed')
    plt.ylabel('Energy (eV)')
    plt.xticks(Nrm)
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f'z_Data_1/N = {N}',bbox_inches='tight')
    plt.show()

def plotAll(N,Nrm,Tmax_sim,Tmax_scaling,err,plotScale=True,save=False): # Plot all scalings
    colors = plt.cm.hsv(np.linspace(0,1,len(N)+1)) # Create unique color for each particle
    plt.figure(figsize=(12,10))
    plt.title(r'$T(N,N_{rm})$ Curves of Constant $N$')
    plt.xlabel(r'$N_{rm}$')
    plt.ylabel('Energy (eV)')
    plt.xticks(Nrm)
    for i, c in zip(np.linspace(1,len(N),len(N),dtype='int'),colors):
        plt.scatter(Nrm,Tmax_sim[i-1,:],color=c)
        plt.errorbar(Nrm,Tmax_sim[i-1,:],yerr=err[i-1,:],xerr=None,ls='none',c=c)
        if plotScale:
            plt.plot(Nrm,Tmax_scaling[i-1,:],label=f'N = {N[i-1]}',c=c)
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f'z_Data_1/N = {N}',bbox_inches='tight')
    plt.show()

def plotFittingSurface(N_MESH,NRM_MESH,model,avg,err):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(N_MESH,NRM_MESH,model,cmap=plt.cm.coolwarm,alpha=0.35,label='Scaling Relation')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax.scatter(N_MESH,NRM_MESH,avg.T,c='black',label='Data')
    for i in np.arange(0,N_MESH.shape[0]):
        for j in np.arange(0,N_MESH.shape[1]):
            ax.plot([N_MESH[i,j], N_MESH[i,j]], [NRM_MESH[i,j], NRM_MESH[i,j]], [avg.T[i,j]+err.T[i,j], avg.T[i,j]-err.T[i,j]], marker="_")
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$N_{rm}$')
    ax.set_zlabel(r'Energy (eV)')
    ax.set_title(r'$T(N,N_{rm})$ Scaling Surface')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10
    ax.legend(loc=2,bbox_to_anchor=(0.1,0.85))
    plt.show()

def plotScalingSurface(R,save=False): # Plot surface of scaling relation as function of N and Nrm
    N = np.linspace(100,1000,901,dtype='int') # Num particles, (3,1000,998)
    Nrm = np.linspace(1,198,198,dtype='int') # Number of particles removed
    X,Y = np.meshgrid(N,Nrm)
    data = modelFunc(X,Y,R)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(12,10))
    surf = ax.plot_surface(X,Y,data,cmap=plt.cm.rainbow,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.75,pad=0.125)
    ax.set_title('Energy Scaling as a Function of N and Nrm')
    ax.set_xlabel('N')
    ax.set_ylabel('Nrm')
    ax.set_zlabel('Peak Energy (eV)')
    if save:
        plt.savefig(f'z_Data_1/Energy Scaling Surface',bbox_inches='tight')
    plt.show()
    return 0

# ---------- Main ----------

dir = 'z_Data_1' # Data directory
NMC = 8 # Number of Monte-Carlo simulations to read
Nrd = 10 # Number of lines to read from data files (i.e., which Nrm values to read)
Nblocks = 3 # Number of blocks data is entered in. For different trials, we write Nrd to the same text file. To parse the files correctly, we have to read the data in blocks.
Nlines = 10 # Number of lines in a block
Nmin = 100 # Number of particles in smallest lattice 
Nmax = 1000 # Number of particles in largest lattice
dN = 100 # Change in N value between subsequent trials
numN = int((Nmax - Nmin)/dN)

freeParams = 1 # Number of free parameters of the model
guess = (1) # Initial guess of free parameter for fitter

plotScale = True # Plot scaling relation
save = False # Save figure

Narr = np.linspace(Nmin,Nmax,numN+1,dtype='int') # Array to loop over for scanning different values of N
Nrm = np.linspace(1,Nrd,Nrd,dtype='int') # Number of particles removed
N_MESH,NRM_MESH = np.meshgrid(Narr,Nrm) # Create meshgrid for parameter space

data = np.zeros([numN+1,NMC,Nrd*Nblocks]) # Format: N,NMC,Nrm
avg = np.zeros([numN+1,Nrd]) # Array for storing averages
err = np.zeros([numN+1,Nrd]) # Table for storing standard deviations

computeAvgs(avg,err,Narr,numN,NMC,Nblocks,Nlines) # Compute average peak energy values

params,chi2 = minimizeLogLikelihood(avg.T,err.T,N_MESH,NRM_MESH,R,guess) # Minimize χ2 to fit model to data
print(f'Fit Parameters: {params}') # Print results
print(f'Reduced χ2 = {chi2/(N_MESH.shape[0]*N_MESH.shape[1] - freeParams)}')
model = modelFitFunc(N_MESH,NRM_MESH,R,*params) # Compute fitting surface

plotFittingSurface(N_MESH,NRM_MESH,model,avg,err) # Plot fitting surface
plotAll(Narr,Nrm,avg,model.T,err,plotScale,save) # Plot curves of constant N of fitting surface