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
NMC = 8 # Number of Monte-Carlo simulations to read
Nrd = 10 # Number of lines to read from data files (i.e., which Nrm values to read)
Nblocks = 3 # Number of blocks data is entered in. For different trials, we write Nrd to the same text file. To parse the files correctly, we have to read the data in blocks.
Nlines = 10 # Number of lines in a block
Nmin = 100 # Number of particles in smallest lattice 
Nmax = 1000 # Number of particles in largest lattice, 500
dN = 100 # Change in N value between subsequent trials
numN = int((Nmax - Nmin)/dN)
#dir = 'z_Data_1/Scaling Data - Run 1/' # Directory storing lattices
dir = 'z_Data_4/Scaling Data/'
save = False # Save figures

# ---------- Data Arrays ----------

Narr = np.linspace(Nmin,Nmax,numN+1,dtype='int') # Array to loop over for scanning different values of N
Nrm = np.linspace(1,Nrd,Nrd,dtype='int') # Number of particles removed
N_MESH,NRM_MESH = np.meshgrid(Narr,Nrm) # Create meshgrid for parameter space

data = np.zeros([numN+1,NMC,Nrd*Nblocks]) # Format: N,NMC,Nrm
avg = np.zeros([numN+1,Nrd]) # Array for storing averages
err = np.zeros([numN+1,Nrd]) # Table for storing standard deviations

# ---------- Plotting Options ----------
# Use LaTeX for rendering
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

# ---------- Functions to Find Equilibrium of 1D Work Formulation of Wsys ----------

# f is the ratio f = q_hole/qNN
# c is the system length scale c = r + d

def delta1(f,c): # First repeated term of the equation
    return pow( 54*pow(c,6)*f + pow(c,6)*pow(f,3) - 6*np.sqrt(3)*np.sqrt(27*pow(c,12)*pow(f,2)+pow(c,12)*pow(f,4)),1/3)

def delta2(f,c): # Second repeated term of the equation
    return np.sqrt(pow(c,2) + 48*pow(c/f,2) + f*pow(c,4)/delta1(f,c) + delta1(f,c)/f)

def findEquilibrium(f,c): # Compute equilibrium point
    return -(4*c)/f - delta2(f,c)/np.sqrt(3) + 0.5*pow( (8/3)*pow(c,3) + 128*pow(c/f,2) - 4*pow(c,4)*f/(3*delta1(f,c)) - (4*delta1(f,c))/(3*f) - (np.sqrt(3)*(-4096*pow(c/f,3)-(128*pow(c,3))/f))/(8*delta2(f,c)),1/2)

def count_n(Nrm): # Count the number of charges in a single q2 = q_hole/2
    return 0.5*( 3 + np.sqrt( 3*(4*Nrm-1) ) )

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

def computeWsys(N,Nrm,R): # Compute system work from 1D formulation using d for separation between qhole and qNN
    qhole,qNN,_d_,_r_ = q_hole(Nrm),q_NN(Nrm),d_N(N,R),r_est(Nrm,N,R) # Compute system scale parameters
    f,c = qhole/qNN,_r_+_d_ # Compute f and c
    beta = findEquilibrium(f,c) # Compute equilibrium point
    Wsys = ((qNN*qhole)/(16*np.pi*ep))*( ((beta+_r_)/(_d_*(beta+_r_+_d_))) - ((beta+_r_)/((2*_r_+_d_)*(_r_+_d_-beta))) + f*((beta+_r_)/(4*beta*_r_)) ) # Compute work
    return Wsys

'''def computeWsys(N,Nrm,R): # Compute system work from 1D formulation using rNN - r for separation between qhole and qNN
    qhole,qNN,_rNN_,_r_ = q_hole(Nrm),q_NN(Nrm),r_NN(Nrm,N,R),r_est(Nrm,N,R) # Compute system scale parameters
    f,c = qhole/qNN,_rNN_ # Compute f and c
    _d_ = _rNN_ - _r_ # Difference between radii gives d
    beta = findEquilibrium(f,c) # Compute equilibrium point
    Wsys = ((qNN*qhole)/(16*np.pi*ep))*( ((beta+_r_)/(_d_*(beta+_r_+_d_))) - ((beta+_r_)/((2*_r_+_d_)*(_r_+_d_-beta))) + f*((beta+_r_)/(4*beta*_r_)) ) # Compute work
    return Wsys'''

maxFree = int(3) # Maximum number of free parameters allowed

if len(sys.argv) == 1:
    freeParams = int(1) # Number of free parameters of model to use
else:
    freeParams = int(sys.argv[1])
    if freeParams <= 0 or freeParams > maxFree:
        print(f'\nNumber of model parameters must be greater than {maxFree}.\n')
        sys.exit()

if freeParams == int(3):

    def modelFitFunc(N,Nrm,R,alpha,beta,c): # Model with 3 free parameters to fit to data (c, alpha, beta)
        return c*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*pow(N,alpha)*pow(Nrm,beta)*(3+np.sqrt(3*(Nrm-1)))*(1/q)

    def logLikelihood(data,err,alpha,beta,c,N,Nrm,R): # Log likelihood
        return np.sum(pow((data-modelFitFunc(N,Nrm,R,alpha,beta,c))/err,2))

    lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],params[1],params[2],N_MESH,NRM_MESH,R) # Function that we will minimize

    def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
        #result = minimize(lnlikehood,x0=guess,method='BFGS')
        result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
        print(result)
        α = result.x[0]
        β = result.x[1]
        c = result.x[2]
        chi2 = logLikelihood(data,err,α,β,c,N,Nrm,R)
        return result.x,chi2

elif freeParams == int(2):

    '''def modelFitFunc(N,Nrm,R,alpha,beta): # Model with 2 free parameters to fit to data (alpha, beta)
        return (1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*pow(N,alpha)*pow(Nrm,beta)*(3+np.sqrt(3*(Nrm-1)))*(1/q)

    def logLikelihood(data,err,alpha,beta,N,Nrm,R): # Log likelihood
        return np.sum(pow((data-modelFitFunc(N,Nrm,R,alpha,beta))/err,2))

    lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],params[1],N_MESH,NRM_MESH,R) # Function that we will minimize

    def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
        #result = minimize(lnlikehood,x0=(-1,1,10),method='BFGS')
        result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
        print(result)
        α = result.x[0]
        β = result.x[1]
        chi2 = logLikelihood(data,err,α,β,N,Nrm,R)
        return result.x,chi2'''
    
    def modelFitFunc(N,Nrm,R,c,alpha): # Model with 2 free parameters to fit to data (alpha, beta)
        return c*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*pow(N,alpha)*Nrm*(3+np.sqrt(3*(Nrm-1)))*(1/q)

    def logLikelihood(data,err,c,alpha,N,Nrm,R): # Log likelihood
        return np.sum(pow((data-modelFitFunc(N,Nrm,R,c,alpha))/err,2))

    lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],params[1],N_MESH,NRM_MESH,R) # Function that we will minimize

    def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
        #result = minimize(lnlikehood,x0=(-1,1,10),method='BFGS')
        result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
        print(result)
        c = result.x[0]
        α = result.x[1]
        chi2 = logLikelihood(data,err,c,α,N,Nrm,R)
        return result.x,chi2

elif freeParams == int(1):

    #def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), constant d(N)
    #    return c*computeWsys(N,Nrm,R)*(1/q)

    #def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), d(N) modified by r_est, fixed q_hole
    #    return c*(1/(4*np.pi*ep))*(q_hole(Nrm)*q_NN(Nrm))*r_est(Nrm,N,R)*0.5*((1/pow(d_N(N,R),2)) + 1/pow(d_N(N,R) + r_est(Nrm,N,R),2))*(1/q)

    def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), d(N) modified by r_est, fixed q_hole
        #return c*(1/(4*np.pi*ep))*(1/pow(d_N(N,R),2))*(q_hole(Nrm)*q_NN(Nrm))*r_est(Nrm,N,R)*(1/q)
        return c*(1/(4*np.pi*ep))*(1/d_N(N,R))*(q_hole(Nrm)*Nrm*Z*q)*(1/q)
        #return c*(1/(4*np.pi*ep))*(1/r_est(Nrm,N,R))*(q_hole(Nrm)*q_NN(Nrm))*(1/q)

    #def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), constant d(N)
    #    return c*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*(Nrm)*(3+np.sqrt(3*(4*Nrm-1)))*(1/q)
    
    #def modelFitFunc(N,Nrm,R,c): # Model with 1 free parameters to fit to data (c), d(N) modified by r_est
    #    return c*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*(Nrm)*(3+np.sqrt(3*(Nrm-1)))*(r_est(Nrm))*(1/q)

    def logLikelihood(data,err,c,N,Nrm,R): # Log likelihood
        return np.sum(pow((data-modelFitFunc(N,Nrm,R,c))/err,2))

    lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],N_MESH,NRM_MESH,R) # Function that we will minimize

    def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
        #result = minimize(lnlikehood,x0=guess,method='BFGS')
        result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
        print(result)
        c = result.x[0]
        chi2 = logLikelihood(data,err,c,N,Nrm,R)
        return result.x,chi2
    
    '''def modelFitFunc(N,Nrm,R,α): # Model with 1 free parameters to fit to data (c)
        return (1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*(Nrm)*(3+np.sqrt(3*(Nrm-1)))*pow(N,α)*(1/q)

    def logLikelihood(data,err,α,N,Nrm,R): # Log likelihood
        return np.sum(pow((data-modelFitFunc(N,Nrm,R,α))/err,2))

    lnlikehood = lambda params: logLikelihood(avg.T,err.T,params[0],N_MESH,NRM_MESH,R) # Function that we will minimize

    def minimizeLogLikelihood(data,err,N,Nrm,R,guess): # Compute best estimators for fit
        #result = minimize(lnlikehood,x0=guess,method='BFGS')
        result = minimize(lnlikehood,x0=guess,method='Nelder-Mead',options={'maxiter':5000, 'maxfev':5000})
        print(result)
        α = result.x[0]
        chi2 = logLikelihood(data,err,α,N,Nrm,R)
        return result.x,chi2'''


# ---------- Analyical Models ----------

def modelFunc(N,Nrm,R): # Model func with length scale d(N) --> r(N,Nrm) = f(Nrm)d(N). Derived using circle area approximation
    return 1*(1/(4*np.pi*ep))*(1/pow(d_N(N,R),2))*(q_hole(Nrm)*q_NN(Nrm))*r_est(Nrm,N,R)*(1/Nrm)*(1/q)

#def modelFunc(N,Nrm,R): # Model func with length scale d(N) --> r(N,Nrm) = f(Nrm)d(N). Derived using circle area approximation
#    return (4e-3)*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*(Nrm)*(3+np.sqrt(3*(Nrm-1)))*(1/r_est(Nrm))*(1/q)

#def modelFunc(N,Nrm,R): # Version with constant d(N) sing scaling constant c ~ 1/30
#    return (1/30)*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*(Nrm)*(3+np.sqrt(3*(Nrm-1)))*(1/q)

#def modelFunc(N,Nrm,R): # Model for testing new relationships. Currently trying d(N) --> d(N - Nrm)
#    return (1/30)*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos(((N-Nrm)*np.pi)/(3*((N-Nrm)-2)))) / (2*(1 - 2*np.cos(((N-Nrm)*np.pi)/(3*((N-Nrm)-2))))) )*(Nrm)*(3+np.sqrt(3*(Nrm-1)))*(1/q)

#def modelFunc(N,Nrm,R,c=1): # Previous form of derived model, removing factor of 1/N
#    return 1/c*(1/(4*np.pi*ep*R))*pow(Z*q,2)*np.sqrt( (1 - np.cos((N*np.pi)/(3*(N-2)))) / (2*(1 - 2*np.cos((N*np.pi)/(3*(N-2))))) )*Nrm*(2/(Nrm+1))*(3+np.sqrt(3*(Nrm-1)))*(1/q)


def chi2Model(data,err,N,Nrm,R): # Chi^2 of analytical model
    return np.sum(pow((data-modelFunc(N,Nrm,R))/err,2))

# ---------- Plotting and Averaging ----------

def computeAvgs(avg,err,Narr,numN,NMC,Nblocks,Nlines): # NEED TO FIX ******************************************
    for i in np.linspace(0,numN,numN+1,dtype='int'):
        for j in np.linspace(0,NMC-1,NMC,dtype='int'):
            fname = dir + f'N_{Narr[i]}_lattice_{j+1}_scaling.txt'
            temp = pd.read_csv(fname,delimiter=',',engine='python',header=None).to_numpy() # Open CSVs
            nrmArr = temp[0:Nrd*Nblocks,1].astype('int')
            sortNrm = np.argsort(nrmArr)
            data[i,j,:] = temp[sortNrm,2] # Data Array Format: N,NMC,Nrm
        for k in np.linspace(0,Nrd-1,Nrd,dtype='int'):
            avg[i,k] = np.average(data[i,:,k*Nblocks:(1+k)*Nblocks])
            err[i,k] = np.std(data[i,:,k*Nblocks:(1+k)*Nblocks])
    
def plotSingle(Nrm,Tmax_sim,Tmax_scaling,err,N,save=False): # Plot single scaling
    plt.figure(figsize=(12,10))
    #plt.figure(figsize=(10,8))
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
    #plt.rcParams.update({'font.size': 18})
    plt.title(r'$T(N,N_{rm})$ Curves of Constant $N$')
    plt.xlabel(r'$N_{rm}$')
    plt.ylabel('Energy (eV)')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xticks(Nrm)
    for i, c in zip(np.linspace(1,len(N),len(N),dtype='int'),colors):
        #plt.scatter(Nrm,Tmax_sim[i-1,:],label=f'N = {N[i-1]} Simulation',color=c)
        plt.scatter(Nrm,Tmax_sim[i-1,:],color=c)
        plt.errorbar(Nrm,Tmax_sim[i-1,:],yerr=err[i-1,:],xerr=None,ls='none',c=c)
        if plotScale:
            #plt.plot(Nrm,Tmax_scaling[i-1,:],label=f'N = {N[i-1]} Scaling',c=c)
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

#plotScalingSurface(R)

if len(sys.argv) == 1 or len(sys.argv) == 2:
    fit = True # Flag to determine whether code should fit data or use analytical model
else:
    fit = int(sys.argv[2])
plotScale = True # Plot scaling relation
save = False # Save figure

if freeParams == int(3): # Set fit initial guess
    guess = (1,1,1) 
elif freeParams == int(2):
    guess = (1,1)
elif freeParams == int(1):
    guess = (1)

computeAvgs(avg,err,Narr,numN,NMC,Nblocks,Nlines)

if fit:
    print(f'Fitting model.')
    params,chi2 = minimizeLogLikelihood(avg.T,err.T,N_MESH,NRM_MESH,R,guess)
    print(params)
    print(f'Reduced χ2 = {chi2/(N_MESH.shape[0]*N_MESH.shape[1] - freeParams)}')
    model = modelFitFunc(N_MESH,NRM_MESH,R,*params)
else:
    print(f'Not fitting, using pre-defined model.')
    model = modelFunc(N_MESH,NRM_MESH,R)
    chi2 = chi2Model(avg.T,err.T,N_MESH,NRM_MESH,R)
    print(f'Reduced χ2 = {chi2/(N_MESH.shape[0]*N_MESH.shape[1])}')

plotFittingSurface(N_MESH,NRM_MESH,model,avg,err)
plotAll(Narr,Nrm,avg,model.T,err,plotScale,save)
#for i in np.linspace(0,numN,numN+1,dtype='int'):
#    plotSingle(NRM_MESH[:,i],avg[i,:],model[:,i],err[i,:],N_MESH[0,i],False)