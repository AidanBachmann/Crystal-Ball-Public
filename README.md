# Overview
This code was developed for the Crystal Ball (CB) model, detailed here: https://arxiv.org/abs/2410.04311. The data reported in this paper can be found in Data.zip. CB is an N-body code for simulating the dynamics of lattice structures composed of particles confined to the surface of a sphere which interact only through Coulomb coupling. The code has two primary functions: to generate lattices, and to study the dynamics of these lattices once formed. For $N$ particles of the same species, we are interested in the configuration
on the sphere which minimizes the Coulomb potential. This is known as the Thomson Problem (see here: https://en.wikipedia.org/wiki/Thomson_problem). Typically, global optimization methods are used to identify global minima candidates. In our case, we treat the problem as a dynamcis problem and push
the system to equilibrium using a first order integrator with an adaptive time step. A damping term is artificially imposed to ensure convergence. We find that a number of our lattices agree within 1 atomic unit of energy with candidates
identified by Wales et al. in their 2006 and 2009 papers, which is sufficient convergence for our purposes. With these lattices formed, we were particularly interested in phase transitions induced by removing particles from the lattice.
We aimed to characterize the average peak kinetic energy any one particle achieved as a function of $N$ and the number of particles removed $N_{rm}$. To this end, we remove $N_{rm}$ particles from our pre-formed lattices,
taking the system out of equilibrium. We then employ the same integrator and evolve the system without damping, tracking the time evolution of the particle kinetic energies, as well as system total energy to ensure sufficient energy conservation.

This method can be extended to Coulomb coupled, $N$ body systems on parametric surfaces.

# Dependencies
This code depends on numpy, pandas, and numba, so ensure that you have installed these dependencies before attempting to run the code. In particular, we use the numba just-in-time compiler to improve computation times.

# Usage

Coming soon.

# Code Optimization
This code is by no means optimal. It has been partially vectorized and it has been parallelized to the extent that it is possible to run identical simulations (with unique seeds) on an arbitary number of cores. The only limit is the number of
CPUs one has access to. Part of what makes the slow versions of both codes as slow as they are is indexing through huge arrays with $N_{t}\sim 10^{4}$ and $N\sim 10^{3}$. This could certainly be improved by only saving some subset of the time steps, for
instance 1 in every 100. The motivation for using the slow code is to generate energy time series plots, as well as spatial plots via the Mercator projection. As it stands,
time steps are so small that we require frame rates on the order of 200 fps to see any meaningful motion anyways. Thus, nothing is really lost by simply saving fewer data points. This change may be implemented if we continue working on the
project.

Additionally, it would be useful to add a checkpoint file system so that we can may initialize new simulations from the final (or intermediate) state of a previous simulation. This may be added in later versions. We may also
develop a C++ version, which would certainly improve computation times.

As it stands, a simulation (fast version of the code) with $N\sim 10^{3}$ and $N_{t}\sim 6\times 10^{4}$ takes around 7 hours to run locally on an M1 Max.
