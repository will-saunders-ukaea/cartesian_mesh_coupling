This is a proof of concept implementation for a use case that executes a copy of a particle simulation on each MPI rank.


## Dependencies

Requires:

* SYCL: Tested with hipsycl 0.9.2 and Intel DPCPP 2022.1.0
* MPI: Tested with Mpich 4.0 (ubuntu 22.04 build) and Intel MPI 2021.6
* HDF5: If particle trajectories are required - will execute without.


## Installing and executing

NESO-Particles should be placed on the `CMAKE_PREFIX_PATH` such that cmake `find_package` works.

```
git clone https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
```

Append `CMAKE_PREFIX_PATH`:

```
export CMAKE_PREFIX_PATH=<absolute path to NESO-Particles>:$CMAKE_PREFIX_PATH
```

Checkout this repository:

```
git clone https://github.com/will-saunders-ukaea/cartesian_mesh_coupling.git
cd cartesian_mesh_coupling
```

Configuring cmake depends on which SYCL implementation/target you wish to use:

```
# Intel DPCPP
cmake -DCMAKE_CXX_COMPILER=dpcpp -DNESO_PARTICLES_DEVICE_TYPE=CPU .
# Hipsycl cpu via omp and host compiler
cmake -DNESO_PARTICLES_DEVICE_TYPE=CPU -DHIPSYCL_TARGETS=omp . 
# Hipsycl cuda using nvcxx
cmake -DNESO_PARTICLES_DEVICE_TYPE=GPU -DHIPSYCL_TARGETS=cuda-nvcxx .
```

Finally executing can be done as follows

```
# Intel DPCPP
SYCL_DEVICE_FILTER=host mpirun -n <nproc> bin/particle_simulation <nparticles> <nsteps>

# hipsycl omp
OMP_NUM_THREADS=<nthreads> mpirun -n <nproc> bin/particle_simulation <nparticles> <nsteps>
```

