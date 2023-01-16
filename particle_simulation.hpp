#ifndef PARTICLE_SIMULATION_H_
#define PARTICLE_SIMULATION_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

/**
 *  Particle simulation object that advects N particles over a cartesian mesh
 *  with peroidic boundary conditions. Particles carry a real valued quantity Q
 *  which is deposited cell wise onto a single node in the centre of the cell.
 */
class ParticleSimulation {
private:
  /**
   *  This function is called by the constructor and initialises the particle
   *  distribution with uniform random positions and Gaussian distributed
   *  velocities.
   */
  inline void add_particles() {
    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    // Distribute the creation of particles over the available MPI ranks.
    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    std::srand(std::time(nullptr));
    int seed = std::rand();
    std::mt19937 rng(seed + rank);

    // Uniform positions within the domain
    std::vector<std::vector<double>> positions =
        uniform_within_extents(N, this->ndim, this->mesh->global_extents, rng);

    std::normal_distribution<> velocity_distribution{0, 1};

    if (N > 0) {
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      for (int px = 0; px < N; px++) {
        initial_distribution[Sym<REAL>("P")][px][0] = positions[0][px];
        initial_distribution[Sym<REAL>("P")][px][1] = positions[1][px];

        initial_distribution[Sym<REAL>("V")][px][0] =
            velocity_distribution(rng);
        initial_distribution[Sym<REAL>("V")][px][1] =
            velocity_distribution(rng);

        initial_distribution[Sym<INT>("ID")][px][0] = rstart + px;

        // This Q value is the value on the particle which will be deposited
        // onto the node in each cell.
        initial_distribution[Sym<REAL>("Q")][px][0] = 1.0;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }

    // This moves the particles from the rank they were created on to the rank
    // that actually owns the point in space where they reside.
    parallel_advection_initialisation(this->particle_group);
  }

  // Helper object to store the mesh values on the compute device and the host
  // device and easily manage the copies between the two.
  std::shared_ptr<BufferDeviceHost<double>> dh_mesh_values;

public:
  // Total number of particles in the simulation.
  const int num_particles;
  // Number of spatial dimensions.
  const int ndim = 2;
  // x and y extent of the coarse cells that form the mesh (i.e. before any
  // subdivision).
  const double cell_extent;
  // Number of times each cell is subdivided (may be 0).
  const int subdivision_order;
  // Time step size.
  const double dt;
  // Number of coarse cells in the x direction.
  const int nx;
  // Number of coarse cells in the y direction.
  const int ny;

  // The object that stores the cartesian mesh itself.
  std::shared_ptr<CartesianHMesh> mesh;
  // A NESO-Particles object that encapsulates a SYCL device and queue.
  SYCLTargetSharedPtr sycl_target;
  // Helper instance to move particles between MPI ranks (unlikely to be useful
  // to a user directly).
  std::shared_ptr<CartesianHMeshLocalMapperT> global_cell_mapper;
  // Object that maps particle positions to cells locally.
  std::shared_ptr<CartesianCellBin> local_cell_mapper;
  // A computational domain that encapsulates a mesh and methods to map into
  // mesh cells.
  DomainSharedPtr domain;
  // A collection of particles that exist on a domain and a SYCL device.
  ParticleGroupSharedPtr particle_group;
  // The boundary condtion methods that are applied to the particles at each
  // time step.
  std::shared_ptr<CartesianPeriodic> boundary_conditions;
  // IO method that writes data to a h5part file (can be opened in paraview).
  std::shared_ptr<H5Part> h5part;

  /**
   * Create a new instance with a given number of particles and mesh size.
   *
   * @param num_particles The total number of particles accross all MPI ranks.
   * @param nx Number of coarse (i.e. before subdivision) mesh cells globally in
   * x (dimension 0).
   * @param ny Number of coarse (i.e. before subdivision) mesh cells globally in
   * y (dimension 1).
   * @param cell_extent The size of the coarse cells before subdivision in both
   * x and y (coarse cells are square).
   * @param dt Timestep size.
   * @param comm MPI communicator the mesh is decomposed over and particles
   * exist on.
   */
  ParticleSimulation(const int num_particles, const int nx, const int ny,
                     const double cell_extent, const int subdivision_order,
                     const double dt, MPI_Comm comm)
      : num_particles(num_particles), h5part(NULL), dt(dt), nx(nx), ny(ny),
        cell_extent(cell_extent), subdivision_order(subdivision_order)

  {

    // Create the requested Cartesian mesh.
    std::vector<int> dims(ndim);
    dims[0] = nx;
    dims[1] = ny;
    this->mesh = std::make_shared<CartesianHMesh>(comm, ndim, dims, cell_extent,
                                                  subdivision_order);

    // Create the NESO-Particles SYCL device.
    this->sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

    // Create the object that maps particles to mpi ranks.
    this->global_cell_mapper =
        CartesianHMeshLocalMapper(this->sycl_target, this->mesh);
    // Create the simulation domain for the particles.
    this->domain =
        std::make_shared<Domain>(this->mesh, this->global_cell_mapper);

    // This is where the particle properties are defined. These are user choice
    // (there must be positions and cells, see documentation).
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<REAL>("V"), ndim),
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("ID"), 1)};

    // Create the collection of particles - no particles exist in the
    // collection yet.
    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    // Create the object that maps positions to cells locally on this MPI rank.
    this->local_cell_mapper = std::make_shared<CartesianCellBin>(
        this->sycl_target, this->mesh, this->particle_group->position_dat,
        this->particle_group->cell_id_dat);

    // Create an object to apply periodic boundary conditions.
    this->boundary_conditions = std::make_shared<CartesianPeriodic>(
        this->sycl_target, this->mesh, this->particle_group->position_dat);

    // Create a distribution of particles, add them to the ParticleGroup and
    // send these particles to the correct MPI ranks for the positions they
    // hold.
    this->add_particles();

    // Create the helper data structure for representing mesh values on the
    // device and host.
    this->dh_mesh_values = std::make_shared<BufferDeviceHost<double>>(
        this->sycl_target, this->domain->mesh->get_cell_count());
  }

  /**
   * Write the current particle state to the trajectory for the particles. This
   * function is mainly indended for visualisation purposes. Output *.h5part
   * files should open directly in Paraview.
   */
  inline void write_particle_trajectory() {
    if (this->h5part == NULL) {
      this->h5part = std::make_shared<H5Part>(
          "particle_tracjectory.h5part", this->particle_group, Sym<REAL>("P"),
          Sym<INT>("NESO_MPI_RANK"), Sym<INT>("ID"), Sym<REAL>("Q"),
          Sym<REAL>("V"), Sym<INT>("CELL_ID"));
    }
    this->h5part->write();
  }

  /**
   *  Get the number of cells locally owned by this MPI rank.
   */
  inline int get_cell_count() { return this->domain->mesh->get_cell_count(); }

  /**
   *  Perform one step of moving particles forward in time with time step size
   *  dt. Automatically moves particles between MPI ranks as needed.
   */
  inline void step() {

    // Integrate positions forward in time.
    auto P = (*this->particle_group)[Sym<REAL>("P")];
    auto k_P = P->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim = this->ndim;
    const auto k_dt = this->dt;
    const auto pl_iter_range = P->get_particle_loop_iter_range();
    const auto pl_stride = P->get_particle_loop_cell_stride();
    const auto pl_npart_cell = P->get_particle_loop_npart_cell();
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  k_P[cellx][dimx][layerx] += k_V[cellx][dimx][layerx] * k_dt;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    // Apply boundary conditions.
    this->boundary_conditions->execute();
    // Move particles between ranks.
    this->particle_group->hybrid_move();
    // Map particles into cells.
    this->local_cell_mapper->execute();
    this->particle_group->cell_move();
  }

  /**
   *  Helper utility to convert a local linear index of a mesh cell to a global
   *  tuple index in the global mesh.
   *
   *  @param linear_index Local linear index in [0,
   * ParticleSimulation.get_cell_count()-1].
   *  @param tuple_index Pointer to int[2] array in which to place tuple index.
   *
   */
  inline void map_local_linear_to_global_tuple(const int linear_index,
                                               int *tuple_index) {
    NESOASSERT((linear_index >= 0) && (linear_index < this->get_cell_count()),
               "Invalid linear index. Should be in [0, get_cell_count()-1].");
    NESOASSERT(this->ndim == 2,
               "This conversion was written for 2 dimensions.");

    // linear_index = index_x + stride_x * index_y
    const int stride_x = this->mesh->cell_counts_local[0];

    // compute the local tuple index
    tuple_index[0] = linear_index % stride_x;
    tuple_index[1] = (linear_index - tuple_index[0]) / stride_x;

    // convert to a global tuple index
    tuple_index[0] += this->mesh->cell_starts[0];
    tuple_index[1] += this->mesh->cell_starts[1];
  }

  /**
   *  Loop over all particles and increment the nodal value (initialised to
   *  zero) in the centre of the cell they reside in with the value of Q stored
   *  on the particle.
   *
   *  @returns Pointer to host array of size
   *  ParticleSimulation.get_cell_count() which contains the nodal values. This
   *  pointer should not be freed by the user.
   */
  inline double *deposit_onto_mesh() {

    double *k_mesh_values = this->dh_mesh_values->d_buffer.ptr;
    const int k_cell_count = this->get_cell_count();

    // zero the values in the array that represents the mesh values
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(k_cell_count),
              [=](sycl::id<1> idx) { k_mesh_values[idx] = 0.0; });
        })
        .wait_and_throw();

    auto Q = (*this->particle_group)[Sym<REAL>("Q")];
    auto k_Q = Q->cell_dat.device_ptr();

    const auto pl_iter_range = Q->get_particle_loop_iter_range();
    const auto pl_stride = Q->get_particle_loop_cell_stride();
    const auto pl_npart_cell = Q->get_particle_loop_npart_cell();

    // increment the values on the mesh from the particles atomically
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                // read the value from the particle
                const REAL particle_quantity = k_Q[cellx][0][layerx];

                // atomically increment the mesh value
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    mesh_value_reference(k_mesh_values[cellx]);
                mesh_value_reference.fetch_add(particle_quantity);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    // copy the computed mesh values to the host
    this->dh_mesh_values->device_to_host();
    // return the pointer to the host values
    return this->dh_mesh_values->h_buffer.ptr;
  }

  /**
   *  Free the ParticleSimulation instance and any resources held. This must be
   *  called by the user collectively on the MPI communicator.
   */
  inline void free() {
    this->particle_group->free();
    this->mesh->free();
    if (this->h5part != NULL) {
      this->h5part->close();
    }
  }
};

#endif
