#ifndef PARTICLE_SIMULATION_H_
#define PARTICLE_SIMULATION_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

class ParticleSimulation {
private:
  inline void add_particles() {
    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    std::srand(std::time(nullptr));
    int seed = std::rand();
    std::mt19937 rng(seed + rank);

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

        initial_distribution[Sym<REAL>("Q")][px][0] = 1.0;
        initial_distribution[Sym<INT>("ID")][px][0] = rstart + px;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }
    parallel_advection_initialisation(this->particle_group);
  }

  std::shared_ptr<BufferDeviceHost<double>> dh_mesh_values;

public:
  const int num_particles;
  const int ndim = 2;
  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  const double dt;

  std::shared_ptr<CartesianHMesh> mesh;
  SYCLTargetSharedPtr sycl_target;
  DomainSharedPtr domain;
  ParticleGroupSharedPtr particle_group;
  std::shared_ptr<CartesianPeriodic> boundary_conditions;
  std::shared_ptr<H5Part> h5part;

  ParticleSimulation(const int num_particles, MPI_Comm comm)
      : num_particles(num_particles), h5part(NULL), dt(0.01)

  {
    std::vector<int> dims(ndim);
    dims[0] = 8;
    dims[1] = 8;

    this->mesh = std::make_shared<CartesianHMesh>(comm, ndim, dims, cell_extent,
                                                  subdivision_order);

    this->sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

    this->domain = std::make_shared<Domain>(mesh);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<REAL>("V"), ndim),
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("ID"), 1)};

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    this->boundary_conditions = std::make_shared<CartesianPeriodic>(
        this->sycl_target, this->mesh, this->particle_group->position_dat);

    this->add_particles();

    this->dh_mesh_values = std::make_shared<BufferDeviceHost<double>>(
        this->sycl_target, this->domain->mesh->get_cell_count());
  }

  inline void write_particle_trajectory() {
    if (this->h5part == NULL) {
      this->h5part = std::make_shared<H5Part>(
          "particle_tracjectory.h5part", this->particle_group, Sym<REAL>("P"),
          Sym<INT>("NESO_MPI_RANK"), Sym<INT>("ID"), Sym<REAL>("Q"),
          Sym<REAL>("V"));
    }
    this->h5part->write();
  }

  inline int get_cell_count() { return this->domain->mesh->get_cell_count(); }

  inline void step() {
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

    this->boundary_conditions->execute();
    this->particle_group->hybrid_move();
  }

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

  inline void free() {
    this->particle_group->free();
    this->mesh->free();
    if (this->h5part != NULL) {
      this->h5part->close();
    }
  }
};

#endif
