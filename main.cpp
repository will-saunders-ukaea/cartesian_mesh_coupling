#include "particle_simulation.hpp"
#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <string>

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }

  if (argc > 2) {

    std::string argv1 = std::string(argv[1]);
    const int N_particles = std::stoi(argv1);

    std::string argv2 = std::string(argv[2]);
    const int N_steps = std::stoi(argv2);

    const int nx = 8;
    const int ny = nx;
    const double cell_extent = 1.0 / ((double)nx);
    const int subdivision_order = 0;

    auto particle_simulation = std::make_shared<ParticleSimulation>(
        N_particles, nx, ny, cell_extent, subdivision_order, MPI_COMM_WORLD);

    for (int stepx = 0; stepx < N_steps; stepx++) {

      particle_simulation->step();

      particle_simulation->write_particle_trajectory();

      double *mesh_values = particle_simulation->deposit_onto_mesh();

      const int local_cell_count = particle_simulation->get_cell_count();
      int global_tuple_index[2];
      for (int linear_cell_index = 0; linear_cell_index < local_cell_count;
           linear_cell_index++) {

        particle_simulation->map_local_linear_to_global_tuple(
            linear_cell_index, global_tuple_index);
        nprint("local linear cell index:", linear_cell_index,
               "global tuple index: (", global_tuple_index[0], ",",
               global_tuple_index[1],
               " ) quantity:", mesh_values[linear_cell_index]);
      }
    }

    particle_simulation->free();

  } else {
    std::cout << "Expected number of particles and number of steps to be "
                 "passed on command line."
              << std::endl;
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
