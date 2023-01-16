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

    auto particle_simulation =
        std::make_shared<ParticleSimulation>(N_particles, MPI_COMM_WORLD);

    for (int stepx = 0; stepx < N_steps; stepx++) {

      particle_simulation->step();
      particle_simulation->write_particle_trajectory();

      double *mesh_values = particle_simulation->deposit_onto_mesh();
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
