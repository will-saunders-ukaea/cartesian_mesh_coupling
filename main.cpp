#include "particle_simulation.hpp"
#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <string>

/**
 * This main should be called with two arguments: The number of particles and
 * the number of time steps.
 */
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

    // The Cartesian mesh contains square cells and has nx coarse cells in
    // dimension 0 and ny cells in dimension 1. The mesh can be refined by
    // setting the subvision order which subdivides each mesh cell n times in
    // both dimensions.
    const int nx = 8;
    const int ny = nx;
    const int subdivision_order = 0;

    // Time step size for particle motion.
    const double dt = 0.01;

    // Sets the extent in both dimensions of each cell. Here we create a unit
    // square domain.
    const double cell_extent = 1.0 / ((double)nx);

    // Create a ParticleSimulation object which can perform basic: time
    // stepping, io and depositing to the mesh.
    auto particle_simulation = std::make_shared<ParticleSimulation>(
        N_particles, nx, ny, cell_extent, subdivision_order, dt,
        MPI_COMM_WORLD);

    // Run the simulation for N_steps
    for (int stepx = 0; stepx < N_steps; stepx++) {

      // Move the particles forward in time and perform boundary conditions and
      // interprocess particle communication.
      particle_simulation->step();

      // Optionally write this step to the particle trajectory.
      particle_simulation->write_particle_trajectory();

      // Deposit the particle values into a single node in the centre of each
      // cell. The returned pointer is to an array of size
      // ParticleSimulation.get_cell_count() which is the local number of mesh
      // cells on this MPI rank.
      double *mesh_values = particle_simulation->deposit_onto_mesh();

      // Get the number of mesh cells on this MPI rank. Local cells are indexed
      // linearly in [0, get_cell_count()-1]
      const int local_cell_count = particle_simulation->get_cell_count();

      // Space to store the global mesh index as a tuple.
      int global_tuple_index[2];

      // For each local cell.
      for (int linear_cell_index = 0; linear_cell_index < local_cell_count;
           linear_cell_index++) {
        // Get the global tuple index that this cell corresponds to.
        particle_simulation->map_local_linear_to_global_tuple(
            linear_cell_index, global_tuple_index);

        // Print the global cell index as a tuple and the value which was
        // deposited into that cell in the last call to deposit_onto_mesh.
        nprint("local linear cell index:", linear_cell_index,
               "global tuple index: (", global_tuple_index[0], ",",
               global_tuple_index[1],
               " ) quantity:", mesh_values[linear_cell_index]);
      }
    }

    // Free the ParticleSimulation object (must be called).
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
