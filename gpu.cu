#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

struct ParticleData {
    particle_t* particles;      // Pointer to the particle data
    int* bin_indices;           // bin index for each particle
    int* bin_counts;            // Number of particles in each bin
    int* bin_scan;              // Prefix sum of bin_counts
    int* particle_bins;         // contiguous particle indices, organized by bin
};

__device__ double bin_size; 
__device__ int num_bins_x, num_bins_y;

__global__ void assign_bins_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = static_cast<int>(particles[tid].x / bin_size);  
    int bin_y = static_cast<int>(particles[tid].y / bin_size);  

    // Clamp bin indices to be within valid range
    bin_x = max(0, min(bin_x, num_bins_x - 1));  
    bin_y = max(0, min(bin_y, num_bins_y - 1));  

    int bin_index = bin_y * num_bins_x + bin_x;  
    bin_indices[tid] = bin_index;

    // Use atomicAdd for thread-safe increment
    atomicAdd(&bin_counts[bin_index], 1);
}

// Kernel for parallel prefix sum (scan)
__global__ void scan_gpu(int* bin_counts, int* bin_scan, int num_bins) {
    extern __shared__ int temp[]; // Shared memory for partial sums

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid; // Global ID within all blocks
    
    int offset = 1;

    // Load input into shared memory.  We load TWO elements per thread.
    temp[2 * tid] = (gid < num_bins) ? bin_counts[gid] : 0;
    temp[2 * tid + 1] = (gid + blockDim.x < num_bins) ? bin_counts[gid + blockDim.x] : 0;
    __syncthreads();  // Ensure all loads are complete

    // Up-sweep (Reduction) Phase
    for (int d = blockDim.x; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads(); // Ensure all threads update memory
    }

    // Set the last element to 0 for exclusive scan
    if (tid == 0) temp[2 * blockDim.x - 1] = 0;
    __syncthreads();

    // Down-sweep Phase
    for (int d = 1; d <= blockDim.x; d *= 2) {
        offset >>= 1;  // Corrected: Divide offset *before* the loop
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write results to global memory (exclusive scan)
    if (gid < num_bins) bin_scan[gid] = temp[2 * tid];
    if (gid + blockDim.x < num_bins) bin_scan[gid + blockDim.x] = temp[2 * tid + 1];
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    // very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    atomicAdd(&particle.ax, coef * dx); // Use atomicAdd for thread safety
    atomicAdd(&particle.ay, coef * dy); // Use atomicAdd for thread safety
}

__global__ void reorder_particles_gpu(int* particle_bins, int* bin_indices, int* bin_scan, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_index = bin_indices[tid];       // Get the bin index for this particle
	
    int offset = atomicAdd(&bin_scan[bin_index], 1);
    particle_bins[offset] = tid; // Store the particle ID in the compacted array
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_scan, int* particle_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particles[tid].ax = 0.0; // Reset accelerations
    particles[tid].ay = 0.0;

    int bin_index = bin_indices[tid];          // Get the bin index for this particle

    //Exclusive scan needs a -1 offset
    int bin_start = (bin_index == 0) ? 0: bin_scan[bin_index-1];        // Where does my bin start in particle_bins?
    int bin_end   = bin_scan[bin_index]; // Where does my bin end in particle_bins?

    // Iterate over particles in the same bin
    for (int i = bin_start; i < bin_end; ++i) {
        int other_particle_index = particle_bins[i];
        if (tid != other_particle_index) {    // Don't compute force with self
            apply_force_gpu(particles[tid], particles[other_particle_index]);
        }
    }

    // Calculate the bin indices of neighboring bins
    int bin_x = bin_index % num_bins_x;
    int bin_y = bin_index / num_bins_x;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;  // Skip the current bin, already handled

            int neighbor_bin_x = bin_x + dx;
            int neighbor_bin_y = bin_y + dy;

            // Check if neighbor bin indices are within valid range
            if (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
                neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_y) {

                int neighbor_bin_index = neighbor_bin_y * num_bins_x + neighbor_bin_x;
                int neighbor_bin_start =  (neighbor_bin_index == 0) ? 0 : bin_scan[neighbor_bin_index-1];        // Where does my bin start in particle_bins?
                int neighbor_bin_end = bin_scan[neighbor_bin_index]; // Where does my bin end in particle_bins?

                // Iterate over particles in the neighboring bin
                for (int i = neighbor_bin_start; i < neighbor_bin_end; ++i) {
                    int other_particle_index = particle_bins[i];
                    apply_force_gpu(particles[tid], particles[other_particle_index]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];

    // slightly simplified Velocity Verlet integration
    // conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // bounce from walls
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // 1. Calculate values on the HOST:
    double host_bin_size = cutoff;
    int host_num_bins_x = static_cast<int>(size / host_bin_size) + 1;
    int host_num_bins_y = static_cast<int>(size / host_bin_size) + 1;
    int num_bins = host_num_bins_x * host_num_bins_y;

    // 2. Copy HOST values to DEVICE variables:
    cudaMemcpyToSymbol(::bin_size, &host_bin_size, sizeof(double));
    cudaMemcpyToSymbol(::num_bins_x, &host_num_bins_x, sizeof(int));
    cudaMemcpyToSymbol(::num_bins_y, &host_num_bins_y, sizeof(int));

    // Allocate device memory for our data structures
    ParticleData pd; // Create a local ParticleData object
    cudaMalloc(&pd.bin_indices, num_parts * sizeof(int));
    cudaMalloc(&pd.bin_counts, num_bins * sizeof(int));
    cudaMalloc(&pd.bin_scan, num_bins * sizeof(int));
    cudaMalloc(&pd.particle_bins, num_parts * sizeof(int));

    // Set the particles pointer
    pd.particles = parts;

    // Copy bin parameters to device-accessible memory
    cudaMemcpyToSymbol(::bin_size, &bin_size, sizeof(double));
    cudaMemcpyToSymbol(::num_bins_x, &num_bins_x, sizeof(int));
    cudaMemcpyToSymbol(::num_bins_y, &num_bins_y, sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Use the HOST variables, which were initialized in init_simulation
    int host_num_bins_x;
    int host_num_bins_y;
    cudaMemcpyFromSymbol(&host_num_bins_x, ::num_bins_x, sizeof(int));
    cudaMemcpyFromSymbol(&host_num_bins_y, ::num_bins_y, sizeof(int));

    int num_bins = host_num_bins_x * host_num_bins_y;
	
	//Allocate memory at every step
    int* d_bin_indices;
    int* d_bin_counts;
    int* d_bin_scan;
    int* d_particle_bins;
    cudaMalloc(&d_bin_indices, num_parts * sizeof(int));
    cudaMalloc(&d_bin_counts, num_bins * sizeof(int));
    cudaMalloc(&d_bin_scan, num_bins * sizeof(int));
    cudaMalloc(&d_particle_bins, num_parts* sizeof(int));

    // Reset bin counts
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));

    // 1. Assign particles to bins and count particles in each bin.
    assign_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_counts);
    cudaDeviceSynchronize(); // Make sure bin assignment is complete

    // 2. Perform an exclusive prefix sum (scan) on the bin_counts.
    int sharedMemSize = 2 * NUM_THREADS * sizeof(int); // Shared memory for scan
    int scanBlks = (num_bins + NUM_THREADS -1) / NUM_THREADS;

    scan_gpu<<<scanBlks, NUM_THREADS, sharedMemSize>>>(d_bin_counts, d_bin_scan, num_bins);
    cudaDeviceSynchronize(); // Ensure scan is finished

    //3. Reorder partilces according the exclusive scan
    reorder_particles_gpu<<<blks, NUM_THREADS>>>(d_particle_bins, d_bin_indices, d_bin_scan, num_parts);
    cudaDeviceSynchronize();

    // 4. Compute forces.
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_scan, d_particle_bins);
    cudaDeviceSynchronize(); // Wait for force calculations

    // 5. Move particles.
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize(); // Wait for movement to finish
	
	//Free allocated memory at every step.
    cudaFree(d_bin_indices);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_scan);
    cudaFree(d_particle_bins);
}