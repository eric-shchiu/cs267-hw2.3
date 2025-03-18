#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "globals.h"

#define NUM_THREADS 256

// Global variable (host-side) for the number of blocks
int blks;

// __device__ variables (accessible by all kernels)
__device__ double bin_size;
__device__ int num_bins_x, num_bins_y;

int* d_bin_indices = nullptr;
int* d_bin_counts = nullptr;
int* d_bin_scan = nullptr;
int* d_particle_bins = nullptr;

static int host_num_bins_x_cache = 0;
static int host_num_bins_y_cache = 0;

// Kernel to assign particles to bins and count particles per bin
__global__ void assign_bins_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Calculate bin coordinates using __device__ variables
    int bin_x = static_cast<int>(particles[tid].x / bin_size);
    int bin_y = static_cast<int>(particles[tid].y / bin_size);

    // Clamp bin indices to the valid range
    bin_x = max(0, min(bin_x, num_bins_x - 1));
    bin_y = max(0, min(bin_y, num_bins_y - 1));

    // Linearize the bin index
    int bin_index = bin_y * num_bins_x + bin_x;

    // Store the bin index for this particle
    bin_indices[tid] = bin_index;

    // Atomically increment the bin count (thread-safe)
    atomicAdd(&bin_counts[bin_index], 1);
}

// Device function to apply force between two particles
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;
    atomicAdd(&particle.ax, coef * dx); // Atomic update for thread safety
    atomicAdd(&particle.ay, coef * dy); // Atomic update for thread safety
}

// Kernel to reorder particle indices based on bin assignment
__global__ void reorder_particles_gpu(int* particle_bins, int* bin_indices, int* bin_scan, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_index = bin_indices[tid];
    int offset = atomicAdd(&bin_scan[bin_index], 1); // Atomic increment to get unique offset
    particle_bins[offset] = tid; // Store particle index
}

__device__ void apply_symmetric_force(particle_t& p1, particle_t& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx*dx + dy*dy;

    if (r2 > cutoff*cutoff) return;

    r2 = fmax(r2, min_r*min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff/r) / (r2 * mass);

    // Update the acceleration of both particles
    double ax = coef * dx;
    double ay = coef * dy;

    // apply force on both particles
    atomicAdd(&p1.ax, ax);
    atomicAdd(&p1.ay, ay);
    atomicAdd(&p2.ax, -ax);
    atomicAdd(&p2.ay, -ay);
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_scan, int* particle_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Reset accelerations
    particles[tid].ax = 0.0;
    particles[tid].ay = 0.0;

    int bin_index = bin_indices[tid];

    // Calculate start and end indices for the current bin (using exclusive prefix sum)
    int bin_start = (bin_index == 0) ? 0 : bin_scan[bin_index - 1];
    int bin_end = bin_scan[bin_index];

    // Iterate over particles within the current bin
    for (int i = bin_start; i < bin_end; ++i) {
        int other_particle_index = particle_bins[i];
        if (tid != other_particle_index) {
            apply_force_gpu(particles[tid], particles[other_particle_index]);
        }
    }

    // Calculate 2D bin coordinates from the linear bin index
    int bin_x = bin_index % num_bins_x;
    int bin_y = bin_index / num_bins_x;

    // Iterate over neighboring bins (right / down / right-down / left-down)
    const int neighbor_dirs[4][2] = {{1,0},  {0,1},  {1,1}, {-1,1}};

    for (int dir = 0; dir < 4; ++dir) {
        int dx = neighbor_dirs[dir][0];
        int dy = neighbor_dirs[dir][1];

        int neighbor_bin_x = bin_x + dx;
        int neighbor_bin_y = bin_y + dy;

        // Check if neighboring bin indices are within bounds
        if (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
            neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_y) {

            int neighbor_bin_index = neighbor_bin_y * num_bins_x + neighbor_bin_x;

            // Calculate start and end indices for the neighboring bin
            int neighbor_bin_start = (neighbor_bin_index == 0) ? 0 : bin_scan[neighbor_bin_index - 1];
            int neighbor_bin_end = bin_scan[neighbor_bin_index];

            // Iterate over particles in the neighboring bin
            for (int i = neighbor_bin_start; i < neighbor_bin_end; ++i) {
                int other_particle_index = particle_bins[i];
                apply_symmetric_force(particles[tid], particles[other_particle_index]);
            }
        }
    }
}

// Kernel to move particles based on calculated forces
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];

    // Velocity Verlet integration
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Bounce off walls
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// Initialization function (host-side)
void init_simulation(particle_t* parts, int num_parts, double size) {
    // calculate the number of blocks
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // calculate the bin parameters
    double host_bin_size = cutoff;
    int host_num_bins_x = static_cast<int>(size / host_bin_size) + 1;
    int host_num_bins_y = static_cast<int>(size / host_bin_size) + 1;

    // Cache calculation results to host variables
    host_num_bins_x_cache = host_num_bins_x;
    host_num_bins_y_cache = host_num_bins_y;

    // calculate the total number of bins
    const int num_bins = host_num_bins_x * host_num_bins_y;

    // copy to device
    cudaMemcpyToSymbol(bin_size, &host_bin_size, sizeof(double));
    cudaMemcpyToSymbol(num_bins_x, &host_num_bins_x, sizeof(int));
    cudaMemcpyToSymbol(num_bins_y, &host_num_bins_y, sizeof(int));

    // allocate new memory
    cudaMalloc(&d_bin_indices, num_parts * sizeof(int));
    cudaMalloc(&d_bin_counts, num_bins * sizeof(int));
    cudaMalloc(&d_bin_scan, num_bins * sizeof(int));
    cudaMalloc(&d_particle_bins, num_parts * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    const int num_bins = host_num_bins_x_cache * host_num_bins_y_cache;

    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float computation_time = 0.0f, synchronization_time = 0.0f;

    // Step 1: Reset the bin counts
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));

    // Assign Bins
    cudaEventRecord(start);
    assign_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    comp_time += computation_time / 1000.0; // transfer to seconds

    cudaEventRecord(start);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&synchronization_time, start, stop);
    sync_time += synchronization_time / 1000.0;

    // Prefix Sum
    cudaEventRecord(start);
    thrust::exclusive_scan(thrust::device_ptr<int>(d_bin_counts),
                           thrust::device_ptr<int>(d_bin_counts + num_bins),
                           thrust::device_ptr<int>(d_bin_scan));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    comp_time += computation_time / 1000.0;

    cudaEventRecord(start);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&synchronization_time, start, stop);
    sync_time += synchronization_time / 1000.0;

    // Reorder Particles
    cudaEventRecord(start);
    reorder_particles_gpu<<<blks, NUM_THREADS>>>(d_particle_bins, d_bin_indices, d_bin_scan, num_parts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    comp_time += computation_time / 1000.0;

    cudaEventRecord(start);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&synchronization_time, start, stop);
    sync_time += synchronization_time / 1000.0;

    // Compute Forces
    cudaEventRecord(start);
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_scan, d_particle_bins);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    comp_time += computation_time / 1000.0;

    cudaEventRecord(start);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&synchronization_time, start, stop);
    sync_time += synchronization_time / 1000.0;

    // Move Particles
    cudaEventRecord(start);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    comp_time += computation_time / 1000.0;

    cudaEventRecord(start);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&synchronization_time, start, stop);
    sync_time += synchronization_time / 1000.0;

    // release cuda event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
