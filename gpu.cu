#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h> // Include for thrust::exclusive_scan

#define NUM_THREADS 256

// Global variable (host-side)
int blks;

// Device variables for binning
__device__ double bin_size;
__device__ int num_bins_x, num_bins_y;

// Device memory pointers (now persistent)
int* d_bin_indices;
int* d_bin_counts;
int* d_bin_scan;
int* d_particle_bins;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Kernel to assign particles to bins and count particles per bin
__global__ void assign_bins_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = static_cast<int>(particles[tid].x / bin_size);
    int bin_y = static_cast<int>(particles[tid].y / bin_size);
    bin_x = max(0, min(bin_x, num_bins_x - 1));
    bin_y = max(0, min(bin_y, num_bins_y - 1));

    int bin_index = bin_y * num_bins_x + bin_x;
    bin_indices[tid] = bin_index;
    atomicAdd(&bin_counts[bin_index], 1);
}

// Kernel to reorder particle indices based on bin assignment
__global__ void reorder_particles_gpu(int* particle_bins, int* bin_indices, int* bin_scan, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_index = bin_indices[tid];
    int offset = atomicAdd(&bin_scan[bin_index], 1);
    particle_bins[offset] = tid;
}

// Kernel to compute forces using binning
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_scan, int* particle_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particles[tid].ax = 0.0;
    particles[tid].ay = 0.0;

    int bin_index = bin_indices[tid];
    int bin_start = (bin_index == 0) ? 0 : bin_scan[bin_index - 1];
    int bin_end = bin_scan[bin_index];

    for (int i = bin_start; i < bin_end; ++i) {
        int other_particle_index = particle_bins[i];
        if (tid != other_particle_index) {
            apply_force_gpu(particles[tid], particles[other_particle_index]);
        }
    }

    int bin_x = bin_index % num_bins_x;
    int bin_y = bin_index / num_bins_x;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;

            int neighbor_bin_x = bin_x + dx;
            int neighbor_bin_y = bin_y + dy;

            if (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
                neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_y) {
                int neighbor_bin_index = neighbor_bin_y * num_bins_x + neighbor_bin_x;
                int neighbor_bin_start = (neighbor_bin_index == 0) ? 0 : bin_scan[neighbor_bin_index - 1];
                int neighbor_bin_end = bin_scan[neighbor_bin_index];

                for (int i = neighbor_bin_start; i < neighbor_bin_end; ++i) {
                    int other_particle_index = particle_bins[i];
                    apply_force_gpu(particles[tid], particles[other_particle_index]);
                }
            }
        }
    }
}

// Kernel to move particles
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

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
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    double host_bin_size = cutoff;
    int host_num_bins_x = static_cast<int>(size / host_bin_size) + 1;
    int host_num_bins_y = static_cast<int>(size / host_bin_size) + 1;

    cudaMemcpyToSymbol(::bin_size, &host_bin_size, sizeof(double));
    cudaMemcpyToSymbol(::num_bins_x, &host_num_bins_x, sizeof(int));
    cudaMemcpyToSymbol(::num_bins_y, &host_num_bins_y, sizeof(int));

    int num_bins = host_num_bins_x * host_num_bins_y;
    
    cudaMalloc(&d_bin_indices, num_parts * sizeof(int));
    cudaMalloc(&d_bin_counts, num_bins * sizeof(int));
    cudaMalloc(&d_bin_scan, num_bins * sizeof(int));
    cudaMalloc(&d_particle_bins, num_parts * sizeof(int));
}

// Simulation step function
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int host_num_bins_x, host_num_bins_y;
    cudaMemcpyFromSymbol(&host_num_bins_x, ::num_bins_x, sizeof(int));
    cudaMemcpyFromSymbol(&host_num_bins_y, ::num_bins_y, sizeof(int));
    int num_bins = host_num_bins_x * host_num_bins_y;

    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));

    assign_bins_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_counts);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> thrust_bin_counts(d_bin_counts);
    thrust::device_ptr<int> thrust_bin_scan(d_bin_scan);
    thrust::exclusive_scan(thrust_bin_counts, thrust_bin_counts + num_bins, thrust_bin_scan);

    reorder_particles_gpu<<<blks, NUM_THREADS>>>(d_particle_bins, d_bin_indices, d_bin_scan, num_parts);
    cudaDeviceSynchronize();

    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_scan, d_particle_bins);
    cudaDeviceSynchronize();

    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
}
