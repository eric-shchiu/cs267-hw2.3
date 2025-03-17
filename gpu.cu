#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Global variables
int blks;
__device__ double bin_size;
__device__ int num_bins_x, num_bins_y;

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

// Kernel to assign particles to bins, count particles, and reorder them in a single pass
__global__ void bin_particles_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_counts, int* bin_scan, int* particle_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = static_cast<int>(particles[tid].x / bin_size);
    int bin_y = static_cast<int>(particles[tid].y / bin_size);
    bin_x = max(0, min(bin_x, num_bins_x - 1));
    bin_y = max(0, min(bin_y, num_bins_y - 1));

    int bin_index = bin_y * num_bins_x + bin_x;
    bin_indices[tid] = bin_index;

    int offset = atomicAdd(&bin_counts[bin_index], 1);
    bin_scan[tid] = offset;
}

// Kernel to compute forces in parallel
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bin_indices, int* bin_scan, int* particle_bins) {
    __shared__ particle_t shared_particles[NUM_THREADS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particles[tid].ax = 0.0;
    particles[tid].ay = 0.0;

    int bin_index = bin_indices[tid];
    int bin_x = bin_index % num_bins_x;
    int bin_y = bin_index / num_bins_x;

    int num_neighbors = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int neighbor_bin_x = bin_x + dx;
            int neighbor_bin_y = bin_y + dy;

            if (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
                neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_y) {
                int neighbor_bin_index = neighbor_bin_y * num_bins_x + neighbor_bin_x;
                int bin_start = (neighbor_bin_index == 0) ? 0 : bin_scan[neighbor_bin_index - 1];
                int bin_end = bin_scan[neighbor_bin_index];

                for (int i = bin_start; i < bin_end; i += NUM_THREADS) {
                    int index = i + threadIdx.x;
                    if (index < bin_end) {
                        shared_particles[threadIdx.x] = particles[particle_bins[index]];
                    }
                    __syncthreads();

                    for (int j = 0; j < min(NUM_THREADS, bin_end - i); ++j) {
                        apply_force_gpu(particles[tid], shared_particles[j]);
                    }
                    __syncthreads();
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

// Initialization function
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

// Cleanup function
void cleanup_simulation() {
    cudaFree(d_bin_indices);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_scan);
    cudaFree(d_particle_bins);
}

// Simulation step function
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int host_num_bins_x, host_num_bins_y;
    cudaMemcpyFromSymbol(&host_num_bins_x, ::num_bins_x, sizeof(int));
    cudaMemcpyFromSymbol(&host_num_bins_y, ::num_bins_y, sizeof(int));
    int num_bins = host_num_bins_x * host_num_bins_y;

    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));

    bin_particles_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_counts, d_bin_scan, d_particle_bins);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> thrust_bin_counts(d_bin_counts);
    thrust::device_ptr<int> thrust_bin_scan(d_bin_scan);
    thrust::exclusive_scan(thrust_bin_counts, thrust_bin_counts + num_bins, thrust_bin_scan);

    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, d_bin_indices, d_bin_scan, d_particle_bins);
    cudaDeviceSynchronize();

    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
}
