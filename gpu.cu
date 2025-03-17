#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int sqrt_n_bins;
double bin_len;
int n_bins;
int* part_ids;
int* bin_counts;
int* bin_prefix; 

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

// Compute forces on each particle in neighboring bins
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* part_ids, int* bin_counts, int sqrt_n_bins, double bin_len) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    // Get bin indices
    int pid = part_ids[tid];
    int bx = (int)particles[pid].x / bin_len;
    int by = (int)particles[pid].y / bin_len;

    // Zero acceleration of current particle
    particles[pid].ax = 0;
    particles[pid].ay = 0;

    // Loop over neighboring bins
    for (int nx = -1; nx <= 1; nx++) {
        for (int ny = -1; ny <= 1; ny++) {
            int nbx = bx + nx;
            int nby = by + ny;

            // Check if neighboring bin is within bounds
            if (nbx >= 0 && nbx < sqrt_n_bins && nby >= 0 && nby < sqrt_n_bins) {
                // Loop over particles in the neighboring bin
                for (int j = 0; j < bin_counts[nbx * sqrt_n_bins + nby]; j++) {
                    int nid = part_ids[j];
                    if (nid != pid) {
                        apply_force_gpu(particles[pid], particles[nid]);
                    }
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Calculate sqrt of number of bins and the length of each bin
    sqrt_n_bins = ceil(size / cutoff);
    bin_len = size / sqrt_n_bins;
    n_bins = std::pow(sqrt_n_bins, 2);

    // Allocate memory for arrays on CUDA device
    cudaMalloc((void**)&part_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_counts, n_bins * sizeof(int));
    cudaMalloc((void**)&bin_prefix, n_bins * sizeof(int));

    // Calculate number of blocks needed for the kernel functions
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

// Step 1: Count number of particles per bin
__global__ void count_parts(particle_t* parts, int num_parts, int sqrt_n_bins, double bin_len, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Get bin indices and update corresponding count
    int bx = (int)parts[tid].x / bin_len;
    int by = (int)parts[tid].y / bin_len;
    atomicAdd(bin_counts + bx * sqrt_n_bins + by, 1);
}

// Step 2: Prefix sum the bin counts
__global__ void prefix_sum(int num_parts, int sqrt_n_bins, int* bin_prefix, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Perform prefix sum from bin counts
    bin_prefix[0] = 0;
    for (int i = 1; i < sqrt_n_bins; i++) {
        bin_prefix[i] = bin_prefix[i - 1] + bin_counts[i - 1];
    }
}

// Step 3: Add particles to separate array starting from bin indices (sort particles by bin index)
__global__ void sort_parts(particle_t* parts, int num_parts, int sqrt_n_bins, double bin_len, int* bin_prefix, int* part_ids) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Get bin indices
    int bx = (int)parts[tid].x / bin_len;
    int by = (int)parts[tid].y / bin_len;

    // Get index to insert particle into sorted array
    int idx = atomicAdd(bin_prefix + bx * sqrt_n_bins + by, 1);
    part_ids[idx] = tid;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset bin counts to 0
    cudaMemset(bin_counts, 0, n_bins * sizeof(int));

    // Step 1: Count number of particles per bin
    count_parts<<<blks, NUM_THREADS>>>(parts, num_parts, sqrt_n_bins, bin_len, bin_counts);

    // Step 2: Prefix sum the bin counts
    prefix_sum<<<blks, NUM_THREADS>>>(num_parts, sqrt_n_bins, bin_prefix, bin_counts);

    // Step 3: Add particles to separate array starting from bin indices
    sort_parts<<<blks, NUM_THREADS>>>(parts, num_parts, sqrt_n_bins, bin_len, bin_prefix, part_ids);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, part_ids, bin_counts, sqrt_n_bins, bin_len);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
} 
