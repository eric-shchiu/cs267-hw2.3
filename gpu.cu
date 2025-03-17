#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

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

__global__ void compute_forces_gpu(particle_t* d_parts, int num_parts, int* d_part_ids, int* d_bin_starts, int nbinsx, double dxbin) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    int i = d_part_ids[tid];
    int ib = (int)(d_parts[i].x / dxbin);
    int jb = (int)(d_parts[i].y / dxbin);

    d_parts[i].ax = d_parts[i].ay = 0;
    
    // The bin holding particle i is nbinsx*ib+jb
    // loop over particles in the neighboring bins to apply forces
    for (int local_i = max(0,ib-1); local_i <= min(nbinsx-1,ib+1); ++local_i){
        for (int local_j = max(0,jb-1); local_j <= min(nbinsx-1,jb+1); ++local_j){
            for (int j = d_bin_starts[nbinsx*local_i+local_j]; j < d_bin_starts[nbinsx*local_i+local_j+1]; j++)
                apply_force_gpu(d_parts[i], d_parts[d_part_ids[j]]);
        }
    }

}

__global__ void move_gpu(particle_t* d_parts, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &d_parts[tid];
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




static int nbinsx;
static int *d_part_ids  ; // array holding all the particles, sorted by bins
static int *d_bin_starts; // array holding the starting index of each bin in d_part_ids
static int *d_bin_idx;    // used as counter when populating bins
static double dxbin;      // length&width of each bin


void init_simulation(particle_t* d_parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // d_parts live in GPU memory
    // Do not do any particle simulation here

    nbinsx = ((int)((double) size / (cutoff)) + 1);  // bin size greater or equal to cutoff length
    dxbin = size / (double) nbinsx;

    cudaMalloc((void**)&d_part_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&d_bin_starts, (nbinsx*nbinsx) * sizeof(int));
    cudaMalloc((void**)&d_bin_idx, (nbinsx*nbinsx) * sizeof(int));
    
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}




__global__ void count_parts_gpu(particle_t* d_parts, int num_parts, int* d_bin_starts, int nbinsx, double dxbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int ib = (int)(d_parts[tid].x / (dxbin));
    int jb = (int)(d_parts[tid].y / (dxbin));
    atomicAdd(d_bin_starts+(nbinsx)*ib+jb,1);
}

__global__ void emplace_parts_gpu(particle_t* d_parts, int num_parts, int* d_bin_idx, int* d_part_ids, int nbinsx, double dxbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int ib = (int)(d_parts[tid].x / dxbin);
    int jb = (int)(d_parts[tid].y / dxbin);
    int idx = atomicAdd(d_bin_idx+nbinsx*ib+jb,1);
    d_part_ids[idx] = tid;
}


void calc_bins(particle_t* d_parts, int num_parts){
    /// Reset counter to 0 ///
    cudaMemset(d_bin_starts, 0, nbinsx*nbinsx*sizeof(int));

    /// Count number of d_parts per bin ///
    count_parts_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_starts, nbinsx, dxbin);

    /// Prefix sum the counts ///
    thrust::exclusive_scan(thrust::device,d_bin_starts,d_bin_starts+nbinsx*nbinsx,d_bin_starts);
    
    /// Copy bin_starts into bin_idx ///
    cudaMemcpy(d_bin_idx, d_bin_starts, (nbinsx*nbinsx) * sizeof(int), cudaMemcpyDeviceToDevice);

    /// emplace parts ///
    emplace_parts_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_idx,d_part_ids, nbinsx, dxbin);
}

void simulate_one_step(particle_t* d_parts, int num_parts, double size) {
    // d_parts live in GPU memory
    calc_bins(d_parts,num_parts);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_part_ids, d_bin_starts, nbinsx, dxbin);

    // Move d_parts
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
}
