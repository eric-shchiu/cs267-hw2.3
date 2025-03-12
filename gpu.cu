#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h> 

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_bins_x, num_bins_y, num_bins; // 网格尺寸
double bin_size;       // Bin 尺寸

// GPU 设备内存
int* d_bin_counts;      // 每个 bin 内的粒子数
int* d_bin_prefix_sum;  // Bin 计数的前缀和
int* d_bin_particles;   // 按 bin 排序的粒子索引

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

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
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

__global__ void count_bins_gpu(particle_t* parts, int* bin_counts, int num_parts, double bin_size, int num_bins_x, int num_bins_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = (int)(parts[tid].x / bin_size);
    int bin_y = (int)(parts[tid].y / bin_size);
    int bin_id = bin_y * num_bins_x + bin_x;

    atomicAdd(&bin_counts[bin_id], 1);
}

__global__ void assign_bins_gpu(particle_t* parts, int* bin_counts, int* bin_prefix_sum, int* bin_particles, int num_parts, double bin_size, int num_bins_x, int num_bins_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = (int)(parts[tid].x / bin_size);
    int bin_y = (int)(parts[tid].y / bin_size);
    int bin_id = bin_y * num_bins_x + bin_x;

    int idx = atomicAdd(&bin_counts[bin_id], 1);
    bin_particles[bin_prefix_sum[bin_id] + idx] = tid;
}

__global__ void compute_forces_bin_gpu(particle_t* parts, int* bin_particles, int* bin_prefix_sum, int* bin_counts, int num_bins_x, int num_bins_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int bin_x = (int)(parts[tid].x / bin_size);
    int bin_y = (int)(parts[tid].y / bin_size);
    int bin_id = bin_y * num_bins_x + bin_x;

    parts[tid].ax = parts[tid].ay = 0;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = bin_x + dx;
            int ny = bin_y + dy;
            if (nx >= 0 && nx < num_bins_x && ny >= 0 && ny < num_bins_y) {
                int neighbor_bin_id = ny * num_bins_x + nx;
                int start_idx = bin_prefix_sum[neighbor_bin_id];
                int end_idx = start_idx + bin_counts[neighbor_bin_id];

                for (int i = start_idx; i < end_idx; i++) {
                    int pj = bin_particles[i];
                    apply_force_gpu(parts[tid], parts[pj]);
                }
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // 计算 Bin 的大小和数量
    bin_size = cutoff;
    num_bins_x = static_cast<int>(size / bin_size) + 1;
    num_bins_y = static_cast<int>(size / bin_size) + 1;
    num_bins = num_bins_x * num_bins_y;

    // 计算 GPU 线程块数
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // 在 GPU 上分配内存
    cudaMalloc((void**)&d_bin_counts, num_bins * sizeof(int));
    cudaMalloc((void**)&d_bin_prefix_sum, num_bins * sizeof(int));
    cudaMalloc((void**)&d_bin_particles, num_parts * sizeof(int));

    // 初始化 Bin 计数 & 前缀和
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));
    cudaMemset(d_bin_prefix_sum, 0, num_bins * sizeof(int));

    // 确保 GPU 设备的初始化正确
    cudaDeviceSynchronize();
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Step 1: 计算每个 bin 内粒子数
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));
    count_bins_gpu<<<blks, NUM_THREADS>>>(parts, d_bin_counts, num_parts, bin_size, num_bins_x, num_bins_y);
    cudaDeviceSynchronize();

    // Step 2: 计算前缀和
    thrust::inclusive_scan(thrust::device, d_bin_counts, d_bin_counts + num_bins, d_bin_prefix_sum);
    cudaDeviceSynchronize();

    // 复制 bin_counts，因为 assign_bins_gpu 还需要它
    cudaMemcpy(d_bin_prefix_sum, d_bin_counts, num_bins * sizeof(int), cudaMemcpyDeviceToDevice);

    // Step 3: 将粒子索引存入 bin
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));
    assign_bins_gpu<<<blks, NUM_THREADS>>>(parts, d_bin_counts, d_bin_prefix_sum, d_bin_particles, num_parts, bin_size, num_bins_x, num_bins_y);
    cudaDeviceSynchronize();

    // Step 4: 计算粒子间相互作用
    compute_forces_bin_gpu<<<blks, NUM_THREADS>>>(parts, d_bin_particles, d_bin_prefix_sum, d_bin_counts, num_bins_x, num_bins_y, num_parts);
    cudaDeviceSynchronize();

    // Step 5: 移动粒子
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

