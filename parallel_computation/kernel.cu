#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ─────────────────────────────────────────────
// Warp and block primitives
// ─────────────────────────────────────────────

__inline__ __device__
float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__
float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : -FLT_MAX;
        val = warpReduceMax(val);
    }
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}

// ─────────────────────────────────────────────
// Naive softmax — single block, correct
// ─────────────────────────────────────────────

__global__
void softmax_naive(const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    // Single block only — tid == threadIdx.x, stride == blockDim.x
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += stride)
        max_val = fmaxf(max_val, input[i]);
    max_val = blockReduceMax(max_val);

    __shared__ float shared_max;
    if (tid == 0) shared_max = max_val;
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < N; i += stride)
        sum += expf(input[i] - shared_max);
    sum = blockReduceSum(sum);

    __shared__ float shared_sum;
    if (tid == 0) shared_sum = sum;
    __syncthreads();

    for (int i = tid; i < N; i += stride)
        output[i] = expf(input[i] - shared_max) / shared_sum;
}

// ─────────────────────────────────────────────
// Online softmax — single block, correct
// ─────────────────────────────────────────────

__global__
void softmax_online(const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    float m = -FLT_MAX, d = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        float m2 = fmaxf(m, x);
        d = d * expf(m - m2) + expf(x - m2);
        m = m2;
    }

    // Warp-level (m,d) merge
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mo = __shfl_down_sync(0xffffffff, m, offset);
        float do_ = __shfl_down_sync(0xffffffff, d, offset);
        float m2 = fmaxf(m, mo);
        d = d * expf(m - m2) + do_ * expf(mo - m2);
        m = m2;
    }

    __shared__ float sm[32], sd[32];
    if (lane == 0) { sm[wid] = m; sd[wid] = d; }
    __syncthreads();

    if (wid == 0) {
        m = (lane < num_warps) ? sm[lane] : -FLT_MAX;
        d = (lane < num_warps) ? sd[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float mo = __shfl_down_sync(0xffffffff, m, offset);
            float do_ = __shfl_down_sync(0xffffffff, d, offset);
            float m2 = fmaxf(m, mo);
            d = d * expf(m - m2) + do_ * expf(mo - m2);
            m = m2;
        }
        if (lane == 0) { sm[0] = m; sd[0] = d; }
    }
    __syncthreads();

    float global_max = sm[0], global_sum = sd[0];
    for (int i = tid; i < N; i += stride)
        output[i] = expf(input[i] - global_max) / global_sum;
}

// ─────────────────────────────────────────────
// Multi-block online softmax — two kernel passes
//
// Kernel 1: each block computes its local (m, d)
//           over its chunk and writes to scratch
// Kernel 2: one block merges all (m, d) pairs
//           from scratch, then normalizes output
// ─────────────────────────────────────────────

__global__
void softmax_multiblock_pass1(const float* __restrict__ input,
    float* __restrict__ partial_m,
    float* __restrict__ partial_d,
    int N)
{
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    float m = -FLT_MAX, d = 0.0f;
    for (int i = start; i < N; i += stride) {
        float x = input[i];
        float m2 = fmaxf(m, x);
        d = d * expf(m - m2) + expf(x - m2);
        m = m2;
    }

    // Warp-level merge
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mo = __shfl_down_sync(0xffffffff, m, offset);
        float do_ = __shfl_down_sync(0xffffffff, d, offset);
        float m2 = fmaxf(m, mo);
        d = d * expf(m - m2) + do_ * expf(mo - m2);
        m = m2;
    }

    __shared__ float sm[32], sd[32];
    if (lane == 0) { sm[wid] = m; sd[wid] = d; }
    __syncthreads();

    if (wid == 0) {
        m = (lane < num_warps) ? sm[lane] : -FLT_MAX;
        d = (lane < num_warps) ? sd[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float mo = __shfl_down_sync(0xffffffff, m, offset);
            float do_ = __shfl_down_sync(0xffffffff, d, offset);
            float m2 = fmaxf(m, mo);
            d = d * expf(m - m2) + do_ * expf(mo - m2);
            m = m2;
        }
        if (lane == 0) {
            partial_m[blockIdx.x] = m;
            partial_d[blockIdx.x] = d;
        }
    }
}

__global__
void softmax_multiblock_pass2(const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_d,
    int N, int num_blocks)
{
    // Single block merges partial results then normalizes
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread loads one or more partial (m,d) pairs
    float m = -FLT_MAX, d = 0.0f;
    for (int i = tid; i < num_blocks; i += stride) {
        float mi = partial_m[i];
        float di = partial_d[i];
        float m2 = fmaxf(m, mi);
        d = d * expf(m - m2) + di * expf(mi - m2);
        m = m2;
    }

    // Block-level merge of (m,d)
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mo = __shfl_down_sync(0xffffffff, m, offset);
        float do_ = __shfl_down_sync(0xffffffff, d, offset);
        float m2 = fmaxf(m, mo);
        d = d * expf(m - m2) + do_ * expf(mo - m2);
        m = m2;
    }
    __shared__ float sm[32], sd[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) { sm[wid] = m; sd[wid] = d; }
    __syncthreads();

    if (wid == 0) {
        m = (lane < num_warps) ? sm[lane] : -FLT_MAX;
        d = (lane < num_warps) ? sd[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float mo = __shfl_down_sync(0xffffffff, m, offset);
            float do_ = __shfl_down_sync(0xffffffff, d, offset);
            float m2 = fmaxf(m, mo);
            d = d * expf(m - m2) + do_ * expf(mo - m2);
            m = m2;
        }
        if (lane == 0) { sm[0] = m; sd[0] = d; }
    }
    __syncthreads();

    float global_max = sm[0], global_sum = sd[0];

    // Normalize — all N elements, this block grid-strides alone
    for (int i = tid; i < N; i += stride)
        output[i] = expf(input[i] - global_max) / global_sum;
}

// ─────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────

void softmax_cpu(const float* input, float* output, int N) {
    float mv = -FLT_MAX;
    for (int i = 0; i < N; i++) mv = fmaxf(mv, input[i]);
    float s = 0.0f;
    for (int i = 0; i < N; i++) s += expf(input[i] - mv);
    for (int i = 0; i < N; i++) output[i] = expf(input[i] - mv) / s;
}

float max_diff(const float* a, const float* b, int N) {
    float d = 0.0f;
    for (int i = 0; i < N; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main() {
    const int N = 1 << 22;   // 4M elements
    const int THREADS = BLOCK_SIZE;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int REPS = 100;

    float* h_in = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++)
        h_in[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;

    softmax_cpu(h_in, h_ref, N);

    float* d_in, * d_out, * d_pm, * d_pd;
    check(cudaMalloc(&d_in, N * sizeof(float)), "malloc d_in");
    check(cudaMalloc(&d_out, N * sizeof(float)), "malloc d_out");
    check(cudaMalloc(&d_pm, BLOCKS * sizeof(float)), "malloc partial_m");
    check(cudaMalloc(&d_pd, BLOCKS * sizeof(float)), "malloc partial_d");
    check(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // ── Single-block naive ─────────────────────
    softmax_naive << <1, THREADS >> > (d_in, d_out, N);  // warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++)
        softmax_naive << <1, THREADS >> > (d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    check(cudaGetLastError(), "naive");
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Naive   (1 block)  avg: %6.3f ms   max_err: %.2e\n",
        ms / REPS, max_diff(h_out, h_ref, N));

    // ── Single-block online ────────────────────
    softmax_online << <1, THREADS >> > (d_in, d_out, N);  // warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++)
        softmax_online << <1, THREADS >> > (d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    check(cudaGetLastError(), "online");
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Online  (1 block)  avg: %6.3f ms   max_err: %.2e\n",
        ms / REPS, max_diff(h_out, h_ref, N));

    // ── Multi-block online (two kernel passes) ─
    // warmup
    softmax_multiblock_pass1 << <BLOCKS, THREADS >> > (d_in, d_pm, d_pd, N);
    softmax_multiblock_pass2 << <1, THREADS >> > (d_in, d_out, d_pm, d_pd, N, BLOCKS);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < REPS; i++) {
        softmax_multiblock_pass1 << <BLOCKS, THREADS >> > (d_in, d_pm, d_pd, N);
        softmax_multiblock_pass2 << <1, THREADS >> > (d_in, d_out, d_pm, d_pd, N, BLOCKS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    check(cudaGetLastError(), "multiblock");
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Online  (multiblk) avg: %6.3f ms   max_err: %.2e\n",
        ms / REPS, max_diff(h_out, h_ref, N));

    printf("\nAll max_err should be < 1e-5\n");
    printf("Multi-block should be fastest — saturates memory bandwidth.\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_pm); cudaFree(d_pd);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}