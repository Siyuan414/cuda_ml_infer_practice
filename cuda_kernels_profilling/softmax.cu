#include <cuda_runtime.h>
#include <cmath>

// ---------------- BLOCK SIZE ----------------
static const int BLOCK = 256;

// ── Reduction kernels ────────────────────────────────────────────────────────
//
// Each kernel collapses one level: every block of BLOCK threads writes a single
// value to out[blockIdx.x].  Calling three times drives an arbitrary N down to
// one scalar (handles N up to BLOCK^3 = ~16 M with BLOCK=256).

__global__ void reduce_max_k(const float* __restrict__ in, float* __restrict__ out, int N) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid]  = (i < N) ? in[i] : -1e38f;
    __syncthreads();
    for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
        if (tid < w) s[tid] = fmaxf(s[tid], s[tid + w]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduce_sum_k(const float* __restrict__ in, float* __restrict__ out, int N) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid]  = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
        if (tid < w) s[tid] += s[tid + w];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ── Element-wise kernels ─────────────────────────────────────────────────────

__global__ void exp_subtract(const float* __restrict__ x,
                             float* __restrict__ tmp,
                             float max_val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) tmp[i] = expf(x[i] - max_val);
}

__global__ void normalize(const float* __restrict__ tmp,
                          float* __restrict__ y,
                          float inv_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = tmp[i] * inv_sum;
}

// ── Two-level host helpers ───────────────────────────────────────────────────
//
// p1 holds ceil(N/BLOCK) partial values; p2 holds ceil(g1/BLOCK) values.
// A third kernel pass (single block) further reduces p2 → p2[0].

static float global_max(const float* d_in, float* d_p1, float* d_p2, int N) {
    int g1 = (N  + BLOCK - 1) / BLOCK;
    int g2 = (g1 + BLOCK - 1) / BLOCK;
    reduce_max_k<<<g1, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_p1, N);
    reduce_max_k<<<g2, BLOCK, BLOCK * sizeof(float)>>>(d_p1, d_p2, g1);
    if (g2 > 1)
        // g2 <= BLOCK for N <= BLOCK^3, so one block suffices
        reduce_max_k<<<1, BLOCK, BLOCK * sizeof(float)>>>(d_p2, d_p2, g2);
    float v;
    cudaMemcpy(&v, d_p2, sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

static float global_sum(const float* d_in, float* d_p1, float* d_p2, int N) {
    int g1 = (N  + BLOCK - 1) / BLOCK;
    int g2 = (g1 + BLOCK - 1) / BLOCK;
    reduce_sum_k<<<g1, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_p1, N);
    reduce_sum_k<<<g2, BLOCK, BLOCK * sizeof(float)>>>(d_p1, d_p2, g1);
    if (g2 > 1)
        reduce_sum_k<<<1, BLOCK, BLOCK * sizeof(float)>>>(d_p2, d_p2, g2);
    float v;
    cudaMemcpy(&v, d_p2, sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

// ── Host runner ──────────────────────────────────────────────────────────────
//
// Device buffers are allocated once and reused across benchmark iterations.
// This isolates pure kernel time from cudaMalloc / cudaMemcpy overhead.
//
// Algorithm (numerically stable global softmax):
//   1. Find global max  (3-level parallel reduction)
//   2. Compute exp(x - max) into tmp  (element-wise)
//   3. Find global sum of tmp  (3-level parallel reduction)
//   4. Divide tmp by sum into y  (element-wise)

extern "C" void run_softmax(int N) {
    static float* d_x      = nullptr;
    static float* d_y      = nullptr;
    static float* d_tmp    = nullptr;
    static float* d_p1     = nullptr;
    static float* d_p2     = nullptr;
    static int    cached_N = -1;

    if (cached_N != N) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_tmp);
        cudaFree(d_p1); cudaFree(d_p2);

        size_t bytes = (size_t)N * sizeof(float);
        int g1 = (N  + BLOCK - 1) / BLOCK;
        int g2 = (g1 + BLOCK - 1) / BLOCK;
        cudaMalloc(&d_x,   bytes);
        cudaMalloc(&d_y,   bytes);
        cudaMalloc(&d_tmp, bytes);
        cudaMalloc(&d_p1,  (size_t)g1 * sizeof(float));
        cudaMalloc(&d_p2,  (size_t)g2 * sizeof(float));

        float* h_x = new float[N];
        for (int i = 0; i < N; i++) h_x[i] = static_cast<float>(i % 100);
        cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
        delete[] h_x;

        cached_N = N;
    }

    int grid = (N + BLOCK - 1) / BLOCK;

    // Step 1: global max
    float max_val = global_max(d_x, d_p1, d_p2, N);

    // Step 2: exp(x - max) → d_tmp
    exp_subtract<<<grid, BLOCK>>>(d_x, d_tmp, max_val, N);

    // Step 3: global sum of exp values
    float sum_val = global_sum(d_tmp, d_p1, d_p2, N);

    // Step 4: normalize → d_y
    normalize<<<grid, BLOCK>>>(d_tmp, d_y, 1.0f / sum_val, N);

    cudaDeviceSynchronize();
}
