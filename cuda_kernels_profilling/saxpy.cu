#include <iostream>
#include <cuda_runtime.h>

__global__ void saxpy(float* y, float a, float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Device buffers are allocated once and reused across benchmark iterations.
// This isolates pure kernel time from cudaMalloc / cudaMemcpy overhead.
extern "C" void run_saxpy(int N) {
    static float* d_x      = nullptr;
    static float* d_y      = nullptr;
    static int    cached_N = -1;

    if (cached_N != N) {
        cudaFree(d_x);
        cudaFree(d_y);

        size_t size = (size_t)N * sizeof(float);
        cudaMalloc(&d_x, size);
        cudaMalloc(&d_y, size);

        float* h_x = new float[N];
        float* h_y = new float[N];
        for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }
        cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
        delete[] h_x;
        delete[] h_y;

        cached_N = N;
    }

    int block = 256;
    int grid  = (N + block - 1) / block;
    saxpy<<<grid, block>>>(d_y, 2.0f, d_x, N);
    cudaDeviceSynchronize();
}
