#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(1); \
    }

__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Device buffers are allocated once and reused across benchmark iterations.
// This isolates pure kernel time from cudaMalloc / cudaMemcpy overhead.
extern "C" void run_matmul_naive(int N) {
    static float* d_A      = nullptr;
    static float* d_B      = nullptr;
    static float* d_C      = nullptr;
    static int    cached_N = -1;

    if (cached_N != N) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        size_t size = (size_t)N * N * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_A, size));
        CHECK_CUDA(cudaMalloc(&d_B, size));
        CHECK_CUDA(cudaMalloc(&d_C, size));

        float* h_A = new float[N * N];
        float* h_B = new float[N * N];
        for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }
        CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
        delete[] h_A;
        delete[] h_B;

        cached_N = N;
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
