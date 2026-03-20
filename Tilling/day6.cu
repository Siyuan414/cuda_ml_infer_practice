#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// --- Error Checking Macro ---
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ==========================================
// SAXPY
// ==========================================
__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void run_saxpy(int N) {
    float* x, * y;
    float* d_x, * d_y;
    size_t size = N * sizeof(float);

    x = (float*)malloc(size);
    y = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_x, size));
    CUDA_CHECK(cudaMalloc(&d_y, size));

    CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    saxpy << <grid, block >> > (N, 2.0f, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("SAXPY time: %f ms\n", ms);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(x);
    free(y);
}

// ==========================================
// MATRIX MULTIPLICATION
// ==========================================
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure we don't write outside the matrix
    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void run_matmul(int N) {
    size_t size = N * N * sizeof(float);
    float* A, * B, * C;
    float* d_A, * d_B, * d_C;

    // 1. Allocate Host Memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // 2. Initialize Host Matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // 3. Allocate Device Memory
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 4. Copy Data: Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // 5. Setup Grid and Block Dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // 6. Setup Timing Events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // 7. Launch Kernel
    matmul << <grid, block >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Catch launch errors

    // 8. Stop Timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Matmul time: %f ms\n", ms);

    // 9. Copy Result: Device -> Host (Crucial step added!)
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // 10. Cleanup Memory and Events
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(A);
    free(B);
    free(C);
}

//tailed matmul
#define TILE_SIZE 16
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // Load A tile
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tile_A[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tile_B[threadIdx.y][threadIdx.x] =
            B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

void run_tiled_matmul(int N) {
    size_t size = N * N * sizeof(float);

    float* A, * B, * C;
    float* d_A, * d_B, * d_C;

    // 1. Allocate Host Memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // 2. Initialize Matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // 3. Allocate Device Memory
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 4. Copy Data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // 5. Configure Grid & Block
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    // 6. Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // 7. Launch Kernel
    matmul_tiled << <grid, block >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Tiled Matmul time: %f ms\n", ms);

    // 8. Copy Result Back
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // 9. Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(A);
    free(B);
    free(C);
}
// 
// ==========================================
// MAIN ENTRY POINT
// ==========================================
int main() {
    int N_saxpy = 1 << 20;
    printf("--- Running SAXPY (N = %d) ---\n", N_saxpy);
    run_saxpy(N_saxpy);

    int N_matmul = 1024;
    printf("\n--- Running Matmul (N = %d x %d) ---\n", N_matmul, N_matmul);
    run_matmul(N_matmul);

    printf("\n--- Running Tiled Matmul ---\n");
    run_tiled_matmul(N_matmul);

    // Force clean teardown of the CUDA context
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}