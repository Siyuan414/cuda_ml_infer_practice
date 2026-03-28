#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <string>

// kernel runners (from your .cu files)
extern "C" void run_saxpy(int N);
extern "C" void run_matmul_naive(int N);
extern "C" void run_matmul_tiled(int N);
extern "C" void run_softmax(int N);

#define CHECK_CUDA(call)                                      \
    if ((call) != cudaSuccess) {                              \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(cudaGetLastError())   \
                  << std::endl;                               \
        exit(1);                                              \
    }

// ---------------- TIMER ----------------
float measure_ms(void (*fn)(int), int arg) {
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    fn(arg);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

// ---------------- BENCHMARK ----------------
float benchmark(void (*fn)(int), int arg, int iters = 10) {
    // warmup
    for (int i = 0; i < 3; i++) fn(arg);
    CHECK_CUDA(cudaDeviceSynchronize());

    float total = 0.0f;
    for (int i = 0; i < iters; i++) {
        total += measure_ms(fn, arg);
    }
    return total / iters;
}

// ---------------- MAIN ----------------
int main(int argc, char** argv) {
    std::cout << "=== CUDA Kernel Benchmark ===\n";

    int N_vec = 1 << 24;   // default for SAXPY / softmax
    int N_mat = 512;       // default for matmul

    if (argc > 1) {
        N_vec = std::stoi(argv[1]);
    }
    if (argc > 2) {
        N_mat = std::stoi(argv[2]);
    }

    std::cout << "Vector size: " << N_vec << "\n";
    std::cout << "Matrix size: " << N_mat << " x " << N_mat << "\n\n";

    // ---------------- SAXPY ----------------
    float t_saxpy = benchmark(run_saxpy, N_vec);
    double bytes = 3.0 * N_vec * sizeof(float); // x, y read + y write
    double bw = bytes / (t_saxpy / 1000.0) / 1e9;

    std::cout << "SAXPY:\n";
    std::cout << "  Time: " << t_saxpy << " ms\n";
    std::cout << "  Bandwidth: " << bw << " GB/s\n\n";

    // ---------------- MATMUL NAIVE ----------------
    float t_naive = benchmark(run_matmul_naive, N_mat);
    double flops_naive = 2.0 * N_mat * N_mat * N_mat;
    double gflops_naive = flops_naive / (t_naive / 1000.0) / 1e9;

    std::cout << "MatMul Naive:\n";
    std::cout << "  Time: " << t_naive << " ms\n";
    std::cout << "  Throughput: " << gflops_naive << " GFLOPS\n\n";

    // ---------------- MATMUL TILED ----------------
    float t_tiled = benchmark(run_matmul_tiled, N_mat);
    double flops_tiled = 2.0 * N_mat * N_mat * N_mat;
    double gflops_tiled = flops_tiled / (t_tiled / 1000.0) / 1e9;

    std::cout << "MatMul Tiled:\n";
    std::cout << "  Time: " << t_tiled << " ms\n";
    std::cout << "  Throughput: " << gflops_tiled << " GFLOPS\n";
    std::cout << "  Speedup vs Naive: " << t_naive / t_tiled << "x\n\n";

    // ---------------- SOFTMAX ----------------
    float t_softmax = benchmark(run_softmax, N_vec);
    double bytes_softmax = 2.0 * N_vec * sizeof(float); // read + write
    double bw_softmax = bytes_softmax / (t_softmax / 1000.0) / 1e9;

    std::cout << "Softmax:\n";
    std::cout << "  Time: " << t_softmax << " ms\n";
    std::cout << "  Effective Bandwidth: " << bw_softmax << " GB/s\n";

    return 0;
}