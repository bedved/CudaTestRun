#include <iostream>
#include <chrono>

// Include the CUDA runtime API
#include <cuda_runtime.h>

// Define the vector size
#define N 100000000

// Define the CPU function for adding two vectors
void add_cpu(long long* a, long long* b, long long* c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

// Define the CUDA kernel for adding two vectors
__global__ void add_gpu(long long* a, long long* b, long long* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    // Allocate memory for the vectors using cudaMallocManaged
    long long* a, * b, * c_cpu, * c_gpu;
    cudaMallocManaged(&a, N * sizeof(long long));
    cudaMallocManaged(&b, N * sizeof(long long));
    cudaMallocManaged(&c_cpu, N * sizeof(long long));
    cudaMallocManaged(&c_gpu, N * sizeof(long long));

    // Initialize the vectors with some values
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
        c_cpu[i] = 0;
        c_gpu[i] = 0;
    }

    // Measure the elapsed time for the CPU implementation
    auto start_time_cpu = std::chrono::high_resolution_clock::now();
    add_cpu(a, b, c_cpu);
    auto end_time_cpu = std::chrono::high_resolution_clock::now();
    auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_time_cpu - start_time_cpu).count();

    // Measure the elapsed time for the CUDA implementation
    auto start_time_gpu = std::chrono::high_resolution_clock::now();
    add_gpu << <1, N >> > (a, b, c_gpu);
    cudaDeviceSynchronize();
    auto end_time_gpu = std::chrono::high_resolution_clock::now();
    auto elapsed_time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_time_gpu - start_time_gpu).count();

    // Check that the CPU and GPU results are the same
    for (int i = 0; i < N; i++) {
        if (c_cpu[i] != c_gpu[i]) {
            std::cerr << "Error: CPU and GPU results differ at index " << i << std::endl;
            break;
        }
    }

    // Print the elapsed time for the CPU and GPU implementations
    std::cout << "Elapsed time (CPU): " << elapsed_time_cpu << " microseconds" << std::endl;
    std::cout << "Elapsed time (GPU): " << elapsed_time_gpu << " microseconds" << std::endl;

    // Free the memory for the vectors using cudaFree
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);

    return 0;
}
