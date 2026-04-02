#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

// Check every CUDA API call and stop immediately on failure.
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool verifyResult(const std::vector<float>& a,
                  const std::vector<float>& b,
                  const std::vector<float>& c) {
    for (size_t i = 0; i < c.size(); ++i) {
        float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << c[i]
                      << ", expected " << expected << std::endl;
            return false;
        }
    }
    return true;
}

// Run one benchmark case and return the elapsed kernel time in milliseconds.
float runVectorAddBenchmark(int n, int blockSize) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int gridSize = (n + blockSize - 1) / blockSize;

    // CUDA events are used for basic GPU-side timing.
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time between start and stop.
    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    if (!verifyResult(h_a, h_b, h_c)) {
        std::cerr << "Result verification failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Release CUDA events.
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Release device memory.
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return elapsedMs;
}

int main() {
    const std::vector<int> problemSizes = {
        1 << 18,
        1 << 20,
        1 << 22
    };
    const std::vector<int> blockSizes = {64, 128, 256, 512};

    std::cout << std::left
              << std::setw(14) << "Elements"
              << std::setw(12) << "BlockSize"
              << std::setw(14) << "Time(ms)"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (int n : problemSizes) {
        for (int blockSize : blockSizes) {
            float elapsedMs = runVectorAddBenchmark(n, blockSize);
            std::cout << std::left
                      << std::setw(14) << n
                      << std::setw(12) << blockSize
                      << std::setw(14) << std::fixed << std::setprecision(4) << elapsedMs
                      << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
