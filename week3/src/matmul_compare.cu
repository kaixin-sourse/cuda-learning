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

constexpr int kTileSize = 16;

__global__ void naiveMatmulKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void tiledMatmulKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (n + kTileSize - 1) / kTileSize; ++tile) {
        int tiledColA = tile * kTileSize + threadIdx.x;
        int tiledRowB = tile * kTileSize + threadIdx.y;

        if (row < n && tiledColA < n) {
            tileA[threadIdx.y][threadIdx.x] = a[row * n + tiledColA];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRowB < n && col < n) {
            tileB[threadIdx.y][threadIdx.x] = b[tiledRowB * n + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Ensure the tile is fully loaded before using it.
        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Ensure all threads are done before overwriting shared memory.
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

bool verifyMatrices(const std::vector<float>& lhs,
                    const std::vector<float>& rhs,
                    float tolerance = 1e-3f) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": lhs = " << lhs[i]
                      << ", rhs = " << rhs[i] << std::endl;
            return false;
        }
    }
    return true;
}

float runKernelAndMeasure(void (*launch)(const float*, const float*, float*, int),
                          const float* d_a,
                          const float* d_b,
                          float* d_c,
                          int n) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    launch(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsedMs;
}

void launchNaive(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    naiveMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
}

void launchTiled(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    tiledMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
}

int main() {
    const int n = 256;
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    std::vector<float> h_a(n * n);
    std::vector<float> h_b(n * n);
    std::vector<float> h_cNaive(n * n, 0.0f);
    std::vector<float> h_cTiled(n * n, 0.0f);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            h_a[row * n + col] = static_cast<float>((row + col) % 7);
            h_b[row * n + col] = static_cast<float>((row * 2 + col) % 11);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_cNaive = nullptr;
    float* d_cTiled = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_cNaive, bytes));
    CHECK_CUDA(cudaMalloc(&d_cTiled, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Warm up once to reduce first-launch overhead noise.
    launchNaive(d_a, d_b, d_cNaive, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    launchTiled(d_a, d_b, d_cTiled, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float naiveMs = runKernelAndMeasure(launchNaive, d_a, d_b, d_cNaive, n);
    float tiledMs = runKernelAndMeasure(launchTiled, d_a, d_b, d_cTiled, n);

    CHECK_CUDA(cudaMemcpy(h_cNaive.data(), d_cNaive, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cTiled.data(), d_cTiled, bytes, cudaMemcpyDeviceToHost));

    bool ok = verifyMatrices(h_cNaive, h_cTiled);

    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Naive time (ms): " << std::fixed << std::setprecision(4) << naiveMs << std::endl;
    std::cout << "Tiled time (ms): " << std::fixed << std::setprecision(4) << tiledMs << std::endl;
    std::cout << "Verification: " << (ok ? "PASS" : "FAIL") << std::endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_cNaive));
    CHECK_CUDA(cudaFree(d_cTiled));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
