#include <cstdlib>
#include <cuda_runtime.h>
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
constexpr int kMatrixSize = 512;
constexpr int kRepeat = 20;

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

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main() {
    const size_t bytes = static_cast<size_t>(kMatrixSize) * kMatrixSize * sizeof(float);

    std::vector<float> h_a(kMatrixSize * kMatrixSize);
    std::vector<float> h_b(kMatrixSize * kMatrixSize);
    std::vector<float> h_c(kMatrixSize * kMatrixSize, 0.0f);

    for (int row = 0; row < kMatrixSize; ++row) {
        for (int col = 0; col < kMatrixSize; ++col) {
            h_a[row * kMatrixSize + col] = static_cast<float>((row + col) % 13);
            h_b[row * kMatrixSize + col] = static_cast<float>((row * 3 + col) % 17);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(kTileSize, kTileSize);
    dim3 grid((kMatrixSize + block.x - 1) / block.x,
              (kMatrixSize + block.y - 1) / block.y);

    for (int iter = 0; iter < kRepeat; ++iter) {
        tiledMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, kMatrixSize);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < 32; ++i) {
        checksum += h_c[i];
    }

    std::cout << "Matmul repeats: " << kRepeat << std::endl;
    std::cout << "Matrix size: " << kMatrixSize << " x " << kMatrixSize << std::endl;
    std::cout << "Checksum(first 32 values): " << checksum << std::endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return EXIT_SUCCESS;
}
