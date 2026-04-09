#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
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

constexpr int kBlockSize = 256;
constexpr int kNumElements = 1 << 24;
constexpr int kRepeat = 100;

__global__ void reductionTwoLoadsKernel(const float* input, float* blockSums, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float value = 0.0f;
    if (globalIdx < static_cast<unsigned int>(n)) {
        value += input[globalIdx];
    }
    if (globalIdx + blockDim.x < static_cast<unsigned int>(n)) {
        value += input[globalIdx + blockDim.x];
    }

    shared[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

int main() {
    std::vector<float> h_input(kNumElements, 1.0f);
    int gridSize = (kNumElements + kBlockSize * 2 - 1) / (kBlockSize * 2);

    float* d_input = nullptr;
    float* d_blockSums = nullptr;
    std::vector<float> h_blockSums(gridSize, 0.0f);

    CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(kNumElements) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blockSums, static_cast<size_t>(gridSize) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          static_cast<size_t>(kNumElements) * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int iter = 0; iter < kRepeat; ++iter) {
        reductionTwoLoadsKernel<<<gridSize, kBlockSize, kBlockSize * sizeof(float)>>>(
            d_input, d_blockSums, kNumElements);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_blockSums.data(),
                          d_blockSums,
                          static_cast<size_t>(gridSize) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float total = std::accumulate(h_blockSums.begin(), h_blockSums.end(), 0.0f);
    std::cout << "Reduction repeats: " << kRepeat << std::endl;
    std::cout << "Partial block sums: " << gridSize << std::endl;
    std::cout << "Final accumulated sum: " << total << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));

    return EXIT_SUCCESS;
}
