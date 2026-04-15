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

constexpr int kBlockSize = 8;

// Each block loads one contiguous chunk into shared memory and computes its sum.
__global__ void blockSumKernel(const float* input, float* blockSums, int n) {
    __shared__ float tile[kBlockSize];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < n) {
        tile[threadIdx.x] = input[globalIdx];
    } else {
        tile[threadIdx.x] = 0.0f;
    }

    // Make sure every thread has finished writing shared memory.
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += tile[i];
        }
        // 放到对应blockSums中
        blockSums[blockIdx.x] = sum;
        // 并没有求和
    }
}

int main() {
    const std::vector<float> h_input = {1, 2, 3, 4, 5, 6, 7, 8,
                                        9, 10, 11, 12, 13, 14, 15, 16};
    const int n = static_cast<int>(h_input.size());
    const int gridSize = (n + kBlockSize - 1) / kBlockSize;

    std::vector<float> h_blockSums(gridSize, 0.0f);

    float* d_input = nullptr;
    float* d_blockSums = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blockSums, gridSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    blockSumKernel<<<gridSize, kBlockSize>>>(d_input, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_blockSums.data(),
                          d_blockSums,
                          gridSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "Input values:" << std::endl;
    for (float value : h_input) {
        std::cout << std::setw(6) << value;
    }
    std::cout << "\n\nBlock sums:" << std::endl;
    for (int block = 0; block < gridSize; ++block) {
        std::cout << "Block " << block << ": " << h_blockSums[block] << std::endl;
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));

    return EXIT_SUCCESS;
}
