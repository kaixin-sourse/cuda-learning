#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
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

constexpr int kNumBins = 16;
constexpr int kBlockSize = 256;
constexpr int kNumElements = 1 << 20;
// clac the bin
__global__ void histogramKernel(const int* input, int* globalHist, int n) {
    __shared__ int localHist[kNumBins];

    for (int bin = threadIdx.x; bin < kNumBins; bin += blockDim.x) {
        localHist[bin] = 0;
    }
    __syncthreads();

    // Each thread processes a strided subset of the input.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        int value = input[idx];
        if (value >= 0 && value < kNumBins) {
            atomicAdd(&localHist[value], 1);
        }
    }
    __syncthreads();

    // Merge the block-local histogram into global memory.
    for (int bin = threadIdx.x; bin < kNumBins; bin += blockDim.x) {
        atomicAdd(&globalHist[bin], localHist[bin]);
    }
}

std::vector<int> cpuHistogram(const std::vector<int>& input) {
    std::vector<int> hist(kNumBins, 0);
    for (int value : input) {
        if (value >= 0 && value < kNumBins) {
            ++hist[value];
        }
    }
    return hist;
}

bool verify(const std::vector<int>& expected, const std::vector<int>& actual) {
    for (int bin = 0; bin < kNumBins; ++bin) {
        if (expected[bin] != actual[bin]) {
            std::cerr << "Mismatch at bin " << bin
                      << ": expected " << expected[bin]
                      << ", got " << actual[bin] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<int> h_input(kNumElements);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, kNumBins - 1);

    for (int& value : h_input) {
        value = dist(rng);
    }

    std::vector<int> h_reference = cpuHistogram(h_input);
    std::vector<int> h_gpuHist(kNumBins, 0);

    int* d_input = nullptr;
    int* d_hist = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(kNumElements) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist, static_cast<size_t>(kNumBins) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          static_cast<size_t>(kNumElements) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hist, 0, static_cast<size_t>(kNumBins) * sizeof(int)));

    int gridSize = 128;
    histogramKernel<<<gridSize, kBlockSize>>>(d_input, d_hist, kNumElements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_gpuHist.data(),
                          d_hist,
                          static_cast<size_t>(kNumBins) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool ok = verify(h_reference, h_gpuHist);

    std::cout << "Histogram verification: " << (ok ? "PASS" : "FAIL") << std::endl;
    for (int bin = 0; bin < kNumBins; ++bin) {
        std::cout << "bin " << bin
                  << ": CPU = " << h_reference[bin]
                  << ", GPU = " << h_gpuHist[bin]
                  << std::endl;
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_hist));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
