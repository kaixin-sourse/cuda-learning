#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
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

constexpr int kBlockSize = 256;
constexpr int kTopK = 8;
constexpr float kNegativeSentinel = -1.0e30f;

__global__ void blockTopKCandidatesKernel(const float* input, float* candidates, int n) {
    __shared__ float tile[kBlockSize];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (globalIdx < n) ? input[globalIdx] : kNegativeSentinel;
    __syncthreads();

    if (threadIdx.x == 0) {
        float localTopK[kTopK];
        for (int i = 0; i < kTopK; ++i) {
            localTopK[i] = kNegativeSentinel;
        }

        for (int i = 0; i < blockDim.x; ++i) {
            float value = tile[i];
            for (int pos = 0; pos < kTopK; ++pos) {
                if (value > localTopK[pos]) {
                    for (int shift = kTopK - 1; shift > pos; --shift) {
                        localTopK[shift] = localTopK[shift - 1];
                    }
                    localTopK[pos] = value;
                    break;
                }
            }
        }

        for (int i = 0; i < kTopK; ++i) {
            candidates[blockIdx.x * kTopK + i] = localTopK[i];
        }
    }
}

std::vector<float> cpuTopK(const std::vector<float>& input) {
    std::vector<float> copy = input;
    std::partial_sort(copy.begin(),
                      copy.begin() + kTopK,
                      copy.end(),
                      std::greater<float>());
    copy.resize(kTopK);
    return copy;
}

std::vector<float> mergeCandidatesTopK(std::vector<float> candidates) {
    std::partial_sort(candidates.begin(),
                      candidates.begin() + kTopK,
                      candidates.end(),
                      std::greater<float>());
    candidates.resize(kTopK);
    return candidates;
}

bool verifyTopK(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > 1e-5f) {
            std::cerr << "Mismatch at position " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

struct TopKBenchmarkResult {
    std::vector<float> topk;
    double elapsedMs;
};

TopKBenchmarkResult runGpuTopK(const std::vector<float>& input) {
    auto start = std::chrono::high_resolution_clock::now();

    int n = static_cast<int>(input.size());
    int gridSize = (n + kBlockSize - 1) / kBlockSize;
    size_t inputBytes = static_cast<size_t>(n) * sizeof(float);
    size_t candidatesBytes = static_cast<size_t>(gridSize) * kTopK * sizeof(float);

    float* d_input = nullptr;
    float* d_candidates = nullptr;
    std::vector<float> h_candidates(static_cast<size_t>(gridSize) * kTopK);

    CHECK_CUDA(cudaMalloc(&d_input, inputBytes));
    CHECK_CUDA(cudaMalloc(&d_candidates, candidatesBytes));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), inputBytes, cudaMemcpyHostToDevice));

    blockTopKCandidatesKernel<<<gridSize, kBlockSize>>>(d_input, d_candidates, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_candidates.data(),
                          d_candidates,
                          candidatesBytes,
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_candidates));

    std::vector<float> result = mergeCandidatesTopK(h_candidates);

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsedMs =
        std::chrono::duration<double, std::milli>(stop - start).count();

    return {result, elapsedMs};
}

std::vector<float> makeRandomInput(int n) {
    std::vector<float> input(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (float& value : input) {
        value = dist(rng);
    }
    return input;
}

int main() {
    const std::vector<int> problemSizes = {1 << 14, 1 << 18, 1 << 20};

    std::cout << std::left
              << std::setw(12) << "Elements"
              << std::setw(18) << "CPU(ms)"
              << std::setw(18) << "GPU(ms)"
              << "Check" << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    for (int n : problemSizes) {
        std::vector<float> input = makeRandomInput(n);

        auto cpuStart = std::chrono::high_resolution_clock::now();
        std::vector<float> cpuResult = cpuTopK(input);
        auto cpuStop = std::chrono::high_resolution_clock::now();
        double cpuMs =
            std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

        TopKBenchmarkResult gpuResult = runGpuTopK(input);
        bool ok = verifyTopK(cpuResult, gpuResult.topk);

        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(18) << std::fixed << std::setprecision(4) << cpuMs
                  << std::setw(18) << gpuResult.elapsedMs
                  << (ok ? "PASS" : "FAIL")
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
