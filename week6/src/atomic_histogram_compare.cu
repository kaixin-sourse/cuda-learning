#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
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

__global__ void globalAtomicHistogramKernel(const int* input, int* globalHist, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        int value = input[idx];
        if (value >= 0 && value < kNumBins) {
            atomicAdd(&globalHist[value], 1);
        }
    }
}

__global__ void sharedAtomicHistogramKernel(const int* input, int* globalHist, int n) {
    __shared__ int localHist[kNumBins];

    for (int bin = threadIdx.x; bin < kNumBins; bin += blockDim.x) {
        localHist[bin] = 0;
    }
    __syncthreads();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        int value = input[idx];
        if (value >= 0 && value < kNumBins) {
            atomicAdd(&localHist[value], 1);
        }
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < kNumBins; bin += blockDim.x) {
        atomicAdd(&globalHist[bin], localHist[bin]);
    }
}

enum class HistogramKernelKind {
    GlobalAtomic,
    SharedAtomic
};

std::vector<int> cpuHistogram(const std::vector<int>& input) {
    std::vector<int> hist(kNumBins, 0);
    for (int value : input) {
        ++hist[value];
    }
    return hist;
}

bool verifyHist(const std::vector<int>& expected, const std::vector<int>& actual) {
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

struct HistogramBenchmarkResult {
    std::vector<int> hist;
    double elapsedMs;
};

HistogramBenchmarkResult runGpuHistogram(const std::vector<int>& input, HistogramKernelKind kind) {
    auto start = std::chrono::high_resolution_clock::now();

    int* d_input = nullptr;
    int* d_hist = nullptr;
    int n = static_cast<int>(input.size());
    int gridSize = 128;

    CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist, static_cast<size_t>(kNumBins) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input,
                          input.data(),
                          static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hist, 0, static_cast<size_t>(kNumBins) * sizeof(int)));

    if (kind == HistogramKernelKind::GlobalAtomic) {
        globalAtomicHistogramKernel<<<gridSize, kBlockSize>>>(d_input, d_hist, n);
    } else {
        sharedAtomicHistogramKernel<<<gridSize, kBlockSize>>>(d_input, d_hist, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> hist(kNumBins, 0);
    CHECK_CUDA(cudaMemcpy(hist.data(),
                          d_hist,
                          static_cast<size_t>(kNumBins) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_hist));

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsedMs =
        std::chrono::duration<double, std::milli>(stop - start).count();

    return {hist, elapsedMs};
}

std::vector<int> makeSkewedInput(int n) {
    std::vector<int> input(n, 0);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> binDist(0, kNumBins - 1);
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);

    for (int& value : input) {
        // Skew the distribution so that contention is visible.
        if (probDist(rng) < 0.70f) {
            value = 0;
        } else {
            value = binDist(rng);
        }
    }
    return input;
}

int main() {
    const std::vector<int> problemSizes = {1 << 18, 1 << 20, 1 << 22};

    std::cout << std::left
              << std::setw(12) << "Elements"
              << std::setw(18) << "CPU(ms)"
              << std::setw(22) << "GPU Global(ms)"
              << std::setw(22) << "GPU Shared(ms)"
              << "Check" << std::endl;
    std::cout << std::string(86, '-') << std::endl;

    for (int n : problemSizes) {
        std::vector<int> input = makeSkewedInput(n);

        auto cpuStart = std::chrono::high_resolution_clock::now();
        std::vector<int> cpuHist = cpuHistogram(input);
        auto cpuStop = std::chrono::high_resolution_clock::now();
        double cpuMs =
            std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

        HistogramBenchmarkResult globalGpu =
            runGpuHistogram(input, HistogramKernelKind::GlobalAtomic);
        HistogramBenchmarkResult sharedGpu =
            runGpuHistogram(input, HistogramKernelKind::SharedAtomic);

        bool okGlobal = verifyHist(cpuHist, globalGpu.hist);
        bool okShared = verifyHist(cpuHist, sharedGpu.hist);
        bool ok = okGlobal && okShared;

        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(18) << std::fixed << std::setprecision(4) << cpuMs
                  << std::setw(22) << globalGpu.elapsedMs
                  << std::setw(22) << sharedGpu.elapsedMs
                  << (ok ? "PASS" : "FAIL")
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
