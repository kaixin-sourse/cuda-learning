#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
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
constexpr int kNumElements = 1 << 22;

__global__ void reductionInterleavedKernel(const float* input, float* blockSums, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (globalIdx < static_cast<unsigned int>(n)) ? input[globalIdx] : 0.0f;
    // wait all threads done
    __syncthreads();

    // Educational baseline: interleaved addressing with modulo based branching.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        // well, Multithreading is truly amazing.
        // Sum each pair of adjacent elements with a fixed step size.
        // Tree Merging
        // But all threads must wait for the modulo operation and the breance evaluation to complete.
        if ((tid % (2 * stride)) == 0) {
            shared[tid] += shared[tid + stride];
        }
        // wait for all threads to complete the work for the
        // current synchronization step.
        __syncthreads();
    }
    // Thread 0 writes the block's partial sum to global memory.
    if (tid == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

__global__ void reductionSequentialKernel(const float* input, float* blockSums, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (globalIdx < static_cast<unsigned int>(n)) ? input[globalIdx] : 0.0f;
    __syncthreads();

    // Sequential addressing avoids the modulo branch in the baseline kernel.
    // Reverse order; avoid modulo operations.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Gradually reduce the length of the effective interval,
        // continuously retaining the left half.
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    // Thread 0 writes the block's partial sum to global memory.
    if (tid == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

__global__ void reductionTwoLoadsKernel(const float* input, float* blockSums, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    // 每个block处理2 * blockDim.x * 2个数据
    unsigned int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float value = 0.0f;
    // 每个线程处理globalIdx和globalIdx + blockDim.x这两个数据
    if (globalIdx < static_cast<unsigned int>(n)) {
        value += input[globalIdx];
    }
    if (globalIdx + blockDim.x < static_cast<unsigned int>(n)) {
        value += input[globalIdx + blockDim.x];
    }
    // 标记一下
    shared[tid] = value;
    // 每个线程进行到这里，进行同步
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
// “第三个 kernel 后面虽然和第二个一样，但前面已经把问题规模压缩了一次，因此后面这段相同代码处理的是更高密度的工作。”

using ReductionKernel = void (*)(const float*, float*, int);

struct BenchmarkResult {
    float gpuSum;
    float elapsedMs;
};

BenchmarkResult runReductionBenchmark(const std::vector<float>& h_input,
                                      int elementsPerThread,
                                      ReductionKernel kernel) {
    int n = static_cast<int>(h_input.size());
    int gridSize = (n + kBlockSize * elementsPerThread - 1)
                   / (kBlockSize * elementsPerThread);
    size_t inputBytes = static_cast<size_t>(n) * sizeof(float);
    size_t partialBytes = static_cast<size_t>(gridSize) * sizeof(float);
    size_t sharedBytes = static_cast<size_t>(kBlockSize) * sizeof(float);

    float* d_input = nullptr;
    float* d_blockSums = nullptr;
    std::vector<float> h_blockSums(gridSize, 0.0f);

    CHECK_CUDA(cudaMalloc(&d_input, inputBytes));
    CHECK_CUDA(cudaMalloc(&d_blockSums, partialBytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));

    // Warm up once to reduce first-launch noise.
    kernel<<<gridSize, kBlockSize, sharedBytes>>>(d_input, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<gridSize, kBlockSize, sharedBytes>>>(d_input, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

    CHECK_CUDA(cudaMemcpy(h_blockSums.data(),
                          d_blockSums,
                          partialBytes,
                          cudaMemcpyDeviceToHost));

    float gpuSum = std::accumulate(h_blockSums.begin(), h_blockSums.end(), 0.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));

    return {gpuSum, elapsedMs};
}

void printResult(const char* name, const BenchmarkResult& result, float referenceSum) {
    float absError = std::fabs(result.gpuSum - referenceSum);
    std::cout << std::left << std::setw(24) << name
              << std::setw(14) << std::fixed << std::setprecision(4) << result.elapsedMs
              << std::setw(16) << result.gpuSum
              << std::setw(12) << absError
              << ((absError < 1e-3f) ? "PASS" : "FAIL")
              << std::endl;
}

int main() {
    std::vector<float> h_input(kNumElements, 1.0f);
    float referenceSum = static_cast<float>(kNumElements);

    BenchmarkResult interleaved =
        runReductionBenchmark(h_input, 1, reductionInterleavedKernel);
    BenchmarkResult sequential =
        runReductionBenchmark(h_input, 1, reductionSequentialKernel);
    BenchmarkResult twoLoads =
        runReductionBenchmark(h_input, 2, reductionTwoLoadsKernel);

    std::cout << "Input elements: " << kNumElements << std::endl;
    std::cout << "Block size: " << kBlockSize << std::endl;
    std::cout << std::left << std::setw(24) << "Kernel"
              << std::setw(14) << "Time(ms)"
              << std::setw(16) << "GPU Sum"
              << std::setw(12) << "Abs Error"
              << "Check" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    printResult("Interleaved", interleaved, referenceSum);
    printResult("Sequential", sequential, referenceSum);
    printResult("Two loads", twoLoads, referenceSum);

    return EXIT_SUCCESS;
}
