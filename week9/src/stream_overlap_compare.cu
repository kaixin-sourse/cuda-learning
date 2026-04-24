#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int kTotalElements = 1 << 24;
constexpr int kChunkElements = 1 << 20;
constexpr int kStreamCount = 4;
constexpr int kBlockSize = 256;
constexpr float kAlpha = 2.5f;
constexpr float kBeta = 7.0f;

__global__ void scaleKernel(const float* input, float* temp, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = input[idx] * alpha;
    }
}

__global__ void biasKernel(const float* temp, float* output, int n, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = temp[idx] + beta;
    }
}

double nowMs() {
    using Clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch())
        .count();
}

void initializeInput(float* input, int n) {
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 1024) * 0.25f;
    }
}

bool verifyOutput(const float* input, const float* output, int n) {
    for (int i = 0; i < n; ++i) {
        float expected = input[i] * kAlpha + kBeta;
        if (std::fabs(output[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i
                      << ": expected " << expected
                      << ", got " << output[i] << std::endl;
            return false;
        }
    }
    return true;
}

struct RunResult {
    std::string name;
    double elapsedMs;
    bool ok;
};

RunResult runPageableSequential() {
    std::vector<float> h_input(kTotalElements);
    std::vector<float> h_output(kTotalElements, 0.0f);
    initializeInput(h_input.data(), kTotalElements);

    float* d_input = nullptr;
    float* d_temp = nullptr;
    float* d_output = nullptr;
    size_t bytes = static_cast<size_t>(kTotalElements) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    int gridSize = (kTotalElements + kBlockSize - 1) / kBlockSize;

    double start = nowMs();
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    scaleKernel<<<gridSize, kBlockSize>>>(d_input, d_temp, kTotalElements, kAlpha);
    CHECK_CUDA(cudaGetLastError());
    biasKernel<<<gridSize, kBlockSize>>>(d_temp, d_output, kTotalElements, kBeta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    double stop = nowMs();

    bool ok = verifyOutput(h_input.data(), h_output.data(), kTotalElements);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));

    return {"Pageable sequential", stop - start, ok};
}

RunResult runPinnedSequential() {
    float* h_input = nullptr;
    float* h_output = nullptr;
    float* d_input = nullptr;
    float* d_temp = nullptr;
    float* d_output = nullptr;

    size_t bytes = static_cast<size_t>(kTotalElements) * sizeof(float);

    CHECK_CUDA(cudaMallocHost(&h_input, bytes));
    CHECK_CUDA(cudaMallocHost(&h_output, bytes));
    initializeInput(h_input, kTotalElements);

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    int gridSize = (kTotalElements + kBlockSize - 1) / kBlockSize;

    double start = nowMs();
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    scaleKernel<<<gridSize, kBlockSize>>>(d_input, d_temp, kTotalElements, kAlpha);
    CHECK_CUDA(cudaGetLastError());
    biasKernel<<<gridSize, kBlockSize>>>(d_temp, d_output, kTotalElements, kBeta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    double stop = nowMs();

    bool ok = verifyOutput(h_input, h_output, kTotalElements);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));

    return {"Pinned sequential", stop - start, ok};
}

RunResult runPinnedMultiStream() {
    float* h_input = nullptr;
    float* h_output = nullptr;

    size_t totalBytes = static_cast<size_t>(kTotalElements) * sizeof(float);
    size_t chunkBytes = static_cast<size_t>(kChunkElements) * sizeof(float);

    CHECK_CUDA(cudaMallocHost(&h_input, totalBytes));
    CHECK_CUDA(cudaMallocHost(&h_output, totalBytes));
    initializeInput(h_input, kTotalElements);

    cudaStream_t streams[kStreamCount]{};
    float* d_input[kStreamCount]{};
    float* d_temp[kStreamCount]{};
    float* d_output[kStreamCount]{};

    for (int i = 0; i < kStreamCount; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaMalloc(&d_input[i], chunkBytes));
        CHECK_CUDA(cudaMalloc(&d_temp[i], chunkBytes));
        CHECK_CUDA(cudaMalloc(&d_output[i], chunkBytes));
    }

    int numChunks = (kTotalElements + kChunkElements - 1) / kChunkElements;

    double start = nowMs();
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        // 当前chunk放到第几个stream中
        int streamId = chunk % kStreamCount;
        // 当前chunk在总数组中的起始位置
        int offset = chunk * kChunkElements;
        // 最后一个chunk可能不满
        int currentElements = std::min(kChunkElements, kTotalElements - offset);
        size_t currentBytes = static_cast<size_t>(currentElements) * sizeof(float);
        int gridSize = (currentElements + kBlockSize - 1) / kBlockSize;
        // 异步拷贝，cpu2gpu，使用第streamId这个stream
        CHECK_CUDA(cudaMemcpyAsync(d_input[streamId],
                                   h_input + offset,
                                   currentBytes,
                                   cudaMemcpyHostToDevice,
                                   streams[streamId]));
        // 不使用共享内存
        scaleKernel<<<gridSize, kBlockSize, 0, streams[streamId]>>>(
            d_input[streamId], d_temp[streamId], currentElements, kAlpha);
        CHECK_CUDA(cudaGetLastError());
        biasKernel<<<gridSize, kBlockSize, 0, streams[streamId]>>>(
            d_temp[streamId], d_output[streamId], currentElements, kBeta);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(h_output + offset,
                                   d_output[streamId],
                                   currentBytes,
                                   cudaMemcpyDeviceToHost,
                                   streams[streamId]));
    }
    // 等待stream上面所有的任务都完成，才能向下走
    CHECK_CUDA(cudaDeviceSynchronize());
    double stop = nowMs();

    bool ok = verifyOutput(h_input, h_output, kTotalElements);

    for (int i = 0; i < kStreamCount; ++i) {
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_temp[i]));
        CHECK_CUDA(cudaFree(d_output[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));

    return {"Pinned multi-stream", stop - start, ok};
}

void printResult(const RunResult& result) {
    std::cout << std::left
              << std::setw(24) << result.name
              << std::setw(14) << std::fixed << std::setprecision(4) << result.elapsedMs
              << (result.ok ? "PASS" : "FAIL")
              << std::endl;
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Async engine count: " << prop.asyncEngineCount << std::endl;
    std::cout << "Total elements: " << kTotalElements << std::endl;
    std::cout << "Chunk elements: " << kChunkElements << std::endl;
    std::cout << "Stream count: " << kStreamCount << std::endl;
    std::cout << std::left
              << std::setw(24) << "Version"
              << std::setw(14) << "Time(ms)"
              << "Check" << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    printResult(runPageableSequential());
    printResult(runPinnedSequential());
    printResult(runPinnedMultiStream());

    return EXIT_SUCCESS;
}
