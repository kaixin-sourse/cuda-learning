#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
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

constexpr int kBlockSize = 256;
constexpr int kScanElements = 512;
constexpr int kTileSize = 16;

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void reductionTwoLoadsKernel(const float* input, float* blockSums, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float value = 0.0f;
    if (globalIdx < n) {
        value += input[globalIdx];
    }
    if (globalIdx + blockDim.x < n) {
        value += input[globalIdx + blockDim.x];
    }

    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

__global__ void exclusiveScanKernel(const int* input, int* output) {
    __shared__ int temp[kScanElements];
    int tid = threadIdx.x;
    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    temp[ai] = input[ai];
    temp[bi] = input[bi];
    __syncthreads();

    for (int offset = 1; offset < kScanElements; offset <<= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < kScanElements) {
            temp[index] += temp[index - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp[kScanElements - 1] = 0;
    }
    __syncthreads();

    for (int offset = kScanElements >> 1; offset > 0; offset >>= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < kScanElements) {
            int left = temp[index - offset];
            temp[index - offset] = temp[index];
            temp[index] += left;
        }
        __syncthreads();
    }

    output[ai] = temp[ai];
    output[bi] = temp[bi];
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

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && tiledColA < n) ? a[row * n + tiledColA] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (tiledRowB < n && col < n) ? b[tiledRowB * n + col] : 0.0f;

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

struct ModuleResult {
    std::string name;
    int size;
    float elapsedMs;
    bool ok;
};

float elapsedKernelMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

ModuleResult runVectorAdd() {
    const int n = 1 << 22;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_a(n), h_b(n), h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1024);
        h_b[i] = static_cast<float>((i * 2) % 1024);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int grid = (n + kBlockSize - 1) / kBlockSize;
    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    vectorAddKernel<<<grid, kBlockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = elapsedKernelMs(start, stop);

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            ok = false;
            break;
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return {"Vector add", n, ms, ok};
}

ModuleResult runReduction() {
    const int n = 1 << 22;
    const size_t inputBytes = static_cast<size_t>(n) * sizeof(float);
    const int grid = (n + kBlockSize * 2 - 1) / (kBlockSize * 2);
    const size_t partialBytes = static_cast<size_t>(grid) * sizeof(float);
    std::vector<float> h_input(n, 1.0f);
    std::vector<float> h_partial(grid);

    float* d_input = nullptr;
    float* d_partial = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, inputBytes));
    CHECK_CUDA(cudaMalloc(&d_partial, partialBytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    reductionTwoLoadsKernel<<<grid, kBlockSize, kBlockSize * sizeof(float)>>>(d_input, d_partial, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = elapsedKernelMs(start, stop);

    CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial, partialBytes, cudaMemcpyDeviceToHost));
    float sum = std::accumulate(h_partial.begin(), h_partial.end(), 0.0f);
    bool ok = std::fabs(sum - static_cast<float>(n)) < 1e-3f;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_partial));
    return {"Reduction", n, ms, ok};
}

ModuleResult runScan() {
    std::vector<int> h_input(kScanElements);
    std::vector<int> h_output(kScanElements);
    std::vector<int> h_ref(kScanElements);

    for (int i = 0; i < kScanElements; ++i) {
        h_input[i] = (i % 5) + 1;
    }
    int running = 0;
    for (int i = 0; i < kScanElements; ++i) {
        h_ref[i] = running;
        running += h_input[i];
    }

    int* d_input = nullptr;
    int* d_output = nullptr;
    size_t bytes = static_cast<size_t>(kScanElements) * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    exclusiveScanKernel<<<1, kScanElements / 2>>>(d_input, d_output);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = elapsedKernelMs(start, stop);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool ok = (h_output == h_ref);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    return {"Exclusive scan", kScanElements, ms, ok};
}

ModuleResult runMatmul() {
    const int n = 256;
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> h_a(n * n), h_b(n * n), h_c(n * n), h_ref(n * n, 0.0f);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            h_a[row * n + col] = static_cast<float>((row + col) % 7);
            h_b[row * n + col] = static_cast<float>((row * 2 + col) % 11);
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += h_a[row * n + k] * h_b[k * n + col];
            }
            h_ref[row * n + col] = sum;
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
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    tiledMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = elapsedKernelMs(start, stop);

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < h_c.size(); ++i) {
        if (std::fabs(h_c[i] - h_ref[i]) > 1e-3f) {
            ok = false;
            break;
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return {"Tiled matmul", n, ms, ok};
}

void printResult(const ModuleResult& result) {
    std::cout << std::left
              << std::setw(18) << result.name
              << std::setw(12) << result.size
              << std::setw(14) << std::fixed << std::setprecision(4) << result.elapsedMs
              << (result.ok ? "PASS" : "FAIL")
              << std::endl;
}

int main() {
    std::cout << std::left
              << std::setw(18) << "Module"
              << std::setw(12) << "Size"
              << std::setw(14) << "Time(ms)"
              << "Check" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    printResult(runVectorAdd());
    printResult(runReduction());
    printResult(runScan());
    printResult(runMatmul());

    return EXIT_SUCCESS;
}
