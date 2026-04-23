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

constexpr int kTileSize = 16;
constexpr int kRepeats = 5;

__global__ void naiveMatmulKernel(const float* a, const float* b, float* c,int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n && col < n) {
        float sum = 0.0f;
        for(int k = 0; k < n; k ++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void tiledMatmulKernel(const float* a,const float* b, float* c, int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(int tile = 0; tile < (n + kTileSize - 1) / kTileSize; tile ++) {
        int tileColA = tile * kTileSize + threadIdx.x;
        int tileRowB = tile * kTileSize + threadIdx.y;
        if(row < n && tileColA < n) {
            tileA[threadIdx.y][threadIdx.x] = a[row * n + tileColA];
        }
        if(tileRowB < n && col < n) {
            tileB[threadIdx.y][threadIdx.x] = b[tileRowB * n + col];
        }
        __syncthreads();

        for(int k = 0; k < kTileSize; k ++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < n && col < n) {
        c[row * n + col] = sum;
    }

}

__global__ void tiledTwoOutputMatmulKernel(const float* a, const float* b, float* c,int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize * 2];

    int row = blockIdx.y * kTileSize + threadIdx.y;
    int col0 = blockIdx.x * (2 * kTileSize) + threadIdx.x;
    int col1 = col0 + kTileSize;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for(int tile = 0; tile <  (n + kTileSize - 1) / kTileSize; tile ++) {
        int tileColA = tile * kTileSize + threadIdx.x;
        int tileRowB = tile * kTileSize + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < n && tileColA < n) ? a[row * n + tileColA]:0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tileRowB < n && col0 < n) ? b[tileRowB * n + col0]:0.0f;
        tileB[threadIdx.y][threadIdx.x + kTileSize] = (tileRowB < n && col1 < n) ? b[tileRowB * n + col1]:0.0f;
        __syncthreads();

        for(int k = 0; k < kTileSize; k ++) {
            float aValue = tileA[threadIdx.y][k];
            sum0 += aValue * tileB[k][threadIdx.x];
            sum1 += aValue * tileB[k][threadIdx.x + kTileSize];
        }
        __syncthreads();
    }
    if(row < n && col0 < n) {
        c[row * n + col0] = sum0;
    }
    if(row < n && col1 < n) {
        c[row * n + col1] = sum1;
    }
}







struct KernelResult {
    std::string name;
    float elapsedMs;
    bool correct;
};

bool verifyMatrices(const std::vector<float>& expected,
                    const std::vector<float>& actual,
                    float tolerance = 1e-3f) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

float timeNaive(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    naiveMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        naiveMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsedMs / kRepeats;
}

float timeTiled(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    tiledMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        tiledMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsedMs / kRepeats;
}

float timeTiledTwoOutput(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + 2 * kTileSize - 1) / (2 * kTileSize),
              (n + kTileSize - 1) / kTileSize);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    tiledTwoOutputMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        tiledTwoOutputMatmulKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsedMs / kRepeats;
}

void runOneSize(int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> h_a(n * n);
    std::vector<float> h_b(n * n);
    std::vector<float> h_naive(n * n, 0.0f);
    std::vector<float> h_tiled(n * n, 0.0f);
    std::vector<float> h_twoOutput(n * n, 0.0f);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            h_a[row * n + col] = static_cast<float>((row + col) % 7);
            h_b[row * n + col] = static_cast<float>((row * 2 + col) % 11);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_naive = nullptr;
    float* d_tiled = nullptr;
    float* d_twoOutput = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_naive, bytes));
    CHECK_CUDA(cudaMalloc(&d_tiled, bytes));
    CHECK_CUDA(cudaMalloc(&d_twoOutput, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    float naiveMs = timeNaive(d_a, d_b, d_naive, n);
    float tiledMs = timeTiled(d_a, d_b, d_tiled, n);
    float twoOutputMs = timeTiledTwoOutput(d_a, d_b, d_twoOutput, n);

    CHECK_CUDA(cudaMemcpy(h_naive.data(), d_naive, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_tiled.data(), d_tiled, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_twoOutput.data(), d_twoOutput, bytes, cudaMemcpyDeviceToHost));

    bool tiledOk = verifyMatrices(h_naive, h_tiled);
    bool twoOutputOk = verifyMatrices(h_naive, h_twoOutput);

    std::cout << std::left
              << std::setw(10) << n
              << std::setw(22) << "Naive"
              << std::setw(14) << std::fixed << std::setprecision(4) << naiveMs
              << std::setw(12) << 1.0
              << "PASS" << std::endl;

    std::cout << std::left
              << std::setw(10) << n
              << std::setw(22) << "Tiled shared"
              << std::setw(14) << tiledMs
              << std::setw(12) << (naiveMs / tiledMs)
              << (tiledOk ? "PASS" : "FAIL") << std::endl;

    std::cout << std::left
              << std::setw(10) << n
              << std::setw(22) << "Tiled two-output"
              << std::setw(14) << twoOutputMs
              << std::setw(12) << (naiveMs / twoOutputMs)
              << (twoOutputOk ? "PASS" : "FAIL") << std::endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_naive));
    CHECK_CUDA(cudaFree(d_tiled));
    CHECK_CUDA(cudaFree(d_twoOutput));
}


int main() {
    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(22) << "Version"
              << std::setw(14) << "Time(ms)"
              << std::setw(12) << "Speedup"
              << "Check" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (int n : {256, 512}) {
        runOneSize(n);
    }

    return EXIT_SUCCESS;
}