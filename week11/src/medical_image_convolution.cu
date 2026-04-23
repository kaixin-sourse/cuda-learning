#include <algorithm>
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

constexpr int kBlockSize = 16;
constexpr int kKernelRadius = 1;
constexpr int kKernelWidth = 3;

__constant__ float c_filter[kKernelWidth * kKernelWidth];

int clampInt(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

__device__ int clampDevice(int value, int low, int high) {
    return max(low, min(value, high));
}

__global__ void convolutionGlobalKernel(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
        for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
            int sx = clampDevice(x + kx, 0, width - 1);
            int sy = clampDevice(y + ky, 0, height - 1);
            float pixel = input[sy * width + sx];
            float weight = c_filter[(ky + kKernelRadius) * kKernelWidth + (kx + kKernelRadius)];
            sum += pixel * weight;
        }
    }

    output[y * width + x] = sum;
}

__global__ void convolutionSharedKernel(const float* input, float* output, int width, int height) {
    constexpr int sharedWidth = kBlockSize + 2 * kKernelRadius;
    constexpr int sharedHeight = kBlockSize + 2 * kKernelRadius;

    __shared__ float tile[sharedHeight][sharedWidth];

    int blockOriginX = blockIdx.x * kBlockSize;
    int blockOriginY = blockIdx.y * kBlockSize;
    int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
    int threadCount = blockDim.x * blockDim.y;

    for (int index = linearThread; index < sharedWidth * sharedHeight; index += threadCount) {
        int localY = index / sharedWidth;
        int localX = index % sharedWidth;
        int globalX = clampDevice(blockOriginX + localX - kKernelRadius, 0, width - 1);
        int globalY = clampDevice(blockOriginY + localY - kKernelRadius, 0, height - 1);
        tile[localY][localX] = input[globalY * width + globalX];
    }

    __syncthreads();

    int x = blockOriginX + threadIdx.x;
    int y = blockOriginY + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    int localX = threadIdx.x + kKernelRadius;
    int localY = threadIdx.y + kKernelRadius;

    for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
        for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
            float pixel = tile[localY + ky][localX + kx];
            float weight = c_filter[(ky + kKernelRadius) * kKernelWidth + (kx + kKernelRadius)];
            sum += pixel * weight;
        }
    }

    output[y * width + x] = sum;
}

std::vector<float> makeSyntheticCtImage(int width, int height) {
    std::vector<float> image(static_cast<size_t>(width) * height);
    float cx = 0.5f * static_cast<float>(width);
    float cy = 0.5f * static_cast<float>(height);
    float radius = 0.32f * static_cast<float>(std::min(width, height));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float distance = std::sqrt(dx * dx + dy * dy);
            float body = (distance < radius) ? 1.0f : 0.05f;
            float gradient = static_cast<float>(x + y) / static_cast<float>(width + height);
            image[static_cast<size_t>(y) * width + x] = body + 0.15f * gradient;
        }
    }
    return image;
}

void cpuConvolution(const std::vector<float>& input,
                    std::vector<float>& output,
                    const float* filter,
                    int width,
                    int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
                for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
                    int sx = clampInt(x + kx, 0, width - 1);
                    int sy = clampInt(y + ky, 0, height - 1);
                    float pixel = input[static_cast<size_t>(sy) * width + sx];
                    float weight = filter[(ky + kKernelRadius) * kKernelWidth + (kx + kKernelRadius)];
                    sum += pixel * weight;
                }
            }
            output[static_cast<size_t>(y) * width + x] = sum;
        }
    }
}

bool verifyImage(const std::vector<float>& expected,
                 const std::vector<float>& actual,
                 float tolerance = 1e-4f) {
    if (expected.size() != actual.size()) {
        return false;
    }
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

float runCudaConvolution(const std::vector<float>& input,
                         std::vector<float>& output,
                         int width,
                         int height,
                         bool useShared) {
    size_t bytes = static_cast<size_t>(width) * height * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    if (useShared) {
        convolutionSharedKernel<<<grid, block>>>(d_input, d_output, width, height);
    } else {
        convolutionGlobalKernel<<<grid, block>>>(d_input, d_output, width, height);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return elapsedMs;
}

void runOneSize(int width, int height, const float* filter) {
    std::vector<float> input = makeSyntheticCtImage(width, height);
    std::vector<float> cpuOutput(static_cast<size_t>(width) * height);
    std::vector<float> gpuGlobal(static_cast<size_t>(width) * height);
    std::vector<float> gpuShared(static_cast<size_t>(width) * height);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuConvolution(input, cpuOutput, filter, width, height);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    double cpuMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    float globalMs = runCudaConvolution(input, gpuGlobal, width, height, false);
    float sharedMs = runCudaConvolution(input, gpuShared, width, height, true);

    bool globalOk = verifyImage(cpuOutput, gpuGlobal);
    bool sharedOk = verifyImage(cpuOutput, gpuShared);

    std::cout << std::left
              << std::setw(12) << (std::to_string(width) + "x" + std::to_string(height))
              << std::setw(16) << std::fixed << std::setprecision(4) << cpuMs
              << std::setw(18) << globalMs
              << std::setw(18) << sharedMs
              << ((globalOk && sharedOk) ? "PASS" : "FAIL")
              << std::endl;
}

int main() {
    const float smoothingFilter[kKernelWidth * kKernelWidth] = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };

    CHECK_CUDA(cudaMemcpyToSymbol(c_filter, smoothingFilter, sizeof(smoothingFilter)));

    std::cout << std::left
              << std::setw(12) << "Image"
              << std::setw(16) << "CPU(ms)"
              << std::setw(18) << "CUDA global(ms)"
              << std::setw(18) << "CUDA shared(ms)"
              << "Check" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    runOneSize(512, 512, smoothingFilter);
    runOneSize(1024, 1024, smoothingFilter);

    return EXIT_SUCCESS;
}
