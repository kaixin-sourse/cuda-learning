#include <cstdlib>
#include <cuda_runtime.h>
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

constexpr int kBlockSize = 256;
constexpr int kNumElements = 2 * kBlockSize;

__global__ void blellochExclusiveScanKernel(const int* input, int* output) {
    __shared__ int temp[kNumElements];

    int tid = threadIdx.x;
    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    temp[ai] = input[ai];
    temp[bi] = input[bi];
    __syncthreads();

    // Up-sweep phase: build partial sums in a tree.
    for (int offset = 1; offset < kNumElements; offset <<= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < kNumElements) {
            temp[index] += temp[index - offset];
        }
        __syncthreads();
    }

    // Clear the root to convert the tree into an exclusive scan.
    if (tid == 0) {
        temp[kNumElements - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase: propagate prefix sums.
    for (int offset = kNumElements >> 1; offset > 0; offset >>= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < kNumElements) {
            int left = temp[index - offset];
            temp[index - offset] = temp[index];
            temp[index] += left;
        }
        __syncthreads();
    }

    output[ai] = temp[ai];
    output[bi] = temp[bi];
}

std::vector<int> cpuExclusiveScan(const std::vector<int>& input) {
    std::vector<int> output(input.size(), 0);
    int runningSum = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = runningSum;
        runningSum += input[i];
    }
    return output;
}

bool verify(const std::vector<int>& expected, const std::vector<int>& actual) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<int> h_input(kNumElements);
    for (int i = 0; i < kNumElements; ++i) {
        h_input[i] = (i % 5) + 1;
    }

    std::vector<int> h_output(kNumElements, 0);
    std::vector<int> h_reference = cpuExclusiveScan(h_input);

    int* d_input = nullptr;
    int* d_output = nullptr;
    size_t bytes = static_cast<size_t>(kNumElements) * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    blellochExclusiveScanKernel<<<1, kBlockSize>>>(d_input, d_output);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify(h_reference, h_output);

    std::cout << "Blelloch exclusive scan result: " << (ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "First 16 values:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
