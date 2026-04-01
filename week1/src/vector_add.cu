#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// Check every CUDA API call and stop immediately on failure.
// This avoids silent errors during early learning.
// 检查调用CUDA API是否有错误，如果有会及时报错
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// GPU kernel: each thread computes one output element.
// __global__ 表示GPU执行
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // Compute the global thread index.
    // 算出线程在全局中的编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-range threads in the last block.
    // 防止线程编号超出数组下标
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Input length. 2^20 is large enough for a simple parallel example.
    // CPU 准备数据 → 拷到 GPU → GPU 并行计算 → 结果拷回 CPU → CPU 验证
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Host-side vectors stored in CPU memory.
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);

    // Initialize input data.
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
    }

    // Device pointers stored in GPU memory.
    // 此时还没有分配内存
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Allocate memory on the GPU.
    // 为这三个指针分配GPU中的内存
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Copy inputs from CPU memory to GPU memory.

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Use 256 threads per block.
    // gridSize is the number of blocks needed to cover n elements.
    // 每个block中有256个线程
    const int blockSize = 256;
    // 每个grid中需要有多少个block
    const int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel with CUDA's <<<grid, block>>> syntax.
    // 在GPU上启动这个多的线程，让他们同时执行vectorAddKernel函数
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Check launch errors and wait for GPU execution to finish.
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to CPU memory.
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify results against the expected CPU-side answer.
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_c[i]
                      << ", expected " << expected << std::endl;
            ok = false;
            break;
        }
    }

    std::cout << "gridSize = " << gridSize << ", blockSize = " << blockSize << std::endl;
    std::cout << "Result check: " << (ok ? "PASS" : "FAIL") << std::endl;

    // Release GPU memory.
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
