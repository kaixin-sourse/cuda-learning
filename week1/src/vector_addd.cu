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



__global__ void vectorAdddKernel(const float* a,const float* b, float* c,int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] + b[idx];
    }

}


int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n),h_b(n),h_c(n,0.0f);
    for(int i = 0; i < n; i ++) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
    }
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a,bytes));
    CHECK_CUDA(cudaMalloc(&d_b,bytes));
    CHECK_CUDA(cudaMalloc(&d_c,bytes));
    CHECK_CUDA(cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice));
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    vectorAdddKernel<<<gridSize,blockSize>>>(d_a,d_b,d_c,n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost));
    // 传到cpu检验一下GPU做的是否正确
    bool ok = true;
    for(int i = 0; i < n; i ++) {
        bool expected = h_a[i] + h_b[i];
        if(std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_c[i]
                      << ", expected " << expected << std::endl;
            ok =false;
            break;
        }
    }
    std::cout << "gridSize = " << gridSize << ", blockSize = " << blockSize << std::endl;
    std::cout << "Result check: " << (ok ? "PASS" : "FAIL") << std::endl;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    CHECK_CUDA(cudaFree(d_c));
    return ok ? EXIT_SUCCESS:EXIT_FAILURE;
}
