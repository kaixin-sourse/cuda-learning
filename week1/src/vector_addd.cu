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

}
