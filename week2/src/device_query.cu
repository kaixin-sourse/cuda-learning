#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

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

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    std::cout << "CUDA device count: " << deviceCount << std::endl;
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable device found." << std::endl;
        return EXIT_SUCCESS;
    }
    // 抄写一遍
    for(int i = 0; i < deviceCount; i ++) {
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop,i));
        std::cout << i << std::endl;
        std::cout << prop.name << std::endl;
        std::cout << prop.major << "." << prop.minor << std:: endl;
        std::cout << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << "GB" << std::endl;
        std::cout << prop.multiProcessorCount << std::endl;
        std::cout << prop.warpSize << std::endl;
        std::cout << prop.maxThreadsPerBlock << std::endl;
        std::cout << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
        std::cout << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
        std::cout << prop.sharedMemPerBlock / 1024.0 << std::endl;
        std::cout << prop.regsPerBlock << std::endl;
    }


    for (int device = 0; device < deviceCount; ++device) {
        // 存储GPU信息的结构体
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        std::cout << "\n=== Device " << device << " ===" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        // 这块 GPU 的“CUDA 功能等级
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total global memory: "
                  << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
                  << " GB" << std::endl;
        // GPU 中 SM（Streaming Multiprocessor，流式多处理器）的数量
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        // warp是线程调度的基本单位，比如1 warp 里面有 32个线程，所以 block的尺寸都是32的整数倍
        std::cout << "Warp size: " << prop.warpSize << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        // 一个 block 在对应方向上最多允许多少个线程数
        // prop.maxThreadsDim[0]：x 维最大线程数
        // prop.maxThreadsDim[1]：y 维最大线程数
        // prop.maxThreadsDim[2]：z 维最大线程数
        std::cout << "Max threads dim: "
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << std::endl;
        //grid 在 x、y、z 三个维度上允许的最大 block 数量
        //
        //也是一个长度为 3 的数组：
        //
        //prop.maxGridSize[0]：x 维最大 block 数
        //prop.maxGridSize[1]：y 维最大 block 数
        //prop.maxGridSize[2]：z 维最大 block 数
        //作用这决定了你 kernel launch 时：
        //
        //kernel<<<grid, block>>>();
        //
        //里面 grid 能开多大。
        //
        //一般一维程序里最常关心 grid.x 的上限。
        //
        //不过对大多数普通程序来说，这个上限通常都非常大，不是最先碰到的瓶颈。
        std::cout << "Max grid size: "
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << std::endl;
        // 每个 block 可用的共享内存大小
        // 共享内存是：
        //
        //同一个 block 内线程共享的一块片上高速内存，同一块肯定比调用外部的内存快得多
        //
        //它比全局显存快得多。
        //
        //作用
        //
        //这个参数很重要，因为如果你的 kernel 用了 shared memory，就必须保证：
        //
        //每个 block 使用的 shared memory
        //不超过这个上限
        std::cout << "Shared memory per block: "
                  << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        // 每个 block 最多可用的寄存器数量
        //
        //寄存器是线程最靠近计算单元、最快的存储资源之一。
        //
        //作用
        //
        //这个参数影响：
        //
        //一个 block 能不能顺利驻留在 SM 上
        //一个 SM 同时能驻留多少个 block
        //occupancy（占用率）
        //
        //如果你的 kernel 每个线程用太多寄存器，就可能导致：
        //
        //
        //同时运行的线程块数量减少
        //并发度下降
        //性能受影响
        std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    }

    return EXIT_SUCCESS;
}
