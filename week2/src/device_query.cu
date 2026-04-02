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

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        std::cout << "\n=== Device " << device << " ===" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total global memory: "
                  << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
                  << " GB" << std::endl;
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Warp size: " << prop.warpSize << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim: "
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid size: "
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << std::endl;
        std::cout << "Shared memory per block: "
                  << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    }

    return EXIT_SUCCESS;
}
