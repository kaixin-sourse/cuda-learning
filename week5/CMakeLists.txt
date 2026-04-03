cmake_minimum_required(VERSION 3.25)

project(cuda_week5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week5_blelloch_scan_demo src/blelloch_scan_demo.cu)
set_target_properties(week5_blelloch_scan_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week5_histogram_shared_atomic src/histogram_shared_atomic.cu)
set_target_properties(week5_histogram_shared_atomic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
