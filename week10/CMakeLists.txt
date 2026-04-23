cmake_minimum_required(VERSION 3.25)

project(cuda_week10 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week10_mini_library_benchmark src/mini_library_benchmark.cu)
set_target_properties(week10_mini_library_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
