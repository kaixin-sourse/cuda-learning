cmake_minimum_required(VERSION 3.25)

project(cuda_week3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week3_shared_memory_demo src/shared_memory_demo.cu)
set_target_properties(week3_shared_memory_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week3_matmul_compare src/matmul_compare.cu)
set_target_properties(week3_matmul_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
