cmake_minimum_required(VERSION 3.25)

project(cuda_week4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week4_reduction_compare src/reduction_compare.cu)
set_target_properties(week4_reduction_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
