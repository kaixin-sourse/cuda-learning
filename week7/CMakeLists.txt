cmake_minimum_required(VERSION 3.25)

project(cuda_week7 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week7_profile_reduction src/profile_reduction_target.cu)
set_target_properties(week7_profile_reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week7_profile_matmul src/profile_matmul_target.cu)
set_target_properties(week7_profile_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
