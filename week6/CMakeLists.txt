cmake_minimum_required(VERSION 3.25)

project(cuda_week6 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week6_atomic_histogram_compare src/atomic_histogram_compare.cu)
set_target_properties(week6_atomic_histogram_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week6_topk_two_stage_demo src/topk_two_stage_demo.cu)
set_target_properties(week6_topk_two_stage_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
