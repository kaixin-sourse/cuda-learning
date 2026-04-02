cmake_minimum_required(VERSION 3.25)

project(cuda_week2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week2_device_query src/device_query.cu)
set_target_properties(week2_device_query PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week2_vector_add_benchmark src/vector_add_benchmark.cu)
set_target_properties(week2_vector_add_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
