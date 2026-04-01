cmake_minimum_required(VERSION 3.25)

# 这个项目同时启用 C++ 和 CUDA 语言。
project(cuda_week1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Week 1 的第一个示例程序：向量加法。
add_executable(week1_vector_add week1_vector_add.cu)

# 这个属性让 CUDA 的编译行为更稳妥，适合学习阶段使用。
set_target_properties(week1_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
