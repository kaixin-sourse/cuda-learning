cmake_minimum_required(VERSION 3.25)

project(cuda_week11 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week11_medical_image_convolution src/medical_image_convolution.cu)
set_target_properties(week11_medical_image_convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
