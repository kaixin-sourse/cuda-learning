cmake_minimum_required(VERSION 3.25)

project(cuda_week9 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(week9_stream_overlap_compare src/stream_overlap_compare.cu)
set_target_properties(week9_stream_overlap_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(week9_stream_overlap_compare PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)