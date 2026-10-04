cmake_minimum_required(VERSION 3.20)
project(cuda_matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(vector_add src/vector_add/00_vector_add.cu)
add_executable(vector_add_float4 src/vector_add/01_vector_add_float4.cu)

foreach(factor IN ITEMS 2 4 8)
    add_executable(vector_add_unrolled${factor} src/vector_add/02_vector_add_unrolled.cu)
    target_compile_definitions(vector_add_unrolled${factor} PRIVATE ELEMENTS_PER_THREAD=${factor})
endforeach()
