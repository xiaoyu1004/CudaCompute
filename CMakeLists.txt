cmake_minimum_required(VERSION 3.10)
project(CudaCompute LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# add_executable(transpose src/transpose.cu src/common.cpp)
add_executable(gemm src/gemm.cu src/common.cpp)