cmake_minimum_required(VERSION 3.10)
project(CudaCompute LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(common_src ${PROJECT_SOURCE_DIR}/src/common/common.cpp)

include_directories(src/common)

add_subdirectory(src/reduce)