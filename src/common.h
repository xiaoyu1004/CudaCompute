#ifndef COMMON_H
#define COMMON_H

#include "timer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define CUDA_NUM_THREADS 256
#define CUDA_GET_BLOCKS(N) ((N - 1) / CUDA_NUM_THREADS + 1)
#define CUDA_GET_TID() (blockIdx.x * blockDim.x + threadIdx.x)

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                 \
    /* Code block avoids redefinition of cudaError_t error */                 \
    do                                                                        \
    {                                                                         \
        cudaError_t error = condition;                                        \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cout << "ERROR: " << cudaGetErrorString(error) << std::endl; \
        }                                                                     \
    } while (0)

void InitData(int m, int n, float *p);

void PrintData(int m, int n, const float *p);

#endif