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

void InitData(std::uint32_t n, float *p);

void PrintData(int m, int n, const float *p);

float ReduceCPU(std::uint32_t n, const float *in);

void VecAddCPU(int n, const float *a, const float *b, float *c);

void transpose_cpu(int m, int n, const float *src, float *dst);

void gemm_cpu(int m, int n, int k,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc);

float CompareResult(int m, int n, const float *a, float *b);

float CompareResult(int n, const float *a, float *b);

#endif