#include "common.h"

#include <iostream>

void InitData(int m, int n, float *p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            p[i * n + j] = static_cast<float>((i + j * 2) % 100);
        }
    }
}

void InitData(std::uint32_t n, float *p)
{
    for (int i = 0; i < n; ++i)
    {
        p[i] = static_cast<float>(i % 100);
    }
}

void PrintData(int m, int n, const float *p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << p[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

float ReduceCPU(std::uint32_t n, const float *in)
{
    float result = 0.f;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        result += in[i];
    }
    return result;
}

void VecAddCPU(int n, const float *a, const float *b, float *c)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

void transpose_cpu(int m, int n, const float *src, float *dst)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dst[j * m + i] = src[i * n + j];
        }
    }
}

float CompareResult(int m, int n, const float *a, float *b)
{
    float diff = 0.f;
    for (int i = 0; i < m * n; ++i)
    {
        diff += std::powf(a[i] - b[i], 2);
    }

    return std::sqrt(diff);
}

float CompareResult(int n, const float *a, float *b)
{
    float diff = 0.f;
    for (int i = 0; i < n; ++i)
    {
        diff += std::powf(a[i] - b[i], 2);
    }

    return std::sqrt(diff);
}

void gemm_cpu(int m, int n, int k,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float v = 0.f;
            for (int p = 0; p < k; ++p)
            {
                v += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = v;
        }
    }
}