#include "common.h"

#include <iostream>

void InitData(int m, int n, float *p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            p[i * n + j] = static_cast<float>(i + j * 2);
        }
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