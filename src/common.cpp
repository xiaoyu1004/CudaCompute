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

void PrintData(int m, int n, const float* p)
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