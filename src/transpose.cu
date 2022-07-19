#include "common.h"

#include <iostream>

__global__ void matrix_trans_kernel(int m, int n, const float *src, float *dst)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n)
    {
        dst[j * m + i] = src[i * n + j];
    }

    // if (j < m && i < n)
    // {
    //     dst[i * m + j] = src[j * n + i];
    // }
}

const int BLOCK_SIZE = 32;

__global__ void matrix_trans_shared_kernel(int m, int n, const float *src, float *dst)
{
    __shared__ int buffer[BLOCK_SIZE][BLOCK_SIZE];

    int offset_x = blockIdx.x * blockDim.x;
    int offset_y = blockIdx.y * blockDim.y;

    int j = offset_x + threadIdx.x;
    int i = offset_y + threadIdx.y;
    if (i < m && j < n)
    {
        buffer[threadIdx.y][threadIdx.x] = src[i * n + j];
    }
    __syncthreads();

    i = offset_y + threadIdx.x;
    j = offset_x + threadIdx.y;
    if (i < m && j < n)
    {
        dst[j * m + i] = buffer[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    int M = 1 << 9;
    int N = 1 << 9;

    // int M = 1 << 5;
    // int N = 1 << 5;

    // host
    float *h_data = new float[M * N];
    float *h_ref_data = new float[M * N];
    InitData(M, N, h_data);

    // PrintData(M, N, h_data);

    // device
    float *src_data = nullptr;
    float *dst_data = nullptr;
    CUDA_CHECK(cudaMalloc(&src_data, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_data, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_data, h_data, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block{32u, 32u};
    // dim3 block{16u, 16u};
    dim3 grid{(M - 1u) / block.x + 1u, (N - 1) / block.y + 1u};

    int warm_count = 3;
    for (int i = 0; i < warm_count; ++i)
    {
        matrix_trans_kernel<<<block, grid>>>(M, N, src_data, dst_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int loop_count = 10;
    timer t;
    t.start();
    for (int i = 0; i < loop_count; ++i)
    {
        matrix_trans_kernel<<<block, grid>>>(M, N, src_data, dst_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    t.stop();
    double kernel_time = t.get_elapsed_nano_seconds();

    double bandwidth = loop_count * 2 * M * N * sizeof(float) / kernel_time;
    std::cout << "execute time: " << kernel_time / loop_count << "\t"
              << "memory bandwidth: " << bandwidth << "\t"
              << "peek performance: " << "" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_ref_data, dst_data, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // PrintData(N, M, h_ref_data);

    delete[] h_data;
    delete[] h_ref_data;

    CUDA_CHECK(cudaFree(src_data));
    CUDA_CHECK(cudaFree(dst_data));

    return 0;
}