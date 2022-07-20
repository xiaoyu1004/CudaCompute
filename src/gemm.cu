#include "common.h"

__global__ void gemm_kernel_naive(int m, int n, int k,
                                  const float *A, int lda,
                                  const float *B, int ldb,
                                  float *C, int ldc)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < n && iy < m)
    {
        float result = 0.f;
        for (int i = 0; i < k; ++i)
        {
            result += (A[iy * lda + i] * B[i * ldb + ix]);
        }
        C[iy * ldc + ix] = result;
    }
}

int main()
{
    int M = 1 << 9;
    int N = 1 << 9;
    int K = 1 << 9;

    // int M = 1 << 3;
    // int N = 1 << 3;
    // int K = 1 << 3;

    int lda = K;
    int ldb = N;
    int ldc = N;

    // host
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    // A
    InitData(M, K, h_A);
    // B
    InitData(K, N, h_B);

    // PrintData(M, K, h_A);
    // PrintData(K, N, h_B);

    // device
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block{32u, 32u};
    // dim3 block{16u, 16u};
    dim3 grid{(M - 1u) / block.x + 1u, (N - 1) / block.y + 1u};

    int warm_count = 3;
    for (int i = 0; i < warm_count; ++i)
    {
        gemm_kernel_naive<<<block, grid>>>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int loop_count = 10;
    timer t;
    t.start();
    for (int i = 0; i < loop_count; ++i)
    {
        gemm_kernel_naive<<<block, grid>>>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    t.stop();
    double kernel_time = t.get_elapsed_nano_seconds();
    double performance = loop_count * 2.f * M * N * K / kernel_time;

    // double bandwidth = loop_count * 2 * M * N * sizeof(float) / kernel_time;
    std::cout << "per kernel execute time: " << kernel_time / loop_count << "\t"
              << "peek performance: " << performance << " GB/S"
              << "\t"
              << "Peak percentage: " << performance / 913.92 * 100 << " %" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // compare result
    gemm_cpu(M, N, K, h_A, lda, h_B, ldb, h_C, ldc);

    // PrintData(M, N, h_C);
    // PrintData(M, N, h_C_ref);

    float err = CompareResult(M, N, h_C, h_C_ref);
    if (err > 1e-2)
    {
        std::cout << "ERROR: " << err << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}