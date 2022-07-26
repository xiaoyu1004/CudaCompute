#include "common.h"

#include <cfloat>
#include <fstream>
#include <iomanip>

#define blocks 1024
// #define grid(n) (((n - 1) / blocks + 1))
// #define grid(n) (((n - 1) / blocks + 1) / 2) // unroll2
// #define grid(n) (((n - 1) / blocks + 1) / 4) // unroll4
#define grid(n) (((n - 1) / blocks + 1) / 8) // unroll8

__global__ void vecadd_base_line(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void vecadd_unroll2(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c)
{
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
    c[idx + blocks] = a[idx + blocks] + b[idx + blocks];
}

__global__ void vecadd_unroll4(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c)
{
    int idx = 4 * blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
    c[idx + blocks] = a[idx + blocks] + b[idx + blocks];
    c[idx + 2 * blocks] = a[idx + 2 * blocks] + b[idx + 2 * blocks];
    c[idx + 3 * blocks] = a[idx + 3 * blocks] + b[idx + 3 * blocks];
}

__global__ void vecadd_unroll8(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c)
{
    int idx = 8 * blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
    c[idx + blocks] = a[idx + blocks] + b[idx + blocks];
    c[idx + 2 * blocks] = a[idx + 2 * blocks] + b[idx + 2 * blocks];
    c[idx + 3 * blocks] = a[idx + 3 * blocks] + b[idx + 3 * blocks];
    c[idx + 4 * blocks] = a[idx + 4 * blocks] + b[idx + 4 * blocks];
    c[idx + 5 * blocks] = a[idx + 5 * blocks] + b[idx + 5 * blocks];
    c[idx + 6 * blocks] = a[idx + 6 * blocks] + b[idx + 6 * blocks];
    c[idx + 7 * blocks] = a[idx + 7 * blocks] + b[idx + 7 * blocks];
}

int main(int argc, const char **argv)
{
    // std::ofstream fs("reduce_shared_memory_unroll2.txt", std::ios::out);
    // fs << "n\t\tperformance\t\tbandwidth\t\terr" << std::endl;

    int n = 1 << 22;
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    float *h_ref_c = new float[n];

    InitData(n, h_a);
    InitData(n, h_b);

    VecAddCPU(n, h_a, h_b, h_c);

    float *d_a;
    float *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    int loop_count = 10;

    double min = FLT_MAX;
    double max = FLT_MIN;
    double avg = 0;
    timer t;

    for (int i = 0; i < loop_count; ++i)
    {
        t.start();
        // vecadd_base_line<<<grid(n), blocks>>>(n, d_a, d_b, d_c);
        // vecadd_unroll2<<<grid(n), blocks>>>(n, d_a, d_b, d_c);
        // vecadd_unroll4<<<grid(n), blocks>>>(n, d_a, d_b, d_c);
        vecadd_unroll8<<<grid(n), blocks>>>(n, d_a, d_b, d_c);
        CUDA_CHECK(cudaDeviceSynchronize());
        t.stop();

        min = std::min(min, t.get_elapsed_nano_seconds());
        max = std::max(max, t.get_elapsed_nano_seconds());
        avg += t.get_elapsed_nano_seconds();
    }
    avg /= loop_count;

    CUDA_CHECK(cudaMemcpy(h_ref_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
    float err = CompareResult(n, h_c, h_ref_c);
    if (err > 1e-2)
    {
        std::cout << "ERROR: " << err << std::endl;
    }
    else
    {
        std::cout << "PASS: result is correct" << std::endl;
    }

    double flops = n;
    double performance = flops / avg;
    double bandwidth = 3 * n * sizeof(float) / avg;
    std::cout << std::setprecision(2) << std::fixed << n << "\t" << performance << "\t" << bandwidth << "\t" << err << std::endl;

    std::cout << "max(ms)\tmin(ms)\tavg(ms)" << std::endl;
    std::cout << max * 1e-6 << "\t" << min * 1e-6 << "\t" << avg * 1e-6 << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref_c;
}