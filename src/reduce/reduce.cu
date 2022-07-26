#include "common.h"

#include <cfloat>
#include <fstream>
#include <iomanip>

/**
 *  16777216	2.80	22.37	0.00 reduce_base_line
    16777216	5.41	43.30	0.00 reduce_branch_differentiation
    16777216	4.79	38.35	0.00 reduce_shared_memory
    16777216	5.08	40.67	0.00 reduce_no_bank_conflict
    16777216	9.32	74.57	0.00 reduce_shared_memory_unroll2
    16777216	15.25	122.02	0.00 reduce_shared_memory_unroll4
    16777216	16.78	134.22	0.00 reduce_shared_memory_unroll8
 */ 

#define blocks 1024
// #define grid(n) (((n - 1) / blocks + 1))
// #define grid(n) (((n - 1) / blocks + 1) / 2) // unroll2
// #define grid(n) (((n - 1) / blocks + 1) / 4) // unroll4
#define grid(n) (((n - 1) / blocks + 1) / 8) // unroll8

__global__ void reduce_base_line(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *__restrict__ in_data = in + blockIdx.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            in_data[tid] += in_data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = in_data[tid];
    }
}

__global__ void reduce_branch_differentiation(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *__restrict__ in_data = in + blockIdx.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            in_data[index] += in_data[index + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = in_data[tid];
    }
}

__global__ void reduce_shared_memory(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    extern __shared__ float s_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    float *in_data = in + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    s_data[tid] = in_data[tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_data[tid];
    }
}

__global__ void reduce_no_bank_conflict(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *in_data = in + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ float s_data[];
    s_data[tid] = in_data[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s /= 2)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_data[tid];
    }
}

__global__ void reduce_shared_memory_unroll2(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *in_data = in + 2 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    // unroll 2
    extern __shared__ float s_data[];
    s_data[tid] = in_data[tid] + in_data[tid + blocks];
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s /= 2)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_data[tid];
    }
}

__global__ void reduce_shared_memory_unroll4(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *in_data = in + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ float s_data[];
    s_data[tid] = in_data[tid] + in_data[tid + blocks] + in_data[tid + 2 * blocks] + in_data[tid + 3 * blocks];
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s /= 2)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_data[tid];
    }
}

__global__ void reduce_shared_memory_unroll8(std::uint32_t n, float *__restrict__ in, float *__restrict__ out)
{
    float *in_data = in + 8 * blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ float s_data[];
    s_data[tid] = in_data[tid] + in_data[tid + blocks] + in_data[tid + 2 * blocks] + in_data[tid + 3 * blocks] +
                  in_data[tid + 4 * blocks] + in_data[tid + 5 * blocks] + in_data[tid + 6 * blocks] + in_data[tid + 7 * blocks];
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s /= 2)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_data[tid];
    }
}

int main(int argc, const char **argv)
{
    // std::ofstream fs("reduce_shared_memory_unroll2.txt", std::ios::out);
    // fs << "n\t\tperformance\t\tbandwidth\t\terr" << std::endl;

    std::uint32_t n = 1 << 20;
    float *h_data = new float[n];

    InitData(n, h_data);
    float h_sum = ReduceCPU(n, h_data);
    float *h_ref_data = new float[grid(n)];

    float *d_data;
    float *d_ref_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_data, grid(n) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref_data, 0, grid(n) * sizeof(float)));

    int loop_count = 10;

    double min = FLT_MAX;
    double max = FLT_MIN;
    double avg = 0;
    timer t;

    for (int i = 0; i < loop_count; ++i)
    {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
        t.start();
        // reduce_base_line<<<grid(n), blocks>>>(n, d_data, d_ref_data);
        // reduce_branch_differentiation<<<grid(n), blocks>>>(n, d_data, d_ref_data);
        // reduce_shared_memory<<<grid(n), blocks, blocks * sizeof(float)>>>(n, d_data, d_ref_data);
        // reduce_no_bank_conflict<<<grid(n), blocks, blocks * sizeof(float)>>>(n, d_data, d_ref_data);
        // reduce_shared_memory_unroll2<<<grid(n), blocks, blocks * sizeof(float)>>>(n, d_data, d_ref_data);
        // reduce_shared_memory_unroll4<<<grid(n), blocks, blocks * sizeof(float)>>>(n, d_data, d_ref_data);
        reduce_shared_memory_unroll8<<<grid(n), blocks, blocks * sizeof(float)>>>(n, d_data, d_ref_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        t.stop();
        CUDA_CHECK(cudaMemcpy(h_ref_data, d_ref_data, grid(n) * sizeof(float), cudaMemcpyDeviceToHost));

        min = std::min(min, t.get_elapsed_nano_seconds());
        max = std::max(max, t.get_elapsed_nano_seconds());
        avg += t.get_elapsed_nano_seconds();
    }
    avg /= loop_count;

    float d_sum = 0.f;
    for (std::uint32_t i = 0; i < grid(n); ++i)
    {
        d_sum += h_ref_data[i];
    }

    float err = CompareResult(1, &d_sum, &h_sum);
    if (err > 1e-2)
    {
        std::cout << "ERROR: " << err << std::endl;
    }
    else
    {
        std::cout << std::setprecision(2) << std::fixed << "PASS: result is correct" << std::endl;
        std::cout << "h_sum: " << h_sum << std::endl;
        std::cout << "d_sum: " << d_sum << std::endl;
    }

    double flops = n;
    double performance = flops / avg;
    double bandwidth = 2 * n * sizeof(float) / avg;
    std::cout << n << "\t" << performance << "\t" << bandwidth << "\t" << err << std::endl;

    std::cout << "max(ms)\tmin(ms)\tavg(ms)" << std::endl;
    std::cout << max * 1e-6 << "\t" << min * 1e-6 << "\t" << avg * 1e-6 << std::endl;

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_ref_data));

    delete[] h_data;
    delete[] h_ref_data;
}