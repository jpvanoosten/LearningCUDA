#include "cuda_runtime.h"

#include <algorithm>
#include <chrono>
#include <iostream>

cudaError_t addWithCuda_1(int n, float* x, float* y);
cudaError_t addWithCuda_256(int n, float* x, float* y);
cudaError_t addWithCuda_256xN(int n, float* x, float* y);

using namespace std::chrono;

void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    const int N = 1e6; // 1M elements.
    float* x = new float[N];
    float* y = new float[N];

    duration<double, std::micro> deltaTime;
    cudaError_t cudaStatus;

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Add vectors on CPU.
    auto t0 = high_resolution_clock::now();
    add(N, x, y );
    auto t1 = high_resolution_clock::now();

    deltaTime = t1 - t0;

    std::cout << "CPU Add: " << N << " numbers: " << deltaTime.count() << " us" << std::endl;

    // Check for errors.
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = std::max(maxError, std::abs(y[i] - 3.0f) );
    }

    std::cout << "Maximum Error: " << maxError << std::endl;

    t0 = high_resolution_clock::now();
    // Add vectors in a single CUDA thread.
    cudaStatus = addWithCuda_1(N, x, y);
    t1 = high_resolution_clock::now();

    deltaTime = t1 - t0;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::cout << "CUDA 1-thread Add: " << N << " numbers: " << deltaTime.count() << " us" << std::endl;

    // Check for errors.
    maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = std::max(maxError, std::abs(y[i] - 4.0f));
    }

    std::cout << "Maximum Error: " << maxError << std::endl;

    t0 = high_resolution_clock::now();
    // Add vectors in a single CUDA thread.
    cudaStatus = addWithCuda_256(N, x, y);
    t1 = high_resolution_clock::now();

    deltaTime = t1 - t0;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::cout << "CUDA 256-thread Add: " << N << " numbers: " << deltaTime.count() << " us" << std::endl;

    // Check for errors.
    maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = std::max(maxError, std::abs(y[i] - 5.0f));
    }

    std::cout << "Maximum Error: " << maxError << std::endl;

    t0 = high_resolution_clock::now();
    // Add vectors in a single CUDA thread.
    cudaStatus = addWithCuda_256xN(N, x, y);
    t1 = high_resolution_clock::now();

    deltaTime = t1 - t0;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::cout << "CUDA Thread Block Add: " << N << " numbers: " << deltaTime.count() << " us" << std::endl;

    // Check for errors.
    maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = std::max(maxError, std::abs(y[i] - 6.0f));
    }

    std::cout << "Maximum Error: " << maxError << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
