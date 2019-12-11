#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include "HighResolutionTimer.h"

void MatrixMultiply(float* C, const float* A, const float* B, unsigned int rank, bool bDevice = true, bool bSharedMem = true);

void main(void)
{
    // Determine the size of the matrices.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, i);
    }

    unsigned int rank = 1e4;
    bool bRunOnDevice = true;
    bool bUseSharedMem = false;

    // Allocate space for the matrices.
    float* A = new float[rank * rank];
    float* B = new float[rank * rank];
    float* C = new float[rank * rank];

    size_t numBytes = rank * rank * sizeof(float);

    // Initialize the matrices.
    memset(A, 0, numBytes);
    memset(B, 0, numBytes);
    memset(C, 0, numBytes);

    for (unsigned int i = 0; i < rank; ++i)
    {
        A[i * rank + i] = 2;
        B[i * rank + i] = 2;
    }

    HighResolutionTimer timer;
    bool quit = false;
    do
    {
        std::cout << "====================================================================" << std::endl;
        std::cout << "\nMatrix rank: " << rank << std::endl;
        std::cout << "Running on device: " << (bRunOnDevice ? "Yes" : "No") << std::endl;
        if (bRunOnDevice)
        {
            std::cout << "Use shared memory: " << (bUseSharedMem ? "Yes" : "No") << std::endl;
        }

        std::cout << "\nComputing result..." << std::endl;

        timer.Tick();
        MatrixMultiply(C, A, B, rank, bRunOnDevice, bUseSharedMem);
        cudaDeviceSynchronize();
        timer.Tick();

        // Verify the result (the diagonal should contain 4.0).
        for (unsigned int i = 0; i < rank; ++i)
        {
            unsigned int index = i * rank + i;
            if (C[index] != 4.0f)
            {
                std::cerr << "Value differs than expected value of 4.0: " << C[index] << std::endl;
            }
        }

        std::cout << "\nTime in microseconds: " << timer.ElapsedMicroSeconds() << std::endl;
        std::cout << "Time in milliseconds: " << timer.ElapsedMilliSeconds() << std::endl;
        std::cout << "Time in seconds:      " << timer.ElapsedSeconds() << std::endl;

        std::cout << "\nPress 'd' or 'h' to toggle device or host computation." << std::endl;
        std::cout << "Press 's' or 'g' to toggle shared memory or global memory." << std::endl;
        std::cout << "Press 'q' to quit or press [ENTER] to run again." << std::endl;

        fflush(stdin); // Flush any input that might be in the stream.
        int c = getc(stdin);

        switch (c)
        {
        case 'd':
        case 'D':
        case 'h':
        case 'H':
        {
            bRunOnDevice = !bRunOnDevice;
        }
        break;
        case 's':
        case 'S':
        case 'g':
        case 'G':
        {
            bUseSharedMem = !bUseSharedMem;
        }
        break;
        case 'q':
        case 'Q':
        {
            quit = true;
        }
        break;
        }

    } while (!quit);

    delete[] A;
    delete[] B;
    delete[] C;
}
