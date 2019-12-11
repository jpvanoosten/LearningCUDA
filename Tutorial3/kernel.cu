#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

void MatrixMultiplyHost(float* C, const float* A, const float* B, unsigned int rank)
{
    for (unsigned int i = 0; i < rank; ++i) // i is the row index
    {
        for (unsigned int j = 0; j < rank; ++j) // j is the column index
        {
            unsigned int index = (i * rank) + j;
            float sum = 0;
            for (unsigned int k = 0; k < rank; ++k)
            {
                sum += A[i * rank + k] * B[k * rank + j];
            }
            C[index] = sum;
        }
    }
}

__global__ void MatrixMultiplyKernel_GlobalMem(float* C, const float* A, const float* B, unsigned int rank)
{
    // Compute the row index
    unsigned int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    // Compute the column index
    unsigned int j = (blockDim.x * blockIdx.x) + threadIdx.x;

    unsigned int index = (i * rank) + j;
    float sum = 0.0f;
    for (unsigned int k = 0; k < rank; ++k)
    {
        sum += A[i * rank + k] * B[k * rank + j];
    }
    C[index] = sum;
}

#define BLOCK_SIZE 16

__global__ void MatrixMultiplyKernel_SharedMem(float* C, const float* A, const float* B, unsigned int rank)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Allocate share memory to store the matrix data in tiles
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // Compute the column index
    unsigned int j = (blockDim.x * bx) + tx;
    // Compute the row index
    unsigned int i = (blockDim.y * by) + ty;

    unsigned int index = (i * rank) + j;
    float sum = 0.0f;

    // Loop through the tiles of the input matrices
    // in separate phases of size BLOCK_SIZE
    for (unsigned int phase = 0; phase < rank / BLOCK_SIZE; ++phase)
    {
        // Allow each thread in the block to populate the shared memory
        sA[ty][tx] = A[i * rank + (phase * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(phase * BLOCK_SIZE + ty) * rank + j];
        __syncthreads();

        for (unsigned int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    C[index] = sum;
}

__constant__ unsigned int offsets[8];

void MatrixMultiply(float* C, const float* A, const float* B, unsigned int rank, bool bDevice, bool bSharedMem)
{
    if (bDevice)
    {
        // Clear the error value
        cudaGetLastError();

        // Allocate device buffers to store matrices.
        float* devA;
        float* devB;
        float* devC;
        size_t bufferSize = (rank * rank) * sizeof(float);

        cudaMalloc(&devA, bufferSize);
        cudaMalloc(&devB, bufferSize);
        cudaMalloc(&devC, bufferSize);

        // Copy data from host->device
        cudaMemcpy(devA, A, bufferSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B, bufferSize, cudaMemcpyHostToDevice);

        // Compute the grid size
        size_t blocks = (size_t)ceilf(rank / (float)BLOCK_SIZE);
        dim3 gridDim(blocks, blocks, 1);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

        // Execute the kernel on the device
        if (bSharedMem)
        {
            // Execute the kernel using shared memory
            MatrixMultiplyKernel_SharedMem << < gridDim, blockDim >> > (devC, devA, devB, rank);
        }
        else
        {
            // Execute the kernel using only global memory
            MatrixMultiplyKernel_GlobalMem << < gridDim, blockDim >> > (devC, devA, devB, rank);
        }

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(error) << std::endl;
        }

        // Copy device buffers back to host memory
        cudaMemcpy(C, devC, bufferSize, cudaMemcpyDeviceToHost);

        // free the device buffers.
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }
    else
    {
        // Exectue the host only version
        MatrixMultiplyHost(C, A, B, rank);
    }
}
