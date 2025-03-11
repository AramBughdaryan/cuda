
#include <iostream>
#include <cuda.h>
#include <cassert>
#include "../helpers/cuda_helpers.h"
#define SECTION_SIZE 2048

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    
    // Perform scan within the block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] += XY[threadIdx.x - stride];
        }
    }
    
    // Write results back to global memory
    if (i < InputSize) {
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void Brent_Kung_scan_kernel(float *X, float *Y,
    int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if (i < InputSize) XY[threadIdx.x] = X[i];
    if (i+blockDim.x < InputSize) XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x+1) * 2* stride -1;
    if (index < SECTION_SIZE) {
    XY[index] += XY[index - stride];
    }
    }
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index + stride < SECTION_SIZE) {
    XY[index + stride] += XY[index];
    }
    }
    __syncthreads();
    if (i < InputSize) Y[i] = XY[threadIdx.x];
    if (i+blockDim.x < InputSize) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
}


// Kernel to compute the sum of each block
__global__ void blockSumKernel(float *input, float *blockSums, int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (i < InputSize) {
        XY[threadIdx.x] = input[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    
    // Perform scan within the block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] += XY[threadIdx.x - stride];
        }
    }
    
    // Last thread in the block writes the block sum
    if (threadIdx.x == blockDim.x - 1) {
        blockSums[blockIdx.x] = XY[threadIdx.x];
    }
}

// Kernel to add block sums to each element
__global__ void addBlockSumsKernel(float *input, float *blockSums, int InputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < InputSize && blockIdx.x > 0) {
        input[i] += blockSums[blockIdx.x - 1];
    }
}

void sequential_scan(float *x, float *y, int Max_i) {
    float *temp = new float[Max_i];
    temp[0] = x[0];
    y[0] = temp[0];
    for (int i = 1; i < Max_i; i++) {
        temp[i] = temp[i-1] + x[i];
        y[i] = temp[i];
    }
    delete[] temp;
}

int main() {
    const int inputSize = SECTION_SIZE;
    float *h_input, *h_output, *h_output_simple;
    float *d_input, *d_output, *d_blockSums;
    
    // Allocate host memory
    h_input = (float*)malloc(inputSize * sizeof(float));
    h_output = (float*)malloc(inputSize * sizeof(float));
    h_output_simple = (float*)malloc(inputSize * sizeof(float));
    initializeMatrix(h_input, inputSize, 1);
    
    // Allocate device memory
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    
    int blockSize = 512;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;
    
    // Allocate memory for block sums
    cudaMalloc(&d_blockSums, numBlocks * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 1: Perform local scan in each block
    Brent_Kung_scan_kernel<<<numBlocks, blockSize>>>(d_input, d_output, inputSize);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // If we have multiple blocks, we need to handle cross-block dependencies
    if (numBlocks > 1) {
        // Step 2: Compute the sum of each block
        blockSumKernel<<<numBlocks, blockSize>>>(d_input, d_blockSums, inputSize);
        
        // Wait for kernel to finish
        cudaDeviceSynchronize();
        
        // Step 3: Compute scan on the block sums (we can use the same kernel for small number of blocks)
        if (numBlocks <= blockSize) {
            Brent_Kung_scan_kernel<<<1, numBlocks>>>(d_blockSums, d_blockSums, numBlocks);
        } else {
            // For large number of blocks, we would need a recursive approach (not implemented here)
            std::cout << "Warning: Number of blocks exceeds block size, recursive scan needed\n";
        }
        
        // Wait for kernel to finish
        cudaDeviceSynchronize();
        
        // Step 4: Add the block sums to each element
        addBlockSumsKernel<<<numBlocks, blockSize>>>(d_output, d_blockSums, inputSize);
        
        // Wait for kernel to finish
        cudaDeviceSynchronize();
    }
    
    // Compute sequential scan for verification
    sequential_scan(h_input, h_output_simple, inputSize);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool testPassed = true;
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << "\t" << h_output_simple[i] << std::endl;
    }
    
    for (int i = 0; i < inputSize; i++) {
        float diff = fabs(h_output[i] - h_output_simple[i]);
        if (diff > 1e-4) {
            std::cout << "Mismatch at index " << i << ": " << diff << std::endl;
            testPassed = false;
        }
    }
    
    if (testPassed) {
        std::cout << "All tests passed" << std::endl;
    } else {
        std::cout << "Some tests failed" << std::endl;
    }
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_simple;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    
    return 0;
}