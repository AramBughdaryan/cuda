#include <cuda.h>
#include <iostream>
#include <cassert>

#define SIZE 1024

__global__ void partialSumKernel(float* X, float* Y) {
    __shared__ float partialSum[SIZE];

    partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];
    unsigned int t = threadIdx.x;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();
    if (t == 0) {
        Y[blockIdx.x] = partialSum[0];
    }
}

float naive_sum(float* X){
    float partial_sum = 0.0f;
    for (int i = 0; i < SIZE; i++)
        partial_sum += X[i];
    return partial_sum;
}

__global__ void optimisedPartialSumKernel(float* X, float* Y){
    __shared__ float partialSum[SIZE];
    partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];

    unsigned int t = threadIdx.x;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1)
    {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();
    if (t == 0){
        Y[blockIdx.x] = partialSum[0];
    }
}

void testPartialSumKernel() {
    const int numBlocks = 1;
    const int numThreads = SIZE;
    const int dataSize = numBlocks * numThreads;

    float h_X[dataSize], h_Y[numBlocks], h_Y_optimised[numBlocks];
    
    for (int i = 0; i < dataSize; ++i) {
        h_X[i] = rand() / (float)RAND_MAX; // Initialize input array with 1.0f
    }

    float *d_X, *d_Y, *d_Y_optimised;
    cudaMalloc(&d_X, dataSize * sizeof(float));
    cudaMalloc(&d_Y, numBlocks * sizeof(float));
    cudaMalloc(&d_Y_optimised, numBlocks * sizeof(float));

    cudaMemcpy(d_X, h_X, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Measure time for partialSumKernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    partialSumKernel<<<numBlocks, numThreads>>>(d_X, d_Y);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for partialSumKernel: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Measure time for optimisedPartialSumKernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    optimisedPartialSumKernel<<<numBlocks, numThreads>>>(d_X, d_Y_optimised);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for optimisedPartialSumKernel: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_Y, d_Y, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y_optimised, d_Y_optimised, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Y_optimised);

    // Check the result
    float expectedSum = naive_sum(h_X);
    assert(h_Y[0] == expectedSum);
    std::cout << "Test passed!" << std::endl;
    std::cout <<"Calculated: "<< h_Y[0] << "\nExpected: " <<expectedSum<<std::endl;
    std::cout <<"Optimised: "<< h_Y_optimised[0] << "\nExpected: " <<expectedSum<<std::endl;
}

int main() {
    testPartialSumKernel();
    return 0;
}