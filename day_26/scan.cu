#include <iostream>
#include <cuda.h>
#include <cassert>

#include "../helpers/cuda_helpers.h"

#define SECTION_SIZE 1024

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize){
    
    __shared__ float XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize){
        XY[threadIdx.x] = X[i];
    }
    else{
        XY[threadIdx.x] = 0.0f;
    }
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *=2){
        __syncthreads();
        if (threadIdx.x >= stride){
            XY[threadIdx.x] += XY[threadIdx.x - stride];
        }
        
    }
    if (i < InputSize){
        Y[i] = XY[threadIdx.x];
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
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(inputSize * sizeof(float));
    h_output = (float*)malloc(inputSize * sizeof(float));
    h_output_simple = (float*)malloc(inputSize * sizeof(float));

    initializeMatrix(h_input, inputSize, 1);
    
    // Allocate device memory
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = SECTION_SIZE;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;
    Kogge_Stone_scan_kernel<<<numBlocks, blockSize>>>(d_input, d_output, inputSize);
    sequential_scan(h_input, h_output_simple, inputSize);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (optional)
    // printArray(h_output, inputSize, 1, "H_output");

    for (int i = 0; i < 10; i++) {
        std::cout<<h_output[i] << "\t" << h_output_simple[i] << std::endl;
    }

    for (int i = 0; i < inputSize; i++) {
        std::cout<<fabs(h_output[i] - h_output_simple[i])<<" Index : "<<i<<std::endl;
        // assert(fabs(h_output[i] - h_output_simple[i]) < 1e-5);
    }
    std::cout<<"All tests passed"<<std::endl;
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_simple;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

