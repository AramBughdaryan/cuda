#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

void matrixAddition(float* C, const float* A, const float* B, int N);

__global__ void matrixAdditionKernel(float* out, float* inp_1, float* inp_2, int n){
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    int Col = blockDim.y * blockIdx.y + threadIdx.y;

    if (Row < n && Col < n ){
        out[n * Row + Col] = inp_1[n * Row + Col] + inp_2[n * Row + Col];
    }
}


void matrixAddition(float* C, const float* A, const float* B, int N){
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, size), "Allocate memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "Allocate memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "Allocate memory for C");


    printArray(A, N, N, "Array A");
    printArray(B, N, N, "Array B");
    // printArray(C, N, N, 'Array C');

    checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "Copy A from host to device");
    checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "Copy B from host to device");

    const int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1)/threadsPerBlock.x, (N + blockSize - 1)/threadsPerBlock.y);

    matrixAdditionKernel<<<gridDim, blockSize>>>(d_C, d_A, d_B, N);

    checkCudaError(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost), "Copy output array from device to host");
    printArray(C, N, N, "Output sum vector");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



int main(){
    const int n = 3;
    float h_arr1[n * n] = {10.12, 20.4, 30.5, 
                            10.0, 20.2, 40.6,
                            10.7, 20.1, 50.8};

    float h_arr2[n * n] = {1.1, 2.2, 3.3, 
                            4.4, 1.1, 2.2,
                            3.3, 4.4, 8.8};
    float h_out[n * n] = {0};
    matrixAddition(h_out, h_arr1, h_arr2, n);
}   