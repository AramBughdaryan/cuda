#include <iostream>
#include <cuda.h>
#include <cassert>
#include "../helpers/cuda_helpers.h"

void matrixAdditionKernelByColumn(float* C, const float* A, const float* B, int N);

__global__ void matrixAdditionKernelByColumn(float* out, float* inp_1, float* inp_2, int n){
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (colIdx < n ){
        for (int i = 0; i < n; i++){
            int idx = n * i + colIdx;
            out[idx] = inp_1[idx] + inp_2[idx];
        }
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
    dim3 threadsPerBlock(blockSize, 1);
    dim3 gridDim((N + blockSize - 1)/threadsPerBlock.x, 1);

    matrixAdditionKernelByColumn<<<gridDim, blockSize>>>(d_C, d_A, d_B, N);

    checkCudaError(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost), "Copy output array from device to host");
    printArray(C, N, N, "Output sum vector");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void testMatrixAddition() {
    const int n = 3;
    float h_arr1[n * n] = {10.12, 20.4, 30.5, 
                            10.0, 20.2, 40.6,
                            10.7, 20.1, 50.8};

    float h_arr2[n * n] = {1.1, 2.2, 3.3, 
                            4.4, 1.1, 2.2,
                            3.3, 4.4, 8.8};

    float expected[n * n] = {11.22, 22.6, 33.8, 
                             14.4, 21.3, 42.8,
                             14.0, 24.5, 59.6};

    float h_out[n * n] = {0};
    matrixAddition(h_out, h_arr1, h_arr2, n);

    for (int i = 0; i < n * n; i++) {
        assert(fabs(h_out[i] - expected[i]) < 1e-5);
    }

    std::cout << "Test passed!" << std::endl;
}

int main() {
    testMatrixAddition();
    return 0;
}