#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

__global__ void matrixAdditionKernel(float* out, float* inp_1, float* inp_2, int n){
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    int Col = blockDim.y * blockIdx.y + threadIdx.y;

    if (Row < n && Col < n ){
        out[n * Row + Col] = inp_1[n * Row + Col] + inp_2[n * Row + Col];
    }
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
    float* d_arr1;
    float* d_arr2;
    float* d_out;

    int size = n * n * sizeof(float);

    printArray(h_arr1, n, n, "Array 1");
    printArray(h_arr2, n, n, "Array 1");

    checkCudaError(cudaMalloc((void**)&d_arr1, size), "Allocate memory for arr1");
    checkCudaError(cudaMalloc((void**)&d_arr2, size), "Allocate memory for arr2");
    checkCudaError(cudaMalloc((void**)&d_out, size), "Allocate memory for output arr");

    checkCudaError(cudaMemcpy(d_arr1, h_arr1, size, cudaMemcpyHostToDevice), "Copy array 1 from host to device");
    checkCudaError(cudaMemcpy(d_arr2, h_arr2, size, cudaMemcpyHostToDevice), "Copy array 2 from host to device");

    dim3 blockSize(n, n);
    dim3 gridDim(ceil(n/blockSize.x), ceil(n/blockSize.y));
    matrixAdditionKernel<<<gridDim, blockSize>>>(d_out, d_arr1, d_arr2, n);

    checkCudaError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "Copy output array from device to host");
    printArray(h_out, n, n, "Output sum vector");



}   