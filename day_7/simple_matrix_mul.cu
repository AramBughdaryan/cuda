#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

#define WIDTH 2

// Kernel to calculate P = M * N where M and N are square matrices
__global__ void simpleMatrixMulKernel(float* M, float* N, float* P, int width){
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    int Col = blockDim.y * blockIdx.y + threadIdx.y;

    if ((Row < width) && (Col < width)){
        float Pvalue = 0;
        for (int i = 0; i < width; i++){
            // Since here we access element of M and one element of N
            // and we apply one addition and one multiplication.
            // compute to global-memory-access ratio is 1. 
            // But we need to increase this ratio!
            Pvalue += M[Row * width + i] * N[i * width + Col];
        }
        P[Row * width + Col] = Pvalue;
    }
}


int main() {
    float h_M[WIDTH * WIDTH] = {1, 2, 3, 4};
    float h_N[WIDTH * WIDTH] = {5, 6, 7, 8};
    float h_P[WIDTH * WIDTH];

    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void**)&d_N, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, WIDTH * WIDTH * sizeof(float));

    cudaMemcpy(d_M, h_M, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(WIDTH, WIDTH);
    dim3 dimGrid(1, 1);

    simpleMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, WIDTH);

    cudaMemcpy(h_P, d_P, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    printArray(h_M, WIDTH, WIDTH, "matrix M");
    printArray(h_N, WIDTH, WIDTH, "matrix N");
    printArray(h_P, WIDTH, WIDTH, "matrix P");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}