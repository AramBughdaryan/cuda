#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

// Kernel to calculate P = M * N where M and N are square matrices
__global__ void matrixMulKernel(float* M, float* N, float* P, int width){
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
