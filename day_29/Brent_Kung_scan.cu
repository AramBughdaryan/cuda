#include <iostream>
#include <cuda.h>

#define SECTION_SIZE 2048

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) 
}
