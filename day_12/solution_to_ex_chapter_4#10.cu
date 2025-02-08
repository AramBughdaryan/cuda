// To manipulate tiles, a new CUDA programmer has written the following
// device kernel, which will transpose each tile in a matrix. The tiles are of
// size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A
// is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code
// are shown below. BLOCK_WIDTH is known at compile time, but could be set
// anywhere from 1 to 20.

// Solution A.

// will work only when BLOCK_WIDTH == BLOCK_SIZe

#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

#define BLOCK_WIDTH 3

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
    
    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads();
    
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

int main(){
    int A_width = 3;
    int A_height = 6;
    int matrixSize = A_width * A_height * sizeof(float);
    float *A = (float*)malloc(matrixSize);
    std::cout<<A<<std::endl;
    std::cout<<&A<<std::endl;
    initializeMatrix(A, A_height, A_width);
    printArray(A, A_width, A_height, "A");


    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim((A_width + blockDim.x - 1)/blockDim.x,(A_height + blockDim.y - 1)/blockDim.y);

    float *d_A;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice);

    BlockTranspose<<<gridDim, blockDim>>>(d_A, A_width, A_height);

    cudaMemcpy(A, d_A, matrixSize, cudaMemcpyDeviceToHost);
    printArray(A, A_width, A_height, "A with transposed TILES");

    cudaFree(d_A);
    free(A);
}