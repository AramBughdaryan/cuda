#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

#define TILE_WIDTH 2

void initializeMatrices(float* A, float* B, int M, int K, int N);


// P = M * N
__global__ void tiledMatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Saved as automatic variables thus in registers. 
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // As we declare this variable as automatic it will be private for each thread!
    float Pvalue = 0;
    for (int tile = 0; tile < Width / TILE_WIDTH; ++tile){
        // Collaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row * Width + tile * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(tile * TILE_WIDTH + threadIdx.y) * Width + Col]; 
        __syncthreads();

        for (int k =0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}
