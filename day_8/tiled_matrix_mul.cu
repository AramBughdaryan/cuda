#include <iostream>
#include <cuda.h>
#include "../helpers/cuda_helpers.h"

#define TILE_WIDTH 2

// P = M * N
__global__ void tiledMatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph){
        Mds[ty][tx] = d_M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[ph * TILE_WIDTH + ph * Width + Col];
    }

}