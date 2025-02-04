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
void initializeMatrices(float* A, float* B, int M, int K, int N) {
    for(int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = rand() / (float)RAND_MAX;
}

void testTiledMatrixMul() {
    int size = 4;
    int matrixSize = size * size * sizeof(float);

    float h_M[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_N[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_P[size * size];

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, matrixSize);
    cudaMalloc((void**)&d_N, matrixSize);
    cudaMalloc((void**)&d_P, matrixSize);

    cudaMemcpy(d_M, h_M, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(size / TILE_WIDTH, size / TILE_WIDTH);

    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, size);

    cudaMemcpy(h_P, d_P, matrixSize, cudaMemcpyDeviceToHost);

    printArray(h_M, size, size, "Matrix M");
    printArray(h_N, size, size, "Matrix N");
    printArray(h_P, size, size, "Matrix P");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main() {
    testTiledMatrixMul();
    return 0;
}