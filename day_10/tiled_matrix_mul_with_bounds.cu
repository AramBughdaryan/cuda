#include <iostream>
#include <cuda.h>
#include <cassert>
#include "../helpers/cuda_helpers.h"

#define TILE_WIDTH 16

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
    for (int phase = 0; phase < Width / (float)TILE_WIDTH; ++phase){
        // Collaborative loading of d_M and d_N tiles into shared memory
        if (Row < Width && phase * TILE_WIDTH + tx < Width)
        {
            Mds[ty][tx] = d_M[Row * Width + phase * TILE_WIDTH + tx];
        }
        else{
            Mds[ty][tx] = 0.0f;
        }

        if (Col < Width && phase * TILE_WIDTH + ty < Width){
            Nds[ty][tx] = d_N[(phase * TILE_WIDTH + ty) * Width + Col]; 
        }
        else{
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k =0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (Row < Width && Col < Width){
        d_P[Row * Width + Col] = Pvalue;
    }
}

__global__ void simpleMatrixMulKernel(float* M, float* N, float* P, int width){
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

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


void testTiledMatrixMul() {
    int size = 512;
    int matrixSize = size * size * sizeof(float);
    float *h_M = (float*)malloc(matrixSize);
    float *h_N = (float*)malloc(matrixSize);

    initializeMatrices(h_M, h_N, size, size, size);

    float h_P_1[size * size];
    float h_P_2[size * size];

    float *d_M, *d_N, *d_P_1, *d_P_2;

    cudaMalloc((void**)&d_M, matrixSize);
    cudaMalloc((void**)&d_N, matrixSize);

    cudaMalloc((void**)&d_P_1, matrixSize);
    cudaMalloc((void**)&d_P_2, matrixSize);

    cudaMemcpy(d_M, h_M, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Timing for tiledMatrixMulKernel
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_1, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiled Matrix Multiplication Time: " << elapsedTime << " ms" << std::endl;

    cudaDeviceSynchronize();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Timing for simpleMatrixMulKernel
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    simpleMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_2, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Simple Matrix Multiplication Time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_P_1, d_P_1, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_2, d_P_2, matrixSize, cudaMemcpyDeviceToHost);

    // uncomment to print arrays
    // printArray(h_M, size, size, "Matrix M");
    // printArray(h_N, size, size, "Matrix N");
    // printArray(h_P_1, size, size, "Matrix P_1 TILED");
    // printArray(h_P_2, size, size, "Matrix P_2 NAIVE");


    for (int i = 0; i < size * size; i++) {
        assert(fabs(h_P_1[i] - h_P_2[i]) < 1e-4);
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P_1);
    cudaFree(d_P_2);

    free(h_M); free(h_N);
}

int main() {
    testTiledMatrixMul();
    return 0;
}
