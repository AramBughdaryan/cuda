#include <iostream>
#include <cuda.h>
#include <chrono>

#include "../helpers/cuda_helpers.h"


#define MAX_MASK_WIDTH 5
#define TILE_SIZE 16
__constant__ float M[MAX_MASK_WIDTH];


__global__ void convolution_1D_tiled_kernel(float *N, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1]; // N_ds[8]

    int n = Mask_Width/2; // 2

    int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x; 

    if (threadIdx.x >= blockDim.x - n) {
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    } // fills 0, 1

    N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x]; // fills 2, 3, 4, 5

    int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;

    if (threadIdx.x < n) {
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
    } // fills 7, 8

    __syncthreads();

    float Pvalue = 0;
    for(int j = 0; j < Mask_Width; j++) {
        Pvalue += N_ds[threadIdx.x + j]*M[j];
    }
    P[i] = Pvalue;
}


void test_convolution_1D_basic_kernel() {
    int Width = 4096 * 128 * 2;

    float *h_N = (float*)malloc(Width * sizeof(float));

    initializeMatrix<float>(h_N, ceil(Width) / 4, 4);
    float h_M[5] = {0.5, 1.05, -1.2, 2.5, -2.02};
    float h_P[Width] = {0};

    float *d_N, *d_P;
    cudaMalloc(&d_N, Width * sizeof(float));
    cudaMalloc(&d_P, Width * sizeof(float));

    cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_M, MAX_MASK_WIDTH * sizeof(float));

    int blockSize = 256;
    int numBlocks = (Width + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    convolution_1D_tiled_kernel<<<numBlocks, blockSize>>>(d_N, d_P, MAX_MASK_WIDTH, Width);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    // std::cout << "Output: ";
    // for (int i = 0; i < Width; i++) {
    //     std::cout << h_P[i] << " ";
    // }
    // std::cout << std::endl;

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    cudaFree(d_N);
    cudaFree(d_P);
}

int main() {
    test_convolution_1D_basic_kernel();
    return 0;
}