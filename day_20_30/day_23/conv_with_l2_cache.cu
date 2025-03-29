#include <iostream>
#include <cuda.h>
#include <chrono>

#include "../helpers/cuda_helpers.h"


#define MAX_MASK_WIDTH 5
#define TILE_SIZE 16
__constant__ float M[MAX_MASK_WIDTH];


__global__ void convolution_1D_tiled_caching_kernel(float *N, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float N_ds[TILE_SIZE]; // N_ds[5]
    
    N_ds[threadIdx.x] = N[i];

    __syncthreads();

    int This_tile_start_point = blockIdx.x * blockDim.x;
    int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width/2);
    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++){
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width){
            if (N_index >= This_tile_start_point && N_index < Next_tile_start_point){
                Pvalue += N_ds[threadIdx.x + j - (Mask_Width/2)] * M[j];
            }
            else{
                Pvalue += N[N_index] * M[j];
            }
        }
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
    convolution_1D_tiled_caching_kernel<<<numBlocks, blockSize>>>(d_N, d_P, MAX_MASK_WIDTH, Width);
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