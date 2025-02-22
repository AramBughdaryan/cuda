#include <cuda.h>
#include <iostream>
#include <chrono>

#include "../helpers/cuda_helpers.h"

#define MAX_MASK_WIDTH 5
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution_1D_basic_kernel(float *N, float *P, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (MAX_MASK_WIDTH/2);
    for (int j = 0; j < MAX_MASK_WIDTH; j++){
        if (N_start_point + j >= 0 && N_start_point + j < Width){
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}


void test_convolution_1D_basic_kernel() {
    int Width = 4096 * 32;

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
    convolution_1D_basic_kernel<<<numBlocks, blockSize>>>(d_N, d_P, Width);
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