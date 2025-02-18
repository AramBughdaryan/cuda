#include <cuda.h>
#include <iostream>
#include <chrono>

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int maskWidth, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (maskWidth/2);
    for (int j = 0; j < maskWidth; j++){
        if (N_start_point + j >= 0 && N_start_point + j < Width){
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}


void test_convolution_1D_basic_kernel() {
    int Width = 10;
    int maskWidth = 3;
    float h_N[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_M[3] = {1, 0, -1};
    float h_P[10] = {0};

    float *d_N, *d_M, *d_P;
    cudaMalloc(&d_N, Width * sizeof(float));
    cudaMalloc(&d_M, maskWidth * sizeof(float));
    cudaMalloc(&d_P, Width * sizeof(float));

    cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, maskWidth * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (Width + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    convolution_1D_basic_kernel<<<numBlocks, blockSize>>>(d_N, d_M, d_P, maskWidth, Width);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (int i = 0; i < Width; i++) {
        std::cout << h_P[i] << " ";
    }
    std::cout << std::endl;

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
}

int main() {
    test_convolution_1D_basic_kernel();
    return 0;
}