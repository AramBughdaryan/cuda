#include <iostream>
#include <cuda.h>
#include <cassert>
#include "../helpers/cuda_helpers.h"

// If debug print's are not needed uncomment this line
#define DEBUG_PRINT

__device__ int numCalls = 0;


void matrixMulVector(float* A, float* B, float* C, int N);

__global__ void matrixMulVectorKernel(float* d_A, float* d_B, float* d_C, int N){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    atomicAdd(&numCalls, 1);
    
    if (idx < N){
        float el_sum = 0;
        for (int i = 0; i < N; i++){
            el_sum += d_C[i] * d_B[idx * N + i];
        }
        d_A[idx] = el_sum;
    }
}


void matrixMulVector(float* A, float* B, float* C, int N){
    size_t sizeB = N * N * sizeof(float);
    size_t sizeVector = N * sizeof(float);
    int h_count;
    

    float *d_A, *d_B, *d_C;
    
    checkCudaError(cudaMalloc(&d_A, sizeVector), "Allocate memory on device for A");
    checkCudaError(cudaMalloc(&d_C, sizeVector), "Allocate memory on device for C");
    checkCudaError(cudaMalloc(&d_B, sizeB), "Allocate memory on device for B");

    checkCudaError(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice), "Copy B to d_B");
    checkCudaError(cudaMemcpy(d_C, C, sizeVector, cudaMemcpyHostToDevice), "Copy C to d_C");

    const int blockSize = 8;
    dim3 threadsPerBlock(blockSize);
    dim3 gridDim((N + blockSize - 1)/ blockSize);
    matrixMulVectorKernel<<<gridDim, threadsPerBlock>>>(d_A, d_B, d_C, N);

    checkCudaError(cudaMemcpy(A, d_A, sizeVector, cudaMemcpyDeviceToHost), "Copy A from device to Host");
    #ifdef DEBUG_PRINT
        printArray(B, N, N, "B");
        printArray(C, N, 1, "C");
        printArray(A, N, 1, "A");
    #endif
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    cudaMemcpyFromSymbol(&h_count, numCalls, sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("Number of kernel calls: %d\n", h_count);

}

void testMatrixMulVector() {
    std::cout<<"Calculate A = B * C where A and C are vectors and B is matrix"<<std::endl;
    int N = 4;
    float A[N] = {0.0};
    float B[N * N] = {  11.22, 22.6, 33.8, 12.05,
                        14.4, 21.3, 42.8, 11.4,
                        14.0, 24.5, 59.6, 9.17, 
                        1.1, 4.7, 7.8, 9.01};
    float C[N] = {1.5, 4.6, 7.8, 1.2};
    float expectedA[N] = {  11.22 * 1.5 + 22.6 * 4.6 + 33.8 * 7.8 + 12.05 * 1.2,
                            14.4 * 1.5 + 21.3 * 4.6 + 42.8 * 7.8 + 11.4 * 1.2,
                            14.0 * 1.5 + 24.5 * 4.6 + 59.6 * 7.8 + 9.17 * 1.2,
                            1.1 * 1.5 + 4.7 * 4.6 + 7.8 * 7.8 + 9.01 * 1.2 
                        };

    matrixMulVector(A, B, C, N);
    #ifdef DEBUG_PRINT
        printArray(expectedA, N, 1, "Expected A");
    #endif

    for (int i = 0; i < N; i++) {
        assert(fabs(A[i] - expectedA[i]) < 1e-4);
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    testMatrixMulVector();
    return 0;
}